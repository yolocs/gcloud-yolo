from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic.v1 import BaseModel, Field
from langchain_core.agents import AgentFinish
from langchain_core.tools import Tool, BaseTool
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.markdown import Markdown
from typing import List, Tuple, Dict, Any
import subprocess
import re
import logging
import argparse
import os
import yaml

parser = argparse.ArgumentParser(description="Google Cloud Debugging Agent")
parser.add_argument(
    "-l",
    "--loglevel",
    default="INFO",
    help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
parser.add_argument(
    "--model",
    default="gemini-2.0-flash-exp",
    help="All Gemini models: gemini-1.5-pro-002, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-1219",
)
parser.add_argument(
    "--project",
    default="gochen",
    help="The project where the Gemini is hosted",
)
parser.add_argument(
    "--location",
    default="us-central1",
    help="The location where the Gemini is hosted",
)
parser.add_argument(
    "--tinytools",
    default="./tinytools",
    help="The directory contains all the 'tiny tools'",
)
args = parser.parse_args()

console = Console()
logging.basicConfig(level=logging.getLevelNamesMapping()[args.loglevel.upper()])

# Not that useful. Limit the results.
search = GoogleSearchAPIWrapper(k=2)
search_tool = Tool(
    name="search-gcp-documentation",
    func=lambda q: search.run(q + " site:cloud.google.com"),
    description="Use this tool to search GCP documentation",
)


def load_tinytools() -> List[Tool]:
    tinytools_dir = args.tinytools
    if not os.path.exists(tinytools_dir):
        raise ValueError(f"Tinytools directory '{tinytools_dir}' does not exist.")

    tools = []
    for filename in os.listdir(tinytools_dir):
        if filename.endswith(".yaml"):
            filepath = os.path.join(tinytools_dir, filename)
            with open(filepath, "r") as f:
                try:
                    tool_config = yaml.safe_load(f)
                    tool_name = tool_config.get("name")
                    tool_description = tool_config.get("description")
                    tool_instruction = tool_config.get("instruction")
                    if not tool_name or not tool_description or not tool_instruction:
                        logging.warning(
                            f"Skipping incomplete tool definition in {filepath}"
                        )
                        continue
                    tool_func = (
                        lambda q, instruction=tool_instruction: instruction.format(
                            input=q
                        )
                    )
                    tools.append(
                        Tool(
                            name=tool_name,
                            description=tool_description,
                            func=tool_func,
                        )
                    )
                except yaml.YAMLError as e:
                    logging.error(f"Error parsing YAML file {filepath}: {e}")

    for t in tools:
        logging.debug(
            f"Loaded tool: {t.name}, {t.description} - sanity check instruction: {t.func('test')}"
        )

    return tools


llm = ChatVertexAI(
    model=args.model,
    project=args.project,
    location=args.location,
)


# Define the state of the agent
class AgentState(BaseModel):
    messages: List[dict] = Field(default_factory=list)
    intermediate_steps: List[Tuple[Any, str]] = Field(default_factory=list)
    user_query: str = Field(default="")
    next: str = Field(default="")


# Define tools for the agent (improved GCloudTool)
class GCloudTool(BaseTool):
    name: str = "gcloud"
    description: str = (
        "Use this tool to interact with Google Cloud using the gcloud CLI. All commands are read-only. Example usage: `gcloud run services describe SERVICE_NAME --format=json`"
    )

    def _run(self, command: str) -> str:
        logging.debug(f"Received command: {command}")

        unsafe_commands = ["delete", "create", "update", "patch"]
        if any(cmd in command for cmd in unsafe_commands):
            return "gcloud command failed: cannot execute gcloud command; only read-only command is supported"

        if not command.startswith("gcloud "):
            command = "gcloud " + command

        try:
            process = subprocess.run(
                command, shell=True, capture_output=True, text=True, check=False
            )

            logging.debug(f"Raw gcloud command output: {process.stdout}")

            if process.returncode != 0:
                error_message = process.stderr.strip()
                # Improved error parsing to extract specific error types
                if "INVALID_ARGUMENT" in error_message:
                    error_type = "Invalid argument"
                elif "NOT_FOUND" in error_message:
                    error_type = "Resource not found"
                elif "PERMISSION_DENIED" in error_message:
                    error_type = "Permission denied"
                else:
                    error_type = "General gcloud error"
                return f"gcloud command failed with {error_type}: {error_message}"

            if process.stdout.strip() == "":
                return "gcloud command executed successfully but returned empty output."

            return process.stdout

        except Exception as e:
            return f"gcloud command failed: {e}"

    async def _arun(self, command: str) -> str:
        raise NotImplementedError


class AskForClarification(BaseTool):
    name: str = "ask_for_clarification"
    description: str = (
        "Use this tool to ask clarifying questions to the user when more information is needed to fulfill the request. The agent will PAUSE and wait for user input."
    )

    def _run(self, question: str) -> str:
        print(f"\n{question}\n")  # Print the question to the console
        user_input = input("Your response: ")  # Get user input
        return f"User clarified: {user_input}"

    async def _arun(self, question: str) -> str:
        raise NotImplementedError


# Initialize LLM and tools
tools = [GCloudTool(), AskForClarification(), search_tool] + load_tinytools()


def tool_desc():
    return "\n".join([f"Name: {t.name}, Description: {t.description}" for t in tools])


# Define the prompt template (improved to handle errors)
template = """You are a helpful AI assistant that helps debug Google Cloud workloads. You have access to the following tools:

{tools}

Always try to search the GCP documentation first to know the right tasks to do.

Cloud Asset Inventory is a powerful tool to find thorough information about any cloud resources, especially to analyze IAM and org policies.

If a gcloud command returns an error, carefully analyze the error message. If the error is due to a missing resource (NOT_FOUND), try to infer the correct resource name or ask for clarification. If the error is due to an invalid argument (INVALID_ARGUMENT), double-check the command syntax and arguments. If the error is due to permission denied, inform the user that they might not have the necessary permissions.

Respond in JSON format with two keys: `action` and `action_input`.

If you need to use a tool, set `action` to the tool's name and `action_input` to the input for the tool.

If you have finished, set `action` to `Final Answer` and `action_input` to your final answer.

Here are the previous conversation history:
{chat_history}

Here are the intermediate steps:
{intermediate_steps}

Here is the user query:
{user_query}
"""

prompt = ChatPromptTemplate.from_template(template)


def continue_next(state: AgentState):
    return state.next


def call_tool(state: AgentState):
    messages = state.messages
    intermediate_steps = state.intermediate_steps
    user_query = state.user_query

    # This function already handles json in markdown format.
    parsed = JSONAgentOutputParser().parse(messages[-1]["content"])

    if isinstance(parsed, AgentFinish):
        logging.debug("Agent finished")
        return {
            "messages": messages,
            "intermediate_steps": intermediate_steps,
            "user_query": user_query,
            "next": "end",
        }

    tool_name = parsed.tool
    tool_input = parsed.tool_input

    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found.")

    logging.debug(f"Calling tool: {tool_name} with input: {tool_input}")

    tool_output = tool.run(tool_input)

    logging.debug(f"Called tool: {tool_name} and got output {tool_output}")

    messages.append(
        {"role": "assistant", "content": f"Tool {tool_name} returned: {tool_output}"}
    )
    intermediate_steps.append((tool_name, tool_input))

    # Check if we asked a clarification question
    if tool_name == "ask_for_clarification":
        return {
            "messages": messages,
            "intermediate_steps": intermediate_steps,
            "user_query": user_query,
            "next": "wait_for_user",
        }
    elif "gcloud command failed" in tool_output:
        return {
            "messages": messages,
            "intermediate_steps": intermediate_steps,
            "user_query": user_query,
            "next": "handle_gcloud_error",
        }
    else:
        return {
            "messages": messages,
            "intermediate_steps": intermediate_steps,
            "user_query": user_query,
            "next": "generate_response",
        }


def wait_for_user(state: AgentState):
    return {
        "messages": state.messages,
        "intermediate_steps": state.intermediate_steps,
        "user_query": state.user_query,
        "next": "generate_response",
    }


def generate_response(state: AgentState):
    messages = state.messages
    intermediate_steps = state.intermediate_steps
    user_query = state.user_query

    prompt_value = prompt.format(
        tools=tool_desc(),
        chat_history=messages,
        intermediate_steps=intermediate_steps,
        user_query=user_query,
    )

    logging.debug(f"Generated prompt: {prompt_value}")

    response = llm.invoke(prompt_value)

    if isinstance(response.content, str):
        content = [response.content]
    else:
        content = response.content

    console.print(f"\n== Step ==\n{content[0]}")
    console.print("-" * 40)
    logging.debug(f"Markdown JSON from LLM response: {content[-1]}")

    messages.append({"role": "assistant", "content": "\n".join(content)})

    return {
        "messages": messages,
        "intermediate_steps": intermediate_steps,
        "user_query": user_query,
    }


def handle_gcloud_error(state: AgentState):
    messages = state.messages
    intermediate_steps = state.intermediate_steps
    user_query = state.user_query
    last_message = messages[-1]
    if "gcloud command failed" in last_message["content"]:
        # Extract error type and message
        error_match = re.search(
            r"gcloud command failed with (.+): (.+)", last_message["content"]
        )
        if error_match:
            error_type = error_match.group(1)
            error_message = error_match.group(2)
            messages.append(
                {
                    "role": "assistant",
                    "content": f"I encountered a {error_type} error: {error_message}. I will try to address this.",
                }
            )
        else:
            messages.append(
                {
                    "role": "assistant",
                    "content": "gcloud command failed with unknown error. I need to investigate further.",
                }
            )

    return {
        "messages": messages,
        "intermediate_steps": intermediate_steps,
        "user_query": user_query,
        "next": "generate_response",
    }


workflow = StateGraph(AgentState)
workflow.add_node("generate_response", generate_response)
workflow.add_node("call_tool", call_tool)
workflow.add_node("handle_gcloud_error", handle_gcloud_error)
workflow.add_node("wait_for_user", wait_for_user)

workflow.set_entry_point("generate_response")

workflow.add_edge("generate_response", "call_tool")
workflow.add_edge("handle_gcloud_error", "generate_response")
workflow.add_edge("wait_for_user", "generate_response")


workflow.add_conditional_edges(
    source="call_tool",
    path=continue_next,
    path_map={
        "generate_response": "generate_response",
        "wait_for_user": "wait_for_user",
        "handle_gcloud_error": "handle_gcloud_error",
        "end": END,
    },
)

app = workflow.compile()


def run_interactive_agent():
    console.print("== Welcome to the Google Cloud Debugging Agent! ==")
    past_answers = []

    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        state = AgentState(
            messages=[
                {"role": "user", "content": user_query},
                {
                    "role": "assistant",
                    "content": f"Knowledge I have learned from the past: {'\n'.join(past_answers)}",
                },
            ],
            user_query=user_query,
        )
        final_state = app.invoke(state)

        parsed = JSONAgentOutputParser().parse(
            final_state.get("messages")[-1]["content"]
        )
        last_answer = parsed.return_values["output"]
        past_answers.append(last_answer)

        console.print("\n== Final Answer ==")
        console.print(Markdown(last_answer))
        console.print("=" * 40)  # Separator for better readability


if __name__ == "__main__":
    run_interactive_agent()
