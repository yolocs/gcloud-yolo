from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic.v1 import BaseModel, Field
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_google_vertexai import VertexAI
from langgraph.graph import StateGraph, END
from typing import List, Tuple, Dict, Any
import subprocess
import json
import re


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
        unsafe_commands = ["delete", "create", "update", "patch", "set"]
        if any(cmd in command for cmd in unsafe_commands):
            return "gcloud command failed: cannot execute gcloud command; only read-only command is supported"

        try:
            full_command = f"gcloud {' '.join(command.split())} --format=json"
            process = subprocess.run(
                full_command, shell=True, capture_output=True, text=True, check=False
            )

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

            try:
                json.loads(process.stdout)
                return process.stdout
            except json.JSONDecodeError:
                return f"gcloud command output is not valid JSON:\n{process.stdout}"

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
llm = VertexAI(model="gemini-2.0-flash-exp", project="gochen", location="us-central1")
tools = [GCloudTool(), AskForClarification()]

# Define the prompt template (improved to handle errors)
template = """You are a helpful AI assistant that helps debug Google Cloud workloads. You can use the `gcloud` tool to interact with Google Cloud and the `ask_for_clarification` tool to ask clarifying questions.

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


# # Create the LangGraph
# def should_continue(state):
#     messages = state["messages"]
#     last_message = messages[-1]
#     if "Final Answer" in last_message["content"]:
#         return "end"
#     return "continue"


def continue_next(state: AgentState):
    return state.next


def call_tool(state: AgentState):
    messages = state.messages
    intermediate_steps = state.intermediate_steps
    user_query = state.user_query

    parsed = JSONAgentOutputParser().parse(messages[-1]["content"])
    # tool_name = parsed["action"]
    # tool_input = parsed["action_input"]
    tool_name = parsed.tool
    tool_input = parsed.tool_input

    if tool_name == "Final Answer":
        return {
            "messages": messages,
            "intermediate_steps": intermediate_steps,
            "user_query": user_query,
            "next": "end",
        }

    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found.")

    tool_output = tool.run(tool_input)
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
        # chat_history=format_log_to_str(messages[:-1]),
        chat_history=messages[:-1],
        intermediate_steps=intermediate_steps,
        user_query=user_query,
    )
    response = llm.invoke(prompt_value)
    messages.append({"role": "assistant", "content": response})
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
    print("Welcome to the Google Cloud Debugging Agent!")
    while True:
        user_query = input("\nEnter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break

        state = AgentState(
            messages=[{"role": "user", "content": user_query}], user_query=user_query
        )
        final_state = app.invoke(state)

        print("\nFinal Answer:")
        print(final_state.messages[-1]["content"])
        print("-" * 40)  # Separator for better readability


if __name__ == "__main__":
    run_interactive_agent()
