from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp_interface import get_market_data_tool, fit_market_curve_tool, detect_sr_zones_tool
import os

# Define Tools
@tool
async def fetch_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetches historical market data for a ticker."""
    return await get_market_data_tool(ticker, period, interval)

@tool
async def analyze_trend(data: str):
    """Analyzes the market trend (uptrend, downtrend, ranging) from market data JSON string."""
    return await fit_market_curve_tool(data)

@tool
async def find_sr(data: str):
    """Finds support and resistance levels from market data JSON string."""
    return await detect_sr_zones_tool(data)

tools = [fetch_data, analyze_trend, find_sr]

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define Graph
def create_graph():
    # Initialize LLM (Ensure GOOGLE_API_KEY is set in environment)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = state['messages']
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))

    workflow.set_entry_point("agent")
    
    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        if last_message.tool_calls:
            return "tools"
        return END

    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile()

async def run_agent(user_input: str, history: List[BaseMessage]):
    graph = create_graph()
    inputs = {"messages": history + [HumanMessage(content=user_input)]}
    
    async for event in graph.astream(inputs):
        for key, value in event.items():
            # You can log steps here
            pass
            
    # Return the final message content
    # Note: This is a simplified return. In a real app, you'd stream the output.
    # We need to capture the final state.
    # Since astream yields updates, we can't easily get the final state without accumulating.
    # Let's use invoke for simplicity in this scaffolding or just return the last message from the event if possible.
    
    # Re-running invoke to get final result for simplicity
    final_state = await graph.ainvoke(inputs)
    return final_state['messages'][-1].content
