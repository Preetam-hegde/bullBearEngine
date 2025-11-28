from typing import TypedDict, Annotated, List, Literal
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from mcp_interface import get_market_data_tool, generate_market_chart_tool, generate_performance_analysis_tool, get_technical_analysis_tool, get_price_prediction_tool, get_company_info_tool, generate_separate_charts_tool
import chainlit as cl
import plotly.io as pio
import json
import os
import pandas as pd

from dotenv import load_dotenv

load_dotenv()

# Helper to get currency symbol
def get_currency_symbol(currency_code: str) -> str:
    symbols = {
        'USD': '$', 'EUR': '€', 'GBP': '£', 'INR': '₹', 'JPY': '¥',
        'CNY': '¥', 'AUD': 'A$', 'CAD': 'C$', 'SGD': 'S$'
    }
    return symbols.get(currency_code, currency_code + ' ')

# Define Tools
@tool
async def fetch_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetches historical market data for a ticker."""
    result = await get_market_data_tool(ticker, period, interval)
    if isinstance(result, tuple):
        data, info = result
        # Convert DataFrame to JSON list of records, preserving date
        return data.reset_index().to_json(orient='records', date_format='iso')
    return result

@tool
async def analyze_market(ticker: str, period: str = "1y", interval: str = "1d"):
    """
    Performs comprehensive technical analysis on a stock.
    Returns AI-generated insights, key metrics (RSI, MACD, etc.), and volume analysis.
    Use this for 'detailed analysis', 'technical analysis', or when asked for advice.
    """
    json_str = await get_technical_analysis_tool(ticker, period, interval)
    try:
        data = json.loads(json_str)
        if "error" in data:
            return f"Error: {data['error']}"
            
        # Format the output for the LLM to interpret and present
        insights = "\n".join([f"- {i}" for i in data.get("insights", [])])
        metrics = data.get("metrics", {})
        
        summary = f"**Analysis for {data.get('symbol')}**\n\n"
        summary += f"**Key Insights:**\n{insights}\n\n"
        summary += "**Technical Metrics:**\n"
        for k, v in metrics.items():
            if v is not None:
                if isinstance(v, float):
                    summary += f"- {k}: {v:.2f}\n"
                else:
                    summary += f"- {k}: {v}\n"
                    
        return summary
    except Exception as e:
        return f"Failed to analyze market: {str(e)}"

@tool
async def predict_price(ticker: str, period: str = "5y", interval: str = "1d"):
    """
    Uses Machine Learning (Random Forest) to predict the next day's closing price.
    Returns the predicted price, confidence score (test accuracy), and feature importance.
    """
    json_str = await get_price_prediction_tool(ticker, period, interval)
    try:
        data = json.loads(json_str)
        if "error" in data:
            return f"Error: {data['error']}"
            
        ensemble = data.get("ensemble_prediction", {})
        if "error" in ensemble:
            return f"Error in prediction: {ensemble['error']}"
            
        prediction = ensemble.get("prediction")
        current = data.get("current_price")
        
        if prediction is None or current is None:
            return "Error: Could not retrieve valid prediction data."
            
        change = ensemble.get("predicted_change_pct")
        if change is None and current:
            change = ((prediction - current) / current) * 100
            
        confidence = ensemble.get("consensus_strength", 0)
        
        # Try to infer currency from symbol or just use generic
        # Ideally we'd pass currency info, but for now let's assume standard formatting
        # or fetch info to get currency.
        # For speed, we'll check if it ends in .NS or .BO for INR
        currency = "$"
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            currency = "₹"
        
        summary = f"**ML Price Prediction for {ticker}**\n\n"
        summary += f"**Next Day Prediction:** {currency}{prediction:.2f} ({change:+.2f}%)\n"
        summary += f"**Current Price:** {currency}{current:.2f}\n"
        summary += f"**Ensemble Consensus:** {confidence:.1f}%\n"
        
        best_model = data.get("best_model", {})
        if best_model.get("name"):
            summary += f"**Best Model:** {best_model.get('name')} (Accuracy: {best_model.get('directional_accuracy', 0):.1f}%)\n\n"
        else:
            summary += "\n"
        
        if confidence < 60:
            summary += "⚠️ **Warning:** Model consensus is low. Treat this prediction with caution.\n"
            
        summary += f"\n*{data.get('disclaimer', '')}*"
            
        return summary
    except Exception as e:
        return f"Failed to predict price: {str(e)}"

@tool
async def get_company_info(ticker: str):
    """Fetches company profile, sector, industry, and key financial metrics."""
    json_str = await get_company_info_tool(ticker)
    try:
        data = json.loads(json_str)
        if "error" in data:
            return f"Error: {data['error']}"
            
        details = data.get("details", {})
        financials = data.get("financials", {})
        
        # Get currency from financials if possible, or infer
        currency = "$"
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            currency = "₹"
            
        summary = f"**Company Profile: {data.get('symbol')}**\n\n"
        summary += f"**Name:** {details.get('Company Name')}\n"
        summary += f"**Sector:** {details.get('Sector')} | **Industry:** {details.get('Industry')}\n"
        summary += f"**Summary:** {details.get('Summary')[:300]}...\n\n"
        
        summary += "**Key Financials:**\n"
        for k, v in financials.items():
            if v is not None and v != 'N/A':
                # Format currency fields
                if k in ["52W High", "52W Low", "Market Cap", "Price to Book"]: # Heuristic
                     val_str = f"{currency}{v:,.2f}" if isinstance(v, (int, float)) else str(v)
                     summary += f"- {k}: {val_str}\n"
                else:
                    if isinstance(v, float):
                         summary += f"- {k}: {v:.2f}\n"
                    else:
                         summary += f"- {k}: {v}\n"
                     
        return summary
    except Exception as e:
        return f"Failed to get company info: {str(e)}"

@tool
async def show_data(ticker: str, period: str = "1mo", interval: str = "1d"):
    """Fetches market data and displays separate charts (Price, Volume, MACD, RSI) to the user."""
    json_str = await generate_separate_charts_tool(ticker, period, interval)
    try:
        # Check if it's an error message
        data = json.loads(json_str)
        if "error" in data:
             return f"Error generating chart: {data['error']}"
        
        # data is a dict of chart_name -> chart_json
        elements = []
        for name, chart_json in data.items():
            fig = pio.from_json(chart_json)
            # Ensure layout is good
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark")
            elements.append(cl.Plotly(figure=fig, name=f"{ticker}_{name}"))
            
        await cl.Message(content=f"Here are the technical charts for {ticker}", elements=elements).send()
        return f"Charts for {ticker} displayed."
    except Exception as e:
        return f"Failed to display chart: {str(e)}"

@tool
async def show_performance(ticker: str, period: str = "1y", interval: str = "1d"):
    """Fetches and displays performance metrics and a chart for a ticker."""
    json_str = await generate_performance_analysis_tool(ticker, period, interval)
    try:
        data = json.loads(json_str)
        if "error" in data and len(data) == 1:
             return f"Error: {data['error']}"
        
        metrics = data.get("metrics", {})
        chart_json = data.get("chart", "{}")
        
        # Format metrics as markdown table
        metrics_md = "| Metric | Value |\n|---|---|\n"
        for k, v in metrics.items():
            metrics_md += f"| {k} | {v} |\n"
            
        elements = []
        if chart_json:
            fig = pio.from_json(chart_json)
            # Optimize layout
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            elements.append(cl.Plotly(figure=fig, name=f"{ticker}_perf_chart"))
            
        await cl.Message(content=f"Performance Metrics for {ticker}:\n\n{metrics_md}", elements=elements).send()
        return f"Performance metrics for {ticker} displayed."
    except Exception as e:
        return f"Failed to display performance: {str(e)}"

tools = [fetch_data, analyze_market, predict_price, get_company_info, show_data, show_performance]

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# Define Graph
def create_graph(model_provider="Gemini", model_name="gemini-2.0-flash"):
    llm = None
    
    if model_provider == "Ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0
        )
    else:
        # Initialize LLM (Ensure GOOGLE_API_KEY is set in environment)
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0,
            google_api_key=os.environ["GOOGLE_API_KEY"]
        )
        
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
    settings = cl.user_session.get("settings")
    if settings is None:
        settings = {}
    provider = settings.get("ModelProvider", "Gemini")
    model = settings.get("ModelName", "gemini-2.0-flash")
    
    graph = create_graph(provider, model)
    
    # System prompt to guide the agent
    system_prompt = SystemMessage(content="""You are a helpful financial assistant powered by BullBear Engine.
    Your goal is to assist users with market analysis, charts, and performance metrics.
    
    Capabilities:
    - **Market Data**: Fetch historical data using `fetch_data`.
    - **Charts**: Display interactive price charts with indicators using `show_data`.
    - **Performance**: Show return metrics (Sharpe, Volatility, Drawdown) using `show_performance`.
    - **Analysis**: Perform deep technical analysis (RSI, MACD, Trends, Insights) using `analyze_market`.
    - **Prediction**: Predict next day's price using ML with `predict_price`.
    - **Company Info**: Get company details and fundamentals using `get_company_info`.
    
    Rules:
    1. **Context is Key**: If the user asks for analysis but doesn't provide a symbol, check the conversation history. If still missing, ASK for it.
    2. **Comprehensive Analysis Protocol**: If the user asks to "analyze" a stock (e.g., "Analyze AAPL", "Check MSFT", "How is GOOGL doing?"), you MUST provide a complete report by calling MULTIPLE tools:
       - First, call `get_company_info` to establish context.
       - Second, call `show_data` to visualize the price action.
       - Third, call `analyze_market` to provide technical insights.
       - Fourth, call `show_performance` to show historical returns.
       - Do NOT ask for permission to run these; just run them to provide a "wow" experience.
    3. **Prediction**: Use `predict_price` ONLY when explicitly asked for a prediction, forecast, or "future price". Warn the user that it's an estimate.
    4. **Parameters**: Default to '1mo' period and '1d' interval unless specified otherwise. For performance and analysis, default to '1y'. For prediction, default to '5y'.
    5. **Error Handling**: If a tool returns an error, explain it clearly to the user.
    6. **Currency**: Be mindful of the currency. For Indian stocks (ending in .NS or .BO), use '₹'.
    """)
    
    messages = [system_prompt] + history + [HumanMessage(content=user_input)]
    inputs = {"messages": messages}
    
    final_content = ""
    tool_steps = {}
    
    async with cl.Step(name="BullBear Agent", type="run") as run_step:
        async for event in graph.astream(inputs):
            for key, value in event.items():
                if key == "agent":
                    msg = value["messages"][0]
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            # Create and send the step
                            step = cl.Step(name=tool_call["name"], type="tool")
                            step.language = "json"
                            step.input = tool_call["args"]
                            await step.send()
                            # Store reference
                            tool_steps[tool_call["id"]] = step
                    else:
                        final_content = msg.content
                        
                elif key == "tools":
                    # value["messages"] is a list of ToolMessages
                    for tool_msg in value["messages"]:
                        # Retrieve the step
                        step = tool_steps.get(tool_msg.tool_call_id)
                        if step:
                            # Update the step with the result
                            # Truncate output if too long for display, but keep enough context
                            content = str(tool_msg.content)
                            if len(content) > 2000:
                                step.output = content[:2000] + "... (truncated)"
                            else:
                                step.output = content
                            await step.update()
                        else:
                            # Fallback if we missed the start (shouldn't happen)
                            async with cl.Step(name=f"{tool_msg.name} Output", type="tool_result") as result_step:
                                result_step.output = str(tool_msg.content)
                            
    return final_content
