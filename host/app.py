import chainlit as cl
from agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_chat_start
async def start():
    settings = await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="ModelProvider",
                label="Model Provider",
                values=["Gemini", "Ollama"],
                initial_index=0,
            ),
            cl.input_widget.TextInput(
                id="ModelName",
                label="Model Name",
                initial="gemini-2.0-flash",
                description="e.g. gemini-2.0-flash, llama3, mistral"
            ),
        ]
    ).send()
    
    cl.user_session.set("settings", settings)
    cl.user_session.set("history", [])
    
    await cl.Message(content="Welcome to BullBear Engine! 🐂🐻\n\nI can analyze stocks, predict prices, and visualize market data.\n\nTry asking:\n- *Analyze RELIANCE.NS*\n- *Predict price for TATAMOTORS.NS*\n- *Show chart for HDFCBANK.NS*").send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    await cl.Message(content=f"Settings updated: Using {settings['ModelProvider']} - {settings['ModelName']}").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    
    # Call the agent
    try:
        response_content = await run_agent(message.content, history)
        await cl.Message(content=response_content).send()
        
        # Update history
        history.append(HumanMessage(content=message.content))
        history.append(AIMessage(content=response_content))
        cl.user_session.set("history", history)
        
    except Exception as e:
        await cl.Message(content=f"Error: {str(e)}").send()
