import chainlit as cl
from agent import run_agent
from langchain_core.messages import HumanMessage, AIMessage

@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="Welcome to BullBear Engine! I can analyze market data for you. Try asking 'Analyze AAPL'.").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("history")
    
    msg = cl.Message(content="")
    await msg.send()
    
    # Call the agent
    try:
        response_content = await run_agent(message.content, history)
        msg.content = response_content
        await msg.update()
        
        # Update history
        history.append(HumanMessage(content=message.content))
        history.append(AIMessage(content=response_content))
        cl.user_session.set("history", history)
        
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
