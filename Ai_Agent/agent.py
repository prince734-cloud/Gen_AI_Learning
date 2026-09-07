import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langsmith import Client # use Client to log the agent's execution and use existing prompts from the LangSmith platform
from langchain_mistralai import ChatMistralAI

load_dotenv()

search_tool = DuckDuckGoSearchRun()

api_key = os.getenv("MISTRAL_API_KEY")
if api_key:
    llm = ChatMistralAI(model="mistral-small-latest", api_key=api_key)

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt="You are a helpful travel assistant. Research travel destinations, attractions, and hotel options, then give clear recommendations.",
        debug=True,
    )

    result = agent.invoke({
        "messages": [
            HumanMessage(content="Plan a trip to Paris and find the best places to visit and hotels.")
        ]
    })
    print(result)
else:
    print("MISTRAL_API_KEY is not set. Add it to your .env file to enable the LLM call.")