# tyoe of messages in langchain
# 1 system message is instruction give to sytem like behave like a doctor
# 2 Human message
# 3 AI message

from langchain_core.messages import SystemMessage , HumanMessage,AIMessage
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

model=ChatMistralAI()
chat_history=[
    SystemMessage(content='You are helpful Ai assistant')
]

message=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='tell me about RAG')
]
result=model.invoke(message)
message.append(AIMessage(result.content))
print(message)