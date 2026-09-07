from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI()

chat_history=[
  SystemMessage(content='You are helpful AI assistant')
]
while True:
  user_input= input('you: ')
  chat_history.append(HumanMessage(user_input))
  if user_input=="exit":
    break
  result = model.invoke(chat_history)
  chat_history.append(AIMessage(result.content))
  print('chatbot:', result.content)

print(chat_history)
# tyoe of messages in langchain
# 1 system message
# 2 Human message
# 3 AI message