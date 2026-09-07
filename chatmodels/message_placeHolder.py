from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

# chat template
chat_temp=ChatPromptTemplate([
    ('system','You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'), # store the past history
    ('human','{query}')
])

# load chat history
chat_history=[]
with open(r'C:\Users\adity\OneDrive\Desktop\genai2\chatmodels\chat_history.txt') as f:
    chat_history.extend(f.readlines())

print(chat_history)

# create prompt
chat_temp.invoke({'chat_history':chat_history, 'query':'what is my refund status now'})
