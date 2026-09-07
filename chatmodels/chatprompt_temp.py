from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
# use for multiple message
load_dotenv()

chat_temp=ChatPromptTemplate([
    ('system','You are a helpful {domain} expert'),
    ('human','Explain in simple terms, what is {topic}')
])

prompt=chat_temp.invoke({'domain':'cricket','topic':'dusra'})
print(prompt)