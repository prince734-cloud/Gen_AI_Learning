
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda ,RunnableBranch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
load_dotenv()

prompt=PromptTemplate(
    template='Write a summary for the following poem -\n {poem} ',
    input_variables=['poem']
)
parser=StrOutputParser()

model=ChatMistralAI()

loader = TextLoader('Document_loader/text.txt')
docs = loader.load()
#print(docs) # type of docs is list  
#print(len(docs)) # 1
#print(docs[0]) # is used to extract document in format
#print(type(docs[0]))
#print(docs[0].metadata)
#print(docs[0].page_content)

chain= prompt |model|parser

result=chain.invoke({'poem':docs[0].page_content})
print(result) 