from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda ,RunnableBranch
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
load_dotenv()

prompt=PromptTemplate(
    template='Answer the following question \n {question} from the following text -\n {text} ',
    input_variables=['question','text']
)
parser=StrOutputParser()

model=ChatMistralAI()

url='https://developer.mozilla.org/en-US/docs/Web/JavaScript'
loader=WebBaseLoader(url)
docs=loader.load()
#print(docs[0].page_content)
#print(len(docs))

chain=prompt|model|parser
result=chain.invoke({'question':'what is javascript','text':docs[0].page_content})
print(result)