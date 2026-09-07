from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # use to make dynamic prompts
from langchain_core.output_parsers import StrOutputParser
# this code is with using stroutputparser
# here we use chain
load_dotenv()


model=ChatMistralAI()
# 1st prompt detailed report
temp1=PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt summary
temp2=PromptTemplate(
    template='write a 5 line summary  on following ./n{text}',
    input_variables=['text']
)
parser=StrOutputParser()

chain=temp1|model|parser|temp2|model|parser

result=chain.invoke({'topic':'blackhole'})
print(result)