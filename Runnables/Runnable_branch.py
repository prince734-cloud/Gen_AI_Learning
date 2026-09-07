# it is like conditional chin where we want to genrate output based on conditions
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda ,RunnableBranch
from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(
    template='Write a report about {topic} ',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='Summarize the following text \n {text} ',
    input_variables=['text']
)
parser=StrOutputParser()

model=ChatMistralAI()

report_gen_chain=RunnableSequence(prompt|model|parser)
branch_chain=RunnableBranch(
    (lambda x:len(x.split())>120,RunnableSequence(prompt2|model|parser)),
    #default
    RunnablePassthrough()

)
final_chain=RunnableSequence(report_gen_chain|branch_chain)

result=final_chain.invoke({'topic':'Machine learning'})
print(result)