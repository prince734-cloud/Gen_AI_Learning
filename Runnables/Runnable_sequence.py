from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a Explanation about {joke}',
    input_variables=['joke']
)
parser=StrOutputParser()

model=ChatMistralAI()
chain=RunnableSequence(prompt|model|parser|prompt2|model|parser)
#chain=prompt|model|parser|prompt2|model|parser
result=chain.invoke({'topic':'Ant and Elephant'})
print(result)