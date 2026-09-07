import os
from pathlib import Path

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FREE_GROQ_MODEL = "openai/gpt-oss-20b"  # free/open-weight Groq model

prompt=PromptTemplate(
    template='Generate a tweet about  {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Generate a linkedin post about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()

model=ChatMistralAI()
model2=ChatGroq(
    model=FREE_GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

parallel_chain=RunnableParallel({
    'tweet':RunnableSequence(prompt|model|parser),
    'post':RunnableSequence(prompt2|model|parser)
})

result=parallel_chain.invoke({
    'topic':'AI'
})

print(result)
#print(result['tweet'])
