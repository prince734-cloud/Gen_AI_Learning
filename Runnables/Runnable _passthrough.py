# print same input as output =passthrough
import os
from pathlib import Path

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

FREE_GROQ_MODEL = "openai/gpt-oss-20b"  # free/open-weight Groq model

model=ChatMistralAI()
model2=ChatGroq(
    model=FREE_GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)
prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a Explanation about {joke}',
    input_variables=['joke']
)
parser=StrOutputParser()

joke_gen_chain=RunnableSequence(prompt|model|parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explanation':RunnableSequence(prompt2|model2|parser)
})

final_chain=RunnableSequence(joke_gen_chain,parallel_chain)
res=final_chain.invoke({'topic':'cricket'})
print(res)