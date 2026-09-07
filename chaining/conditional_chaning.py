import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
FREE_GROQ_MODEL = "openai/gpt-oss-20b"  # free/open-weight Groq model

model2=ChatGroq(
    model=FREE_GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)
parser=StrOutputParser()


class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='Give the sentiment of the feedback')
parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt=PromptTemplate(
    template='classify the sentiment of the following feedback text into positive or negative , only give output in one word either negative or positive \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)
classifier_chain= prompt|model2|parser2
prompt2=PromptTemplate(
    template='Write an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)
prompt3=PromptTemplate(
    template='Write an appropriate response to this Negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain= RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model2|parser),
    (lambda x:x.sentiment=='negative',prompt3|model2|parser),
    RunnableLambda(lambda x:"could not find sentiment")
)

chain=classifier_chain |branch_chain

result=chain.invoke({'feedback':'This is a terrible phone'})
print(result)