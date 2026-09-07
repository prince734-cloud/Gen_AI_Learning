from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field

load_dotenv()

# Model

model=ChatMistralAI()


class Person(BaseModel):
    name:str=Field(description='Name of the person'),
    age:int=Field(gt=18,description='age of the person'),
    city:str=Field(description='Name of city the person belong to')

parser=PydanticOutputParser(pydantic_object=Person)

template=PromptTemplate(
    template='Generate the name ,age and city of a fictional {place} person \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions}

    
)
chain = template | model |parser
res = chain.invoke({'place':'German'})
print(res)

#prompt=template.invoke({'place':'indian'})

#result=model.invoke(prompt)

#final_result=parser.parse(result.content)
#print(final_result)