import os
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)

# Output structure
schemas = [
    ResponseSchema(name="fact_1", description="First fact"),
    ResponseSchema(name="fact_2", description="Second fact"),
    ResponseSchema(name="fact_3", description="Third fact")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

# Prompt
prompt = PromptTemplate(
    template="Give 3 facts about {topic}.\n{format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Chain
chain = prompt | model | parser

# Run
result = chain.invoke({"topic": "Black Hole"})

print(result)