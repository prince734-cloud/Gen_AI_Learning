import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    raise RuntimeError(
        "Missing HUGGINGFACEHUB_API_TOKEN. Add it to .env or export it in your terminal."
    )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


parser=JsonOutputParser()
# 2nd prompt summary
template=PromptTemplate(
    template='give me the name, age and address of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)
# prompt=template.format()

#result=model.invoke(prompt)

#final_result=parser.parse(result.content)
#print(final_result)


 # with chain
chain=template |model|parser

result=chain.invoke({})
print(result)



# through json we cannot decide out schema ,schema is decided by model itself



