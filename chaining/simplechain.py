from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt=PromptTemplate(
    template='Generte five interesting facts about {topic}',
    input_variables=['topic']
)
model=ChatMistralAI()
parser=StrOutputParser()

chain=prompt|model|parser
result=chain.invoke({'topic':'Tiger'})
print(result)
chain.get_graph().print_ascii() # show the flow of chain