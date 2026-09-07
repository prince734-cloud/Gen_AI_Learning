from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate   # use to make dynamic prompts
# this code is without using stroutputparser
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

prompt1=temp1.invoke({  
    'topic':'black hole'
})

result= model.invoke(prompt1)
  
prompt2=temp2.invoke({
    'text':result.content
})

fresult=model.invoke(prompt2)

print(fresult.content)
