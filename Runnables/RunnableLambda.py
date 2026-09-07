# runnablelembda is used to convert python function into runnables
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda 
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

prompt=PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser=StrOutputParser()

model=ChatMistralAI()

def word_count(text):
    return len(text.split())


joke_gen_chain=RunnableSequence(prompt|model|parser)

paralle_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':RunnableLambda(word_count)
    #'word_count':RunnableLambda(lambda x:len(x.split()))
})

final_chain=RunnableSequence(joke_gen_chain,paralle_chain)

result=final_chain.invoke({'topic':'AI'})
print(result)