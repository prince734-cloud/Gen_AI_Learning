import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
FREE_GROQ_MODEL = "openai/gpt-oss-20b"  # free/open-weight Groq model
model1=ChatMistralAI()

model2=ChatGroq(
    model=FREE_GROQ_MODEL,
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

prompt1=PromptTemplate(
    template='Generate a short and simple notes from following text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate five short question and answers from the following {text}',
    input_variables=['text']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz int a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)
parser= StrOutputParser()

parallel_chain=RunnableParallel({
    'notes':prompt1 | model1 |parser,
    'quiz': prompt2 | model2 |parser
}
)

merge_chain= prompt3 | model2 |parser

chain = parallel_chain|merge_chain
text="""Artificial Intelligence (AI)
1. **Artificial Intelligence (AI)** enables machines to perform tasks that normally require human intelligence.
2. AI systems can **learn from data, recognize patterns, and make decisions**.
3. **Machine Learning (ML)** allows computers to learn from data without being explicitly programmed.
4. **Deep Learning** uses neural networks to solve complex problems.
5. **NLP** helps computers understand and generate human language.
6. **Computer Vision** enables AI to understand images and videos.
7. **Generative AI** can create text, images, code, audio, and other content.
8. Popular AI models include **GPT, Gemini, Claude, Llama, Mistral, and DeepSeek**.
9. Lightweight models such as **Gemma and TinyLlama** can also be used for local AI applications.
10. AI is widely used in **chatbots, healthcare, education, recommendation systems, automation, and software development**.

"""
result=chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()