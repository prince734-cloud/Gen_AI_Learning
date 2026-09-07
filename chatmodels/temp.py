from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI()

prompt=PromptTemplate(
    input_variables=['topic'],
    template='Suggest a catchy blog title about {topic}'
)

topic=input('enter your topic for blog:')

formatted_prompt=prompt.format(topic=topic)

blog_title=model.invoke(formatted_prompt)

print('generated blog title:',blog_title.content)