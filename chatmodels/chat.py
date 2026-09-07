from dotenv import load_dotenv # Load environment variables from .env file
load_dotenv()

#from langchain.chat_models import init_chat_model

#model = init_chat_model("google_genai:gemini-flash-lite-latest")
#response = model.invoke("what is llm.")
#print(response.content)
from langchain_mistralai import ChatMistralAI

model = ChatMistralAI(
    model="mistral-small-2603",temperature=0.7  # temperature and max_output_tokens are optional parameters that helps to do creative work and limit the output length of the model respectively.
    # if we put value of temperature 0 it will always give the same output for the same input prompt, if we put value of temperature more than 0 it will give different output for the same input prompt and differnces increaces as we increase the value.
)
response = model.invoke("write a poem.")
print(response.content)