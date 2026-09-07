from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

#text = "LangChain is used to build GenAI applications." # for one line text
document = ["LangChain is used to build GenAI applications.", 
            "LangChain is a framework for developing applications powered by language models.",
              "hello my name is Prince Yadav"] # for multiple lines text
#vector = embeddings.embed_query(text)
result= embeddings.embed_documents(document)
#print(len(vector))
#print(vector[:5])\
print(str(result))