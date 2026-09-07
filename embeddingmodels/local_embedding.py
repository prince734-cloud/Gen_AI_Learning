from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)
text="LangChain is used to build GenAI applications." # for one line text
document = ["LangChain is used to build GenAI applications.",
            "LangChain is a framework for developing applications powered by language models.",
              "hello my name is Prince Yadav"] # for multiple lines text

embeddings_result = embeddings.embed_query(text)
print(str(embeddings_result))