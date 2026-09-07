import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

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
response = model.invoke("who are you")
print(response.content)
