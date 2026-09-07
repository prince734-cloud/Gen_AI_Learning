from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
# here we use tiny lama locally in our system, you can use any other model from huggingface hub as well. and we have full control over the model parameters like temperature, max_new_tokens, etc. you can also use GPU if available in your system.
pipe = pipeline(
    task="text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto",      # Uses GPU if available, otherwise CPU
    torch_dtype="auto",
    max_new_tokens=30,
    temperature=0.7,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke("<|user|>\nExplain LangChain in simple words.\n<|assistant|>\n")

print(response)