from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

text_splitter=SemanticChunker(
    MistralAIEmbeddings(),breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)
sample="""
Technology has changed the way people learn and communicate,
 making information available within seconds through smartphones and the internet. 
 At the same time, sports such as cricket teach important values like teamwork,
discipline, patience, and leadership, which are useful in both personal and professional life.

Environmental protection is another important topic in today's world.
Increasing pollution, deforestation, and climate change are affecting nature and human health,
so people should reduce waste, save water, plant trees, and use renewable sources of energy whenever possible.
"""
docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)