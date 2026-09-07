from langchain_mistralai import ChatMistralAI,MistralAIEmbeddings
from dotenv import load_dotenv # Load environment variables from .env file
load_dotenv()
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable, IpBlocked, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


# Step 1a - Indexing (Document Ingestion)

video_id = "7bEStwqNPqU"  # only the ID, not full URL
transcript = None

try:
    # If you don't care which language, this returns the "best" one
    transcript_list = YouTubeTranscriptApi().fetch(
        video_id,
        languages=['en']
    )

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)

    #print(transcript)

except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")
except VideoUnavailable:
    print(f"Video {video_id} is unavailable, so its transcript cannot be retrieved.")
except IpBlocked:
    print(f"YouTube blocked access for video {video_id}. Try a different video or use a network that can access YouTube.")
except NoTranscriptFound:
    print(f"No transcript found for video {video_id} in English or Hindi.")

if transcript is None:
    raise SystemExit(1)

# step2 Splitting
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
chunks=splitter.create_documents([transcript])
print(len(chunks))

# Step3 Embeddings Vectors

embeddings=MistralAIEmbeddings()
vector_store=FAISS.from_documents(chunks,embeddings)

# Step4 Retriever

retriever=vector_store.as_retriever(search_type='similarity',search_kwargs={"k":3})
#print(retriever.invoke('what is langraph'))

# llm
llm= ChatMistralAI()

prompt=PromptTemplate(
    template="""
     you are a helpful assistant.
     Answer only from the provided transcript context.
     If the context is insufficient , just say i don't know the answer
       {context} 
       Question:{question}""",
       input_variables=['context','question']

)
question='is the topic of english practice is discussed in this vedio? If yes then what was discussed'
retriever_docs=retriever.invoke(question)
#print(retriever_docs)
#context_text="\n\n".join(doc.page_content for doc in retriever_docs)
#final_prompt=prompt.invoke({'context':context_text,'question':question})
#print(final_prompt)

# Step5 Generation

#answer = llm.invoke(final_prompt)
#print(answer.content)

parser=StrOutputParser()
def format_docs(retriever_docs):
    context_text="\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text

parallel_chain=RunnableParallel(
    {
        'context':retriever|RunnableLambda(format_docs),  # here we pass answer of question and context
        'question':RunnablePassthrough()
    }
)
main_chain=parallel_chain|prompt|llm|parser
result=main_chain.invoke('how to learn english fast')
print(result)