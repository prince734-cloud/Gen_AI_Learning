from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)
documents = [
    """
    Virat Kohli is an Indian cricketer known for his aggressive batting style,
    consistency, and strong chasing ability. He is one of the leading run-scorers
    in international cricket and has represented India in Test, ODI, and T20 cricket.
    """,

    """
    Rohit Sharma is an Indian cricketer and an explosive opening batsman.
    He is famous for his ability to score big centuries and has recorded multiple
    double centuries in ODI cricket. Rohit has also served as India's captain.
    """,

    """
    Jasprit Bumrah is an Indian fast bowler known for his unusual bowling action,
    accurate yorkers, and ability to perform well under pressure. He is effective
    in both the powerplay and death overs and is considered one of India's leading bowlers.
    """,

    """
    MS Dhoni is a former Indian cricketer and one of the most successful captains
    in Indian cricket. He was known for his calm leadership, wicketkeeping skills,
    finishing ability, and powerful batting. He led India to major international trophies.
    """,

    """
    Ravindra Jadeja is an Indian all-rounder who contributes with batting, bowling,
    and fielding. He is a left-handed batsman and left-arm spin bowler. Jadeja is
    particularly known for his excellent fielding and ability to change matches.
    """,

    """
    Sachin Tendulkar is a legendary Indian batsman and one of the most respected
    cricketers in history. He holds numerous batting records and was the first player
    to score a double century in men's ODI cricket. He represented India for many years.
    """
]

query = "tell me about Ravindra jadeja ?"

documents_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

cosine_similarities = cosine_similarity([query_embedding], documents_embeddings)[0] # value passed in consine similarity should be in 2D array format
#print(list(enumerate(cosine_similarities)))
#print("Cosine Similarities:", cosine_similarities)
index, best_score = sorted(list(enumerate(cosine_similarities)), key=lambda x: x[1], reverse=True)[0] # sorting the cosine similarity values in descending order and getting the index of the highest value
print("Most similar document index:", index)
print("Best similarity score:", best_score)
print("Answer:", documents[index].strip())
print("similarity score of query with each document:", list(enumerate(cosine_similarities)))
