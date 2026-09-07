from langchain_community.retrievers import WikipediaRetriever

retriver=WikipediaRetriever(top_k_results=2,lang='en')


query="the geopolitical history of india and pakistan from the perspective of a chinese"

try:
    docs=retriver.invoke(query)
    print(f"Found {len(docs)} result(s).")
    for i, doc in enumerate(docs):
        print(f"\n--- Result{i+1}---")
        print(f"content:\n{doc.page_content}---")
except Exception as error:
    print(f"Wikipedia request failed: {error}")