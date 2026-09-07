from langchain_community.document_loaders import PyPDFLoader

loader=PyPDFLoader("C:/Users/adity/Downloads/Result_file.pdf")
docs=loader.load()
print(len(docs))
print(docs[0].metadata)
print(docs[1].page_content)
