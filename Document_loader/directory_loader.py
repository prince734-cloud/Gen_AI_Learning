from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader

loader=DirectoryLoader(
    path='',
    glob='*.pdf',# extract all .pdf file
    loader_cls=PyPDFLoader
)
docs=loader.load()   # when we have large amout of data (pdf) then we use lazy_load() 
print(docs) 