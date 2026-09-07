from langchain_community.document_loaders import CSVLoader


loader=CSVLoader(file_path='Document_loader\Books.csv')
data=loader.load()

print(data[0])  #  len= no. of rows 