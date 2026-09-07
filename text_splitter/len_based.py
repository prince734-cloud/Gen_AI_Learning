from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('c:/Users/adity/AppData/Local/Packages/5319275A.WhatsAppDesktop_cv1g1gvanyjgm/LocalState/sessions/AE5B9BCD926BE091C7D795C12036051804F758B0/transfers/2026-34/Result_file.pdf')
docs=loader.load()
text="""
On a sunny field where dreams take flight,
The bat meets ball with all its might.
The crowd roars loud, the players run,
Cricket brings joy to everyone.

A bowler charges, fierce and fast,
A batsman hopes his shot will last.
A soaring six goes through the sky,
While fielders leap and balls fly high.

Through every win and every defeat,
The game teaches us to never retreat.
With courage, teamwork, and a fighting heart,
Cricket reminds us—believe from the start.

From dusty grounds to stadium light,
Cricket unites us day and night.
More than a game, it’s passion and pride,
A beautiful journey played side by side.
"""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)
result=splitter.split_documents(docs)
print(result[0].page_content)
