# do chucking based on structure like paragraph,line ,space
#\n\n=paragraph,\n=line,'_'=space
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

splitter=RecursiveCharacterTextSplitter(
      chunk_size=20,
      chunk_overlap=0
)
chunks=splitter.split_text(text)

print(len(chunks))
print(chunks)