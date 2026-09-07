from langchain_text_splitters import RecursiveCharacterTextSplitter , Language

text="""
class Student:
    def __init__(self, name, age, branch):
        self.name = name
        self.age = age
        self.branch = branch

    # Method
    def display_info(self):
        print("Name:", self.name)
        print("Age:", self.age)
        print("Branch:", self.branch)

    # Another method
    def study(self):
        print(self.name, "is studying Python")


# Creating objects
student1 = Student("Prince", 21, "AIML")
student2 = Student("Rahul", 22, "CSE")

# Calling methods
student1.display_info()
student1.study()

print()

student2.display_info()
student2.study()
"""

splitter=RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0,
)

chunks=splitter.split_text(text)
print(len(chunks))
print(chunks[2])