from pydantic import BaseModel,EmailStr,Field
from typing import Annotated,Optional

class Student(BaseModel):
    name:str='nithish'
    age:Optional[int]=None
    email:EmailStr
    cgpa:float=Field(gt=0,lt=10,default=5,description='cgpa shows the performance of the student' )

new_student={'age':23,'email':'abc@gmail.com','cgpa':8}
student=Student(**new_student)
print(student)

student_dict=dict(student)
print(student_dict['cgpa'])