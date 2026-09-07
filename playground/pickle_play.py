import pickle
from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int


student_1 = Student(name="arlo", age=47)

# write bynary file
with open("playground/student", mode='wb') as fd:
        pickle.dump(student_1, fd)

# read from bynary file
with open("playground/student", mode='rb') as fd:
        student_2 = pickle.load(fd, encoding='utf8')

print(student_2.name)
print(student_2.age)
