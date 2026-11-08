# a decorator function in python is a functiopn that takes as input another function and extends the behavior of the latter function without explicitly modifying it. In this example, we will create a decorator function that logs the execution time of a function.

# def my_func():
#     pass

# my_func=decorator(my_func)


# from curses import wrapper


# def create_decorator(func):
#     def wrapper():
#         result=func()
#         return result.upper() + "!!!"
#     return wrapper


# @create_decorator
# def say_hello():
#     return "hello"

# print(say_hello())


from fastapi import FastAPI, Body, File, Form, HTTPException, UploadFile




app=FastAPI()
# @app.get("/hello")
# async def hello():
#     return "Hello Deepseed World"






@app.get("/hello")
def index():
    return {"data":{"name":"Gita", "email":"fon@gmail.com", "favMeal":"fufu"}}



# how this can be used in AI apps
# agents, llms, prompt templates , chains and much more 
@app.get("/get-response")
def get_airesponse():
    # use an AI model to generate a response based on the input provided
    return {"data":"This is a sample response from the AI model.", "final_response":"Agent response"}

@app.get("/sentiment/{text}")
def analyze_sentiment(text):
    if text=="":
        return {"data":"No text provided"}
    else:
        # perform sentiment analysis on the text and return the result
        # use one of the hugginface models to do sentiment analysis
        # USE REAL HUGGING PHASE SENTIMENT ANALYSIS MODEL HERE
        return {"text":text, "sentiment":"Positive"}


@app.get("/model-response/{model_id}")
def model_response(model_id:int):
    models={1:"gpt2", 2:"gpt3", 3:"gpt-5", 4:"gpt-6"}
    get_model=models.get(model_id, "not found")
    return {
        "id":model_id,
        "model_name":get_model
    }





# @app.get("/about-seed")
# def about_seed():
#     return {'data':"about page"}
""

# # path bparameter is  a variabke inside an endpoint

# # it lets client sensd infor directly cvia  a  url: eg prompt:

# # "/predict/{text}"

# # @app.get("/predict/{text}")
# # def sanswer_question(text):
# #     if text=="":
# #         print("No text provided")
# #         return {"data":"No text provided"}
# #     else:
# #         return {"text":text, "answer":"This is a sample answer for the text provided."} 




# @app.get("/model/{model_id}")
# def get_model_info(model_id:int):
#     model={1:"text", 2:"image", 3:"audio"}
#     model_name=model.get(model_id, "Unknown Model")
#     return {"model_id":model_id, "model_name":model_name}


# @app.get("/hello")
# def say_hello():
#     return {"message":"Hello Deepseed World"}

# @app.get("/blog/{id}")
# def show(id:int):
#     return {"data":id}

# @app.get("/blog/{unpublish}")
# def unpublish():
#     return {'data':'unpublish all blogs'}




# # Query parametrs

@app.get("/blog")

def index(limit:int, published:str):
    if published:
        return {"data":f"{limit} published blogs FROM DB"}
    else:
        return {"data":f"{limit} blogs FROM DB"}


# @app.post("/raw-body")
# def raw_body(data:dict=Body(...)):
#     return {"data":data}
    


# SOLUTION,
MODELS=["gpt", "gemini", "claude"]

@app.get("/response")
def generate_response(prompt:str, model:str):
    if model in MODELS:
        return {
            "data":"This is your response",
            "model":model,
            "prompt":prompt
        }
    else:
        return {
            "data":"no model found"
        }



# import base64t
from typing import Optional

# # REQUEST BODY
from pydantic import BaseModel

class HelloRequest(BaseModel):
    name:str
    email:str
    age:int


class StudentRequest(BaseModel):
    name:str
    email:str
    age:int
    course:Optional[str]="Ai fundamentals"


# class StudentPatch(BaseModel):
#     # Optional fields let PATCH callers send only the values they want to change.
#     name:Optional[str]=None
#     email:Optional[str]=None
#     age:Optional[int]=None
#     course:Optional[str]=None

@app.post("/hello")
def hello(request:HelloRequest):
    # do alot here
    return {"data":f"Hello {request.name}, your email is {request.email} and you are {request.age} years old."}




@app.post("/student")
def student(request:StudentRequest):
    return {"data":f"Hello {request.name}, your email is {request.email}, you are {request.age} years old and you are enrolled in the {request.course} course."}


# # This in-memory store is for learning only: its data resets whenever the app restarts.
students:dict[int, StudentRequest]={}
next_student_id=1


@app.post("/students", status_code=201)
def create_student(request:StudentRequest):
    """Create a student; POST sends a new record to the collection."""
    global next_student_id
    student_id=next_student_id
    next_student_id+=1
    students[student_id]=request
    return {"id":student_id, **request.model_dump()}


@app.get("/students/{student_id}")
def get_student(student_id:int):
    """Read one student by ID; the ID is part of the URL path."""
    student_record=students.get(student_id)
    if student_record is None:
        raise HTTPException(status_code=404, detail="Student not found")
    return {"id":student_id, **student_record.model_dump()}


@app.put("/students/{student_id}")
def replace_student(student_id:int, request:StudentRequest):
    """Replace every student field; PUT expects the complete record."""
    if student_id not in students:
        raise HTTPException(status_code=404, detail="Student not found")
    students[student_id]=request
    return {"id":student_id, **request.model_dump()}


# @app.patch("/students/{student_id}")
# def update_student(student_id:int, request:StudentPatch):
#     """Change only fields sent by the caller; PATCH keeps all other values."""
#     student_record=students.get(student_id)
#     if student_record is None:
#         raise HTTPException(status_code=404, detail="Student not found")
#     updated_fields=request.model_dump(exclude_unset=True)
#     updated_record=student_record.model_copy(update=updated_fields)
#     students[student_id]=updated_record
#     return {"id":student_id, **updated_record.model_dump()}


# @app.delete("/students/{student_id}", status_code=204)
# def delete_student(student_id:int):
#     """Delete one student; DELETE succeeds without returning a response body."""
#     if student_id not in students:
#         raise HTTPException(status_code=404, detail="Student not found")
#     del students[student_id]


# # PROJECT HERE:

# @app.post("/generate-story")
# async def  generate_story(file:UploadFile=File(...), prompt:str=Form(...)):
#     try:
#         image_bytes=await file.read()
#         image_base64=base64.b64encode(image_bytes).decode("utf-8")
#         image_url=f"data:image/jpeg;base64,{image_base64}\nDescribe this image"

#         description=llm.invoke(image_url.to_text())  # convert the image to text using the llm model

#         story_prompt=f"""create a  beatiful story based on the following description: {description} and the prompt: {prompt}"""

#         story=llm2.invoke(story_prompt)  # generate a story based on the description and prompt

#         return {
#             "success":True,
#             "description":description,
#             "story":story,
#             "filename":file.filename,
#         }


#     except Exception as e:
#         pass
