from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import openai
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

openai.api_key = " "

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

model.eval()

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_file(file: UploadFile = File(...), teacher_answer: str = Form(...), request: Request = None):
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    pixel_values = processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)

    student_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    '''Calculating similarity'''
    vectorizer = TfidfVectorizer().fit_transform([teacher_answer, student_answer])
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    similarity_score = similarity_matrix[0][0] * 100  # Convert to percentage for readability

    prompt = (
        "The following is a student's answer extracted from an image:\n"
        f"Student's answer: {student_answer}\n"
        f"Teacher's answer: {teacher_answer}\n"
        "Please evaluate the student's answer and provide feedback on its accuracy, "
        "completeness, and relevance to the teacher's answer. Additionally, provide "
        "a quantitative grade out of 100."
    )

    gpt_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert teacher evaluating student answers."},
            {"role": "user", "content": prompt}
        ]
    )

    grading_feedback = gpt_response['choices'][0]['message']['content']

    '''similarity score'''
    print("OCR Result (Student Answer):", student_answer)
    print("Similarity Score:", similarity_score)
    print("Grading Feedback:", grading_feedback)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "student_answer": student_answer,
        "teacher_answer": teacher_answer,
        "similarity_score": f"{similarity_score:.2f}%",
        "grading_feedback": grading_feedback,
    })

