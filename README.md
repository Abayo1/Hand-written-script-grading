# Handwritten Script Grading

This project is a FastAPI application that uses the TrOCR model for Optical Character Recognition (OCR) of handwritten student answers and utilizes GPT-4 for evaluating and grading those answers based on provided teacher answers. 

## Features

- Upload handwritten images for processing.
- Extract text from images using the TrOCR model.
- Compare extracted text with teacher answers for similarity.
- Generate feedback and a grade based on the comparison using GPT-4.

## Technologies Used

- FastAPI: A modern web framework for building APIs with Python.
- TrOCR: A transformer model for text recognition from images.
- OpenAI GPT-4: For evaluating student responses.
- scikit-learn: For computing similarity between the teacher's and student's answers.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Abayo1/Hand-written-script-grading.git
   cd Hand-written-script-grading

2. Create a Virtual Environment (optional): It’s a good practice to isolate your project dependencies.
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   
3. Install Dependencies: Use the provided `requirements.txt` to install all necessary packages.
```bash
pip install -r requirements.txt

4. Set OpenAI API Key: Ensure that your OpenAI API key is set correctly in the `main.py` file.
5. Run the Application: Use `uvicorn` to start your FastAPI application, and access it in your web browser.
```bash
uvicorn app:app --reload
You can access the application at http://127.0.0.1:8000.


