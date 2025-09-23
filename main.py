from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import fitz
import base64
import io
import os
from PIL import Image
from openai import AzureOpenAI
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT", "gpt-4o")
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version="2024-08-01-preview",
    azure_endpoint=AZURE_ENDPOINT
)

app = FastAPI()
templates = Jinja2Templates(directory="templates")
def input_pdf_setup(uploaded_file: bytes):
    """Convert first page of PDF into base64 encoded image (JPEG)."""
    pdf_document = fitz.open(stream=uploaded_file, filetype="pdf")
    first_page = pdf_document[0]
    pix = first_page.get_pixmap()

    img_byte_arr = io.BytesIO()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    pdf_parts = [
        {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(img_byte_arr).decode()
        }
    ]
    return pdf_parts
def extract_pdf_text(uploaded_file: bytes) -> str:
    """Extract text from all pages of PDF."""
    pdf_document = fitz.open(stream=uploaded_file, filetype="pdf")
    text = ""
    for page in pdf_document:
        text += page.get_text("text")
    return text

def get_gpt_response(messages: list) -> str:
    """Send messages to Azure OpenAI GPT and return response text."""
    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": None,
            "job_desc": "",
            "resume_name": ""
        }
    )
@app.post("/process", response_class=HTMLResponse)
async def process(
    request: Request,
    job_desc: str = Form(...),
    action: str = Form(...),
    resume: UploadFile = None
):
    if not resume:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "result": "Please upload a resume.",
                "job_desc": job_desc,
                "resume_name": ""
            }
        )
    file_bytes = await resume.read()
    _ = input_pdf_setup(file_bytes) 
    pdf_text = extract_pdf_text(file_bytes)
    prompts = {
        "about": "Review the resume vs job description. Return analysis in structured plain text with sections: SUMMARY, STRENGTHS, WEAKNESSES.",
        "percentage": "Give ATS match percentage, missing keywords, and comments. Output must be structured plain text with: SCORE, MISSING KEYWORDS, COMMENTS.",
        "suggestion": "Suggest improvements. Output as structured text with: SUGGESTIONS (numbered list).",
        "skills": "List required and missing job skills in structured text: REQUIRED SKILLS, MISSING SKILLS.",
        "format": "Evaluate formatting and readability. Return structured plain text: FORMATTING ISSUES, RECOMMENDATIONS.",
        "breakdown": "Provide section-wise analysis (Experience, Education, Skills, Certifications). Output structured text with clear headings.",
        "density": "Check keyword frequency. Output in structured text: FREQUENT KEYWORDS, LOW-FREQUENCY KEYWORDS, OPTIMIZATION SUGGESTIONS.",
        "experience": "Compare required vs candidate experience. Output structured plain text: REQUIRED EXPERIENCE, CANDIDATE EXPERIENCE, GAP.",
        "grammar": "Check grammar, spelling, stylistic errors. Return structured plain text: ERRORS, CORRECTIONS, COMMENTS."
    }
    if action == "grammar":
        messages = [
            {"role": "system", "content": prompts[action]},
            {"role": "user", "content": pdf_text}
        ]
    else:
        messages = [
            {"role": "system", "content": prompts[action]},
            {"role": "user", "content": f"Job Description:\n{job_desc}"},
            {"role": "user", "content": f"Resume text:\n{pdf_text}"}
        ]
    result = get_gpt_response(messages)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "job_desc": job_desc,
            "resume_name": resume.filename
        }
    )
