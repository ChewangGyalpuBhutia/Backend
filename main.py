from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import fitz  # PyMuPDF
from models import PDFDocument, Question, SessionLocal
from operations import Operations

app = FastAPI()

# Set up CORS
origins = [
    "http://localhost:3000",  # React frontend
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type")

    content = ""
    with fitz.open(stream=file.file.read(), filetype="pdf") as doc:
        for page in doc:
            content += page.get_text()

    pdf_document = PDFDocument(filename=file.filename, content=content)
    db.add(pdf_document)
    db.commit()
    db.refresh(pdf_document)

    return {"filename": file.filename, "id": pdf_document.id}

@app.post("/ask_question/")
async def ask_question(
    pdf_id: int = Form(...), 
    question: str = Form(...), 
    db: Session = Depends(get_db)
):
    pdf_document = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
    if not pdf_document:
        raise HTTPException(status_code=404, detail="PDF not found")

    try:
        operations = Operations()
        answer = operations.ask_question(pdf_document.content, question)

        question_record = Question(pdf_id=pdf_id, question=question, answer=answer)
        db.add(question_record)
        db.commit()
        db.refresh(question_record)

        return {"question": question, "answer": answer}
        
    except Exception as e:
        print(f"Error in ask_question endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))