# AiPlanet Backend

This backend application allows users to upload PDF documents and ask questions regarding the content of these documents. The backend processes these documents and utilizes natural language processing to provide answers to the questions posed by the users.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [API Endpoints](#api-endpoints)
- [Code Overview](#code-overview)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **PDF Upload**: Users can upload PDF documents to the application.
- **Question Answering**: Users can ask questions related to the content of an uploaded PDF, and the system will provide answers based on the document content.
- **Context Retrieval**: The system retrieves relevant context from the document to generate accurate answers.

## Technologies Used

- **Backend Framework**: FastAPI
- **NLP Processing**: Hugging Face Transformers
- **Database**: PostgreSQL
- **PDF Processing**: PyMuPDF
- **Vector Store**: FAISS
- **Embeddings**: SentenceTransformers

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/ChewangGyalpuBhutia/Backend.git
    cd aiplanet-backend
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

4. **Set up the database**:
    Update the `DATABASE_URL` in `models.py` with your PostgreSQL database credentials.

5. **Run the database migrations**:
    ```sh
    python -c 'from models import Base, engine; Base.metadata.create_all(bind=engine)'
    ```

## Running the Application

1. **Start the FastAPI server**:
    ```sh
    uvicorn main:app --reload
    ```

2. **Access the API documentation**:
    Open your browser and navigate to `http://127.0.0.1:8000/docs` to view the interactive API documentation.

## API Endpoints

### Upload PDF

- **Endpoint**: `/upload_pdf/`
- **Method**: `POST`
- **Description**: Upload a PDF document.
- **Parameters**:
  - `file`: The PDF file to upload.
- **Response**:
  - `filename`: The name of the uploaded file.
  - `id`: The ID of the uploaded document.

### Ask Question

- **Endpoint**: `/ask_question/`
- **Method**: `POST`
- **Description**: Ask a question related to the content of an uploaded PDF.
- **Parameters**:
  - `pdf_id`: The ID of the uploaded PDF document.
  - `question`: The question to ask.
- **Response**:
  - `question`: The question asked.
  - `answer`: The answer generated based on the document content.

## Code Overview

### `main.py`

- Sets up the FastAPI application.
- Configures CORS to allow requests from the frontend.
- Defines the endpoints for uploading PDFs and asking questions.

### `models.py`

- Defines the SQLAlchemy models for `PDFDocument` and `Question`.
- Sets up the PostgreSQL database connection.

### `operations.py`

- Contains the `Operations` class which handles document processing and question answering.
- Uses Hugging Face Transformers for NLP tasks.
- Uses FAISS for vector storage and similarity search.

### `requirements.txt`

- Lists all the dependencies required to run the application.

## Troubleshooting

- **CORS Issues**: Ensure that the frontend URL is included in the `origins` list in `main.py`.
- **Database Connection**: Verify that the `DATABASE_URL` in `models.py` is correct and that the PostgreSQL server is running.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
