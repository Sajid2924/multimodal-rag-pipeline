# Multimodal RAG Pipeline

A Multi-Modal Retrieval-Augmented Generation (RAG) system for question answering over PDF documents.  
The pipeline supports text, tables, and image OCR and produces grounded answers with page-level citations.

This project was developed as part of a technical assessment to demonstrate an end-to-end RAG workflow.

---

## Features

- Multi-modal PDF ingestion
  - Text extraction
  - Table extraction
  - Image and chart OCR using Tesseract

- Chunking
  - Character-based chunking with overlap for text and images
  - Tables preserved as atomic chunks

- Retrieval
  - Dense retrieval using Sentence Transformers and FAISS
  - Sparse retrieval using BM25
  - Hybrid retrieval for improved recall

- Reranking
  - Cross-encoder reranking to improve relevance

- Grounded Question Answering
  - Answers generated strictly from retrieved context
  - Page-level citations

- Hallucination Prevention
  - Question answerability check
  - Post-generation answer verification

---

## Project Structure


├── ingest.py

├── chunk.py

├── embeddings.py

├── retrieve_generate.py

├── config.py

├── requirements.txt

├── data/

│      ├── chunks.json

│      ├── chunked.json

│      ├── faiss.index

│      └── metadata.json

└── README.md

---

## Setup

1. Clone the repository
 
   `git clone https://github.com/Sajid2924/multimodal-rag-pipeline.git`
 
   `cd multimodal-rag-pipeline`

2. Create a virtual environment (recommended)
   
   `python -m venv venv`
   
   `source venv/bin/activate`   # Linux / macOS
  
   `venv\Scripts\activate`   # Windows

3. Install dependencies

   `pip install -r requirements.txt`

4. Set environment variables

   Set the OpenAI API key:

   `export OPENAI_API_KEY="your_api_key"`        # Linux / macOS

   `setx OPENAI_API_KEY "your_api_key"`           # Windows

Ensure Tesseract OCR is installed and configured in config.py.

---

## Running the Pipeline

1. Ingest the PDF

   `python ingest.py`

2. Chunk the extracted content

   `python chunk.py`

3. Create embeddings and FAISS index

   `python embeddings.py`

4. Run the QA system

   `python retrieve_generate.py`

Ask questions interactively. Type `exit` to quit.

---

## Retrieval and QA Flow

- Hybrid retrieval combines dense semantic search and sparse keyword search.
- Retrieved chunks are reranked using a cross-encoder.
- Only the most relevant chunks are used to build the prompt.
- Answers are generated strictly from context and verified before being shown.

---

## Limitations

- Character-based chunking is used instead of token-based chunking.
- OCR extracts text but does not fully interpret visual elements in charts.
- Designed for clarity and correctness rather than large-scale performance.

---

## Future Improvements

- Token-based or semantic chunking
- Better table structure preservation
- Cross-modal image embeddings
- Advanced score fusion methods
