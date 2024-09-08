# EVOOQ: RAG-Doc App

A simple RAG-Doc App for EVOOQ


## Prerequisites
- Python 3.10~3.11 
- PyPDF2 (Extracts text data from PDFs)
- OpenAI API-Key (as a LLM and Embedding Model)
- Pinecone API-Key (as a vector DB)
- FastAPI (runs the services)
- Prepare your .env file, learn in [Configuration](#configuration)


## Configuration
1. Fill a `.env` file in the root directory and add the following environment variables:
    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    PINECONE_API_KEY=your_pinecone_api_key_here
    ```


## Run

Run the sytem, it is based on FastAPI. 

 ```bash
 git clone https://github.com/shohmax/evooq.git
 cd evooq/

 uvicorn app:app --port 8000
 ```


## Usage

There are **two ways** to test the system:
1. Using `cli.py`
2. Using The Swagger Tool

### 1. Test with a simple `cli.py`! 

The **Upload** command finds all `.pdf` documents in the folder, sends them to the server, where the server extracts the text data, generates text embeddings using OpenAI, and uploads them to Pinecone DB.

```bash

python3 cli.py upload /path/folder

 ```

The **Query** command answers questions based on the context of the uploaded documents.

```bash

python3 cli.py query "What is Consult+ prediction for the price of Tesla stock?"

 ```


### 2. Test The Swagger Tool

Test at `http://localhost:8000/docs`.


