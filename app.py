from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from typing import List
import PyPDF2
import io

from process import split_text, get_embedding, upload_to_pinecone, search, chat_completions

app = FastAPI()

@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum of 100 PDF files can be uploaded.")
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF.")
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(await file.read()))

        for page in pdf_reader.pages:
            for chunk in await split_text(page.extract_text(), chunk_size=3000):
                embeddings_chunks = await get_embedding(chunk)
                await upload_to_pinecone(embeddings_chunks, chunk, file.filename)

        
    return {
            "message": "PDFs uploaded, text extracted, and saved to the DB successfully.",
            "file_count": len(files),
            "files": files
        }

@app.post("/query")
async def query(
    query: str = Form(...)
):
    """
    Answer the query by the search results context
    """
    search_results = await search(query, top_k=5)
    answers = await chat_completions(search_results, query)

    return {"reply": answers}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)