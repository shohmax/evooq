import os
import re
from fastapi import status, HTTPException
from openai import OpenAI
import uuid
from openai import (
    NotFoundError,
    BadRequestError,
    AuthenticationError,
)
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.client.exceptions import (
    ServiceException,
    UnauthorizedException,
    PineconeApiKeyError,
    PineconeApiException,
)
from dotenv import load_dotenv

load_dotenv()

def init_services():
    """
    Initialize the OpenAI and Pinecone services.
    """
    try:
        openapi = OpenAI()
    except (AuthenticationError, NotFoundError) as e:
        raise Exception(f"The OpenAI API-key is not valid, {e}")

    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    except (UnauthorizedException, PineconeApiKeyError) as e:
        raise Exception(f"The Pinecone API-key is not valid, {e}")

    return openapi, pc


def create_index_pinecone(pc):
    """
    Create a Pinecone index if it does not exist.
    """
    pc_index = os.getenv("PINECONE_INDEX_NAME")
    if pc_index not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=pc_index,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        except (ServiceException, PineconeApiException, BadRequestError) as e:
            raise Exception(f"The creation of index {pc_index} failed in Pinecone, {e}")
    return pc.Index(pc_index)


openapi, pc = init_services()
PINECONE_INDEX = create_index_pinecone(pc)


async def get_embedding(text, model="text-embedding-3-small"):
    """
    Get embeddings for the given text using OpenAI's API.
    """
    try:
        embeddings = openapi.embeddings.create(
            input=[text], model=model).data[0].embedding
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return embeddings


def clean_text(text):
    """
    Clean the extracted text.
    """
    # Remove unwanted newlines within words
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Remove extra spaces, tabs, etc.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


async def split_text(text, chunk_size):
    """
    Split the text into chunks of the specified size.
    """
    try:
        text = clean_text(text)
        chunk_text = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to parse text {e}",
        )
    return chunk_text


async def upload_to_pinecone(embeddings_chunk, chunk, file_id):
    """
    Upload the embeddings to Pinecone with metadata.
    """
    try:
        chunk_id = str(uuid.uuid4())
        metadata = {"file_id": file_id, "chunk_id": chunk_id, "text": chunk}
        PINECONE_INDEX.upsert([(f"{file_id}_chunk_{chunk_id}", embeddings_chunk, metadata)])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def search(query, top_k):
    """
    Search for the text using embeddings in Pinecone by the query & top_k.
    """
    embeddings = await get_embedding(query)
    results = []
    try:
        pc_results = PINECONE_INDEX.query(
            vector=[embeddings],
            top_k=top_k,
            include_metadata=True,
        )
        for r in pc_results["matches"]:
            results.append({"score": r["score"], "text": r["metadata"]["text"]})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return results


async def chat_completions(search_results, query, model="gpt-3.5-turbo"):
    """
    Give answers based on the context.
    """
    try:
        context = " ".join(result["text"] for result in search_results)
        answers = openapi.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Keep answer in English language"},
                {"role": "user", "content": f"{query}"},
                {
                    "role": "assistant",
                    "content": f"follow only this context: {context}",
                },
            ],
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    return answers.choices[0].message.content
