from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import logging

# Initialize FastAPI app
app = FastAPI()

# Load the SentenceTransformer model once at startup
model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

# Define the request body structure
class TextRequest(BaseModel):
    input: str

@app.post("/embed")
async def embed(request: TextRequest):
    logging.info("FastAPI endpoint for text embedding has been called.")
    
    text = request.input

    if not text:
        raise HTTPException(status_code=400, detail="Error: 'input' field is empty.")

    try:
        # Generate embedding using the SentenceTransformer model
        embedding = model.encode(text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    # Convert the embedding (likely a numpy array) to a list for JSON serialization
    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding

    # Wrap the vector in the desired JSON format
    response_data = {
        "data": [
            {"embedding": embedding_list}
        ]
    }

    return response_data
