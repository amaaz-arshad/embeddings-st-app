import fastapi
import logging
import requests
from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv() 

# Create FastAPI instance
app = fastapi.FastAPI()

# Hugging Face API token and model details
hf_token = os.getenv("HF_TOKEN")  # Replace with your token
model_id = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Replace with your model

# Define the Hugging Face API endpoint
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {'Authorization': f'Bearer {hf_token}'}

# Define a Pydantic model for the request body
class InputText(BaseModel):
    input: str

@app.post("/embed")
async def embed(request: InputText):
    logging.info("FastAPI endpoint for text embedding has been called.")

    # Extract the text from the request body
    text = request.input

    if not text:
        return fastapi.HTTPException(status_code=400, detail="Error: 'input' field missing or empty in JSON body.")

    # Prepare the payload for the Hugging Face API using the provided text
    payload = {"inputs": text}

    # Make the API request to Hugging Face
    try:
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for 4xx/5xx status codes
        result = response.json()  # Parse the response JSON
    except requests.exceptions.RequestException as e:
        return fastapi.HTTPException(status_code=500, detail=f"Error calling Hugging Face API: {str(e)}")

    # If the result is a list with one element, assume it is the vector array
    if isinstance(result, list) and len(result) == 1:
        vector = result[0]
    else:
        vector = result

    # Wrap the vector in the desired format
    response_data = {
        "data": [
            {"embedding": vector}
        ]
    }

    # Return the embedding result directly as a JSON array (vector array format)
    return response_data
