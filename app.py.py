#!/usr/bin/env python
# coding: utf-8

# In[3]:


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from gensim.models import Word2Vec
import threading
import uvicorn

# Load the trained Logistic Regression model
model = joblib.load("logistic_regression_model.pkl")

# Load the trained Word2Vec model
w2vec_vect = Word2Vec.load("word2vec_model.bin")

# Initialize FastAPI app
app = FastAPI()

# Define request model
class TextRequest(BaseModel):
    text: list

# Function to get document vector
def get_document_vector(doc, model):
    tokens = [word for word in doc if word in model.wv.key_to_index]
    if tokens:
        doc_embedding = np.mean(model.wv[tokens], axis=0)
    else:
        doc_embedding = np.zeros(model.vector_size)
    return doc_embedding

# Define API endpoint
@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    if not isinstance(text, list) or not all(isinstance(word, str) for word in text):
        raise HTTPException(status_code=400, detail="Invalid input format. Expected a list of strings.")
    
    # Tokenize and create document vector
    doc_embedding = get_document_vector(text, w2vec_vect)
    doc_embedding = np.array(doc_embedding).reshape(1, -1)
    
    # Predict
    prediction = model.predict(doc_embedding)
    
    # Return prediction
    return {"prediction": int(prediction[0])}

# Function to start FastAPI server
def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the server in a separate thread if this is the main module
if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    server_thread.start()


# In[ ]:




