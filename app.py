from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import requests

app = FastAPI()

class Review(BaseModel):
    id: int
    title: str
    overview: str

@app.get('/')
def Index():
    return 'server is running'


@app.post('/predict')
def LoadAndPredict(review: Review):
    model = tf.saved_model.load('./my_model')
    
    prediction = model([review.overview])
    data = prediction.numpy()[0][0]
    
    evaluation = ''
    if data > 0.5:
        evaluation = 'positive'
    else:
        evaluation = 'negative'

    response = requests.post('http://localhost:5000/movies/revies{review.id}', {
      "id": review.id,
      "prediction": evaluation
    })

    return {
        "id": review.id,
        "prediction": evaluation
    }

