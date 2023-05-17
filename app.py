from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import tensorflow_text
import requests

app = FastAPI()


class Review(BaseModel):
    id: int
    movie_id: int
    content: str


@app.get('/')
def Index():
    return 'server is running'


@app.post('/predict')
def LoadAndPredict(review: Review):
    BERT = '../bert_model'
    model = tf.saved_model.load(BERT)

    prediction = model([review.content])
    data = prediction.numpy()[0][0]

    evaluation = True
    if data >= 0:
        evaluation = True
    else:
        evaluation = False

    response = requests.post('http://localhost:8080/api/ai', {
        "id": review.id,
        "movie_id": review.movie_id,
        "result": evaluation
    })

    return {
        "id": review.id,
        "movie_id": review.movie_id,
        "result": evaluation
    }
