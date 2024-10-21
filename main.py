import os
from http.client import HTTPException
from fastapi import FastAPI, HTTPException
from surprise import SlopeOne, Reader, Dataset
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from threading import Lock
import asyncio
from queue import Queue
from contextlib import asynccontextmanager

model_path = 'model/slope_one_model.pkl'
model_lock = Lock()
model_training = False

rating_queue = Queue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(process_rating_queue())
    yield


async def process_rating_queue():
    while True:
        if not rating_queue.empty():
            request = rating_queue.get()
            await handle_rating_request(request)
        await asyncio.sleep(1)

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get("/init")
async def init():
    global model_training
    with model_lock:
        if model_training:
            raise HTTPException(status_code=409, detail="Model is already being trained")
        model_training = True
    try:
        print("Initializing model")
        joblib.dump(train_model(), model_path)
    finally:
        with model_lock:
            model_training = False
    return {"message": "Model initialized and trained"}



@app.get("/predict")
async def predict(user_id: int, movie_id: int):
    model = joblib.load(model_path)
    prediction = model.predict(user_id, movie_id)
    return {"prediction": prediction.est}



class RateMovieRequest(BaseModel):
    user_id: str
    movie_id: int
    rating: float

@app.post("/rate_movie")
async def rate_movie(request: RateMovieRequest):
    print("Received rating request: ", request)
    rating_queue.put(request)
    return {"message": "Rating request added to queue"}



class RateMovieRequest(BaseModel):
    user_id: str
    movie_id: int
    rating: float


async def handle_rating_request(request: RateMovieRequest):
    print("Processing rating request: ", request)
    user_id = get_user_id(request.user_id)
    ratings = pd.read_csv('ratings.csv')

    # Check if the rating already exists
    existing_rating = ratings[(ratings['userId'] == user_id) & (ratings['movieId'] == request.movie_id)]
    if not existing_rating.empty:
        # Update the existing rating
        ratings.loc[
            (ratings['userId'] == user_id) & (ratings['movieId'] == request.movie_id), 'rating'] = request.rating
    else:
        # Add a new rating
        new_rating = pd.DataFrame({'userId': [user_id], 'movieId': [request.movie_id], 'rating': [request.rating]})
        ratings = pd.concat([ratings, new_rating], ignore_index=True)

    ratings.to_csv('ratings.csv', index=False)

    global model_training
    with model_lock:
        if not model_training:
            model_training = True
            try:
                joblib.dump(train_model(), model_path)
                print("Rating added and model retrained")
            finally:
                model_training = False



class PredictMoviesRequest(BaseModel):
    user_id: str
    number_of_movies: int

@app.post("/predict_movies")
async def predict_movies(request: PredictMoviesRequest):
    print("Predicting movies for user: ", request.user_id, " number of movies: ", request.number_of_movies)
    user_id = get_user_id(request.user_id)
    model = joblib.load(model_path)
    # get all unique movie ids from the ratings file
    movies_ids = pd.read_csv('ratings.csv')['movieId'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = predictions[:request.number_of_movies]
    top_movie_ids = []
    for prediction in top_movies:
        movie_id = int(prediction.iid)  # Convert numpy.int64 to int
        estimated_rating = prediction.est
        print(f"Movie ID: {movie_id}, Estimated Rating: {estimated_rating}")
        top_movie_ids.append(movie_id)
    return top_movie_ids



@app.get("/retrain")
async def retrain():
    global model_training
    with model_lock:
        if model_training:
            raise HTTPException(status_code=409, detail="Model is already being trained")
        model_training = True
    try:
        joblib.dump(train_model(), model_path)
    finally:
        with model_lock:
            model_training = False
    return {"message": "Model retrained"}



def train_model():
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ratings = pd.read_csv('ratings.csv')
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    model = SlopeOne()
    model.fit(data.build_full_trainset())
    joblib.dump(model, model_path)
    return model



user_path = "users.csv"

def get_user_id(user_id):
    # check if the file exists, if not, create it with the necessary columns
    if not os.path.exists(user_path):
        users_df = pd.DataFrame(columns=['user_id', 'model_id'])
        users_df.to_csv(user_path, index=False)

    users = pd.read_csv(user_path)
    user = users[users['user_id'] == user_id]
    if len(user) == 0:
        ratings = pd.read_csv('ratings.csv')
        max_user_id = ratings['userId'].max()
        new_model_id = max_user_id + 1
        new_user = pd.DataFrame({'user_id': [user_id], 'model_id': [new_model_id]})
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(user_path, index=False)
        model_id = new_model_id
    else:
        model_id = user['model_id'].values[0]
    return model_id