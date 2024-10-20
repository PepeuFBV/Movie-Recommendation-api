import os
from fastapi import FastAPI
from surprise import SlopeOne, Reader, Dataset
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model_path = 'model/slope_one_model.pkl'

@app.get("/init")
async def init():
    print("Initializing model")
    joblib.dump(train_model(), model_path)
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
    user_id = get_user_id(request.user_id)
    ratings = pd.read_csv('ratings_10.csv')
    new_rating = pd.DataFrame({'userId': [user_id], 'movieId': [request.movie_id], 'rating': [
        request.rating]})
    ratings = pd.concat([ratings, new_rating], ignore_index=True)
    ratings.to_csv('ratings_10.csv', index=False)
    joblib.dump(train_model(), model_path)
    print("Rating added and model retrained")
    return {"message": "Rating added and model retrained"}

class PredictMoviesRequest(BaseModel):
    user_id: str
    number_of_movies: int

@app.post("/predict_movies")
async def predict_movies(request: PredictMoviesRequest):
    print("Predicting movies for user: ", request.user_id, " number of movies: ", request.number_of_movies)
    user_id = get_user_id(request.user_id)
    model = joblib.load(model_path)
    # get all unique movie ids from the ratings file
    movies_ids = pd.read_csv('ratings_10.csv')['movieId'].unique()
    predictions = [model.predict(user_id, movie_id) for movie_id in movies_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movies = predictions[:request.number_of_movies]
    top_movie_ids = [int(prediction.iid) for prediction in top_movies]  # Convert numpy.int64 to int
    return top_movie_ids



@app.get("/retrain")
async def retrain():
    joblib.dump(train_model(), model_path)
    return {"message": "Model retrained"}



def train_model():
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    ratings = pd.read_csv('ratings_10.csv')
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

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    users = pd.read_csv(user_path)
    user = users[users['user_id'] == user_id]
    if len(user) == 0:
        ratings = pd.read_csv('ratings_cleaned.csv')
        max_user_id = ratings['userId'].max()
        new_model_id = max_user_id + 1
        new_user = pd.DataFrame({'user_id': [user_id], 'model_id': [new_model_id]})
        users = pd.concat([users, new_user], ignore_index=True)
        users.to_csv(user_path, index=False)
        model_id = new_model_id
    else:
        model_id = user['model_id'].values[0]
    return model_id