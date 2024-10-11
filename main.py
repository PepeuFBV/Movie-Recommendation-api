import os
import string
from fastapi import FastAPI
from surprise import SlopeOne, Reader, Dataset
import pandas as pd
import joblib

app = FastAPI()

model_path = 'model/slope_one_model.pkl'
# initialize the model and save it to a file
@app.get("/init")
async def init():

    # save the model to a file
    joblib.dump(train_model(), model_path)

    return {"message": "Model initialized and trained"}


# make a prediction using the model
@app.get("/predict")
async def predict(user_id: int, movie_id: int):
    # load the model from the file
    with open(model_path, 'rb') as f:
        model = joblib.load(f)

    # make a prediction
    prediction = model.predict(user_id, movie_id)

    return {"prediction": prediction.est}


# gets user id and movie id and rating and updates the model
@app.post("/rate_movie")
async def rate_movie(user_email: string, movie_id: int, rating: float):
    user_id = get_user_id(user_email) # get the user id

    # load the ratings data
    ratings = pd.read_csv('ratings_10.csv')

    # add the new rating to the ratings data
    new_rating = pd.DataFrame({'userId': [user_id], 'movieId': [movie_id], 'rating': [rating]})
    ratings = ratings.append(new_rating)

    # save the new ratings data
    ratings.to_csv('ratings_10.csv', index=False)


@app.get("/retrain")
async def retrain():

    # save the model to a file
    joblib.dump(train_model(), model_path)

    return {"message": "Model retrained"}




def train_model():

    # ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True) # will create the directory if it does not exist

    # load the ratings data
    ratings = pd.read_csv('ratings_10.csv')

    # define a Reader and load the data into a Dataset
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    model = SlopeOne()
    model.fit(data.build_full_trainset())

    # save the model to a file
    joblib.dump(model, model_path)

    return model


user_path = "users.csv"
def get_user_id(user_email):

    # ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True) # will create the directory if it does not exist

    # compare email to the emails in the users.csv file
    users = pd.read_csv(user_path)
    user = users[users['email'] == user_email]
    if len(user) == 0: # create a new user

        # find the biggest userID value in ratings file
        ratings = pd.read_csv('ratings_cleaned.csv')
        max_user_id = ratings['userId'].max()
        new_user_id = max_user_id + 1

        # save the new user to the users.csv file (email + new_user_id)
        new_user = pd.DataFrame({'email': [user_email], 'userId': [new_user_id]})
        users = users.append(new_user)
        users.to_csv(user_path, index=False)

        user_id = new_user_id

    else:
        user_id = user['userId'].values[0]

    return user_id
