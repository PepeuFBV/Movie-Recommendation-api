from fastapi import FastAPI
from surprise import SlopeOne, Reader, Dataset
import pandas as pd
import pickle

app = FastAPI()

model_path = 'model/slope_one_model.pkl'

# initialize the model and save it to a file
@app.get("/init")
async def init():
    # load the ratings data
    ratings = pd.read_csv('ratings_cleaned.csv')

    # define a Reader and load the data into a Dataset
    reader = Reader()
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

    model = SlopeOne()
    model.fit(data.build_full_trainset())

    # save the model to a file
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    return {"message": "Model initialized"}

# make a prediction using the model
@app.get("/predict")
async def predict(user_id: int, movie_id: int):
    # load the model from the file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # make a prediction
    prediction = model.predict(user_id, movie_id)

    return {"prediction": prediction.est}