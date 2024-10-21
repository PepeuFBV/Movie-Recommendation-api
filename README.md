# Movie Recommendation API

This is a simple API that recommends movies based on the user's previous ratings. It uses the MovieLens dataset and a collaborative filtering algorithm to make recommendations. The API is used in the [Kotflix](https://github.com/PepeuFBV/KotFlix) project to recommend movies to users.

The chosen model and data for the project were obtained in the [Movie Recommendation](https://github.com/PepeuFBV/Movie-Recommendation) repository. Just get the model, data and the model's best params to insert in the API.

## Installation

To install the API, you need to have Python 3.11 or higher installed on your machine. You can download it [here](https://www.python.org/downloads/).

After installing Python, you can clone the repository and install the dependencies by running the following commands:

Clone the repository:
```bash
git clone https://github.com/PepeuFBV/Movie-Recommendation-api.git
```

Change to the project directory:
```bash
cd Movie-Recommendation-api
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
 
## Usage

To start the API, you can run the following command:

```bash
python main.py
```

The API will start running on `http://127.0.0.1:8000/`. Tou can start by calling the `/init` endpoint to load the model and best params.

## Endpoints

The API has the following endpoints:

- `/init`: Load the model and best params.
- `/predict`: Predict the rating for a given user and movie.
- `/rate_movie`: Add a new rating for a movie by a user and retrain the model.
- `/predict_movies`: Predict the top N movies for a given user.
- `/retrain`: Retrain the model with the current ratings.

## Example Requests

### Initialize the Model

```bash
curl -X GET "http://127.0.0.1:8000/init"
```

### Predict a Rating

```bash
curl -X GET "http://127.0.0.1:8000/predict?user_id=1&movie_id=10"
```

### Rate a Movie

```bash
curl -X POST "http://127.0.0.1:8000/rate_movie" -H "Content-Type: application/json" -d '{"user_id": "user123", "movie_id": 10, "rating": 4.5}'
```

### Predict Top Movies

```bash
curl -X POST "http://127.0.0.1:8000/predict_movies" -H "Content-Type: application/json" -d '{"user_id": "user123", "number_of_movies": 5}'
```

### Retrain the Model

```bash
curl -X GET "http://127.0.0.1:8000/retrain"
```

## Data files

- ratings_10.csv: Contains the ratings data used for training and predictions.
- users.csv: Contains the mapping of user IDs to model-specific user IDs.
- model/slope_one_model.pkl: The trained SlopeOne model.

## Contributing

If you want to contribute to the project, you can open an issue or submit a pull request. Any help is appreciated!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
