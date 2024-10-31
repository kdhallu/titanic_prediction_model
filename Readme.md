Titanic Survival Prediction API

This project provides a REST API for predicting Titanic passenger survival using a pre-trained machine learning model. The model was trained on historical Titanic data and uses passenger features to make predictions about survival probability.

Features

	•	Model: Utilizes a RandomForestClassifier for prediction.
	•	API Framework: Flask provides the REST API endpoint.
	•	Input Fields: Takes in features like passenger class, gender, age, siblings/spouses aboard, parents/children aboard, fare, and port of embarkation.

Requirements

	•	Python 3.x
	•	Flask
	•	Scikit-Learn
	•	Pickle

Installation

Clone the repository:

```
git clone <repository-url>
cd <repository-name>
```

2.	Install dependencies:

```
pip install -r requirements.txt
```


	2.	Prediction Endpoint:
	•	URL: /predict
	•	Method: GET
	•	Query Parameters:
	•	Pclass: Passenger class (1, 2, or 3).
	•	Sex: Passenger gender (male or female).
	•	Age: Passenger age as a floating point.
	•	SibSp: Number of siblings or spouses aboard.
	•	Parch: Number of parents or children aboard.
	•	Fare: Fare paid by the passenger.
	•	Embarked: Port of embarkation (S, C, or Q).
	•	Example Request:

GET http://localhost:8001/predict?Pclass=3&Sex=male&Age=22&SibSp=1&Parch=0&Fare=7.25&Embarked=S


A prediction of 1 means survival, and 0 means non-survival.

Additional Information

Model Training

The model was trained on the Titanic dataset using Scikit-Learn’s RandomForestClassifier. To retrain or customize the model, refer to train_model.py (if provided) or your own Jupyter Notebook with the Titanic dataset.

Error Handling

The API checks for:

	•	Missing or invalid values for required fields (Pclass, Sex, Age).
	•	Correct format for Sex (male or female) and Embarked (S, C, or Q).
