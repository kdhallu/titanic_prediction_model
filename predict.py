import pickle
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Tuple, Union

# Load the model
with open('titanic_model.pkl', 'rb') as f:
    model: RandomForestClassifier = pickle.load(f)

# Create a Flask application
app: Flask = Flask(__name__)

# Define the prediction endpoint
@app.route('/predict', methods=['GET'])
def predict() -> Union[Dict[str, Any], Tuple[Dict[str, str], int]]:
    # Get input data from the query parameters
    pclass = request.args.get('Pclass', type=int)
    sex = request.args.get('Sex', type=str)
    age = request.args.get('Age', type=float)
    sibsp = request.args.get('SibSp', type=int, default=0)  # Siblings/Spouses aboard
    parch = request.args.get('Parch', type=int, default=0)  # Parents/Children aboard
    fare = request.args.get('Fare', type=float, default=0.0)  # Fare paid
    embarked = request.args.get('Embarked', type=str, default='S')  # Port of embarkation

    # Validate input data
    if pclass is None or sex is None or age is None:
        return jsonify({"error": "Missing data in request"}), 400

    # Map the 'Sex' to numerical values
    sex_mapping: Dict[str, int] = {'male': 0, 'female': 1}
    sex_num: Union[int, None] = sex_mapping.get(sex.lower())

    if sex_num is None:
        return jsonify({"error": "Invalid value for Sex, should be 'male' or 'female'"}), 400

    # Map 'Embarked' to numerical values
    embarked_mapping: Dict[str, int] = {'S': 0, 'C': 1, 'Q': 2}
    embarked_num: Union[int, None] = embarked_mapping.get(embarked.upper(), 0)  # Default to 'S' if not found

    # Prepare the data for prediction with additional features
    input_features: list[list[Union[int, float]]] = [[pclass, sex_num, age, sibsp, parch, fare, embarked_num]]

    # Make the prediction
    prediction: int = model.predict(input_features)[0]

    # Return the prediction result
    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True, port=8001)
