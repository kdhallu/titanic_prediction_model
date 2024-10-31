import sys
import pickle
import json
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the path to the model
model_path = os.path.join(script_dir, 'titanic_model.pkl')

# Load the model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def predict(data):
    # Encode the categorical variable 'sex'
    sex_mapping = {"male": 1, "female": 0}

    # Prepare the data for prediction
    # Ensure you only include the features your model expects
    # Example features: age, sex (encoded), and fare
    data_array = [[
        data["age"],
        sex_mapping[data["sex"]],  # Convert 'sex' to numerical
        data["fare"]
    ]]  # Wrap in another list for 2D array

    prediction = model.predict(data_array)
    return prediction[0]

# Get the input data from Node.js
if __name__ == "__main__":
    input_data = json.loads(sys.argv[1])
    result = predict(input_data)
    print(json.dumps({"prediction": int(result)}))
