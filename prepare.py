import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the dataset
data = pd.read_csv('train.csv')

# Feature engineering (handle missing values, categorical variables)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Age'].fillna(data['Age'].mean(), inplace=True)  # Fill missing age with mean
data['Fare'].fillna(data['Fare'].mean(), inplace=True)  # Fill missing fare with mean
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)  # Fill missing Embarked with mode
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})  # Encode Embarked column

# Selecting more features (Pclass, Sex, Age, SibSp, Parch, Fare, Embarked)
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Check the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Save the model as a pickle
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)
