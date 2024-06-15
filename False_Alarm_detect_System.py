from flask import Flask, jsonify, request
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Creating Flask application
app = Flask(__name__)

# Load and preprocess data
df = pd.read_excel('Historical Alarm Cases.xlsx', engine='openpyxl')
df = df.drop(columns=['Case No.', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'])
df = df.dropna(axis=1, how='any')

# Renaming the columns to match the JSON data we get for testing the trained model
df = df.rename(columns={
    'Ambient Temperature( deg C)': 'Ambient Temperature',
    'Calibration(days)': 'Calibration',
    'Unwanted substance deposition(0/1)': 'Unwanted substance deposition',
    'Humidity(%)': 'Humidity',
    'H2S Content(ppm)': 'H2S Content',
    'detected by(% of sensors)': 'detected by'
})

# Extract features and target
X = df[['Ambient Temperature', 'Calibration', 'Unwanted substance deposition', 'Humidity', 'H2S Content', 'detected by']]
y = df['Spuriosity Index(0/1)']

# Create a pipeline with StandardScaler and LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Train the pipeline
pipeline.fit(X, y)

# Save the trained pipeline
with open('model_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

@app.route('/train_model')
def train():
    global pipeline
    pipeline.fit(X, y)
    with open('model_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    return "Model trained successfully"

@app.route('/test_model', methods=['POST'])
def test():
    try:
        # Receive JSON data
        test_data = request.get_json()

        # Check if the input is a list
        if not isinstance(test_data, list):
            return jsonify({"error": "Invalid input format, expected a list of JSON objects"}), 400

        # Convert JSON data to DataFrame
        features = pd.DataFrame(test_data)

        # Load the trained pipeline
        with open('model_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)

        # Standardize features and make predictions
        features = pipeline['scaler'].transform(features)
        predictions = pipeline['model'].predict(features)

        # Create result messages
        results = []
        for prediction in predictions:
            if prediction == 1:
                results.append({"message": "False Alarm, No Danger"})
            else:
                results.append({"message": "True Alarm, Danger"})

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5070)
