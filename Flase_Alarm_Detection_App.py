from flask import Flask, jsonify, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Creating Flask application
app = Flask(__name__)

# Load and preprocess data
df = pd.read_excel('Historical Alarm Cases.xlsx')
df = df.drop(columns=['Case No.', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10'])
df = df.dropna(axis=1, how='any')

# Renaming the columns for match the json data we get for test the trained model
df = df.rename(columns={
    'Ambient Temperature( deg C)': 'Ambient Temperature',
    'Calibration(days)': 'Calibration',
    'Unwanted substance deposition(0/1)': 'Unwanted substance deposition',
    'Humidity(%)': 'Humidity',
    'H2S Content(ppm)': 'H2S Content',
    'detected by(% of sensors)': 'detected by'

})
# Extract features and target
X = df[['Ambient Temperature', 'Calibration',
          'Unwanted substance deposition', 'Humidity',
          'H2S Content', 'detected by']]
y = df['Spuriosity Index(0/1)']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model for predictions of alarm detection
model = LogisticRegression()
model.fit(X_train, y_train)

@app.route('/train_model')
def train():
    global model,scaler
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return "Model trained successfully"

@app.route('/test_model', methods=['POST'])
def test():
    test_data = request.get_json()
    features = pd.DataFrame([[                     #here i also able to use np.array for convert this in array formate
        test_data['Ambient Temperature'],          #instead of pd.DataFrame but for avoid some error in run
        test_data['Calibration'],                  #command i used pd.dataframe with columns .
        test_data['Unwanted substance deposition'],
        test_data['Humidity'],
        test_data['H2S Content'],
        test_data['detected by']
    ]],columns=['Ambient Temperature', 'Calibration', 'Unwanted substance deposition', 'Humidity', 'H2S Content', 'detected by'])
     #in this dataFrame when i send the test data request from postman
    # i got error in my run command so i entered columns here so those error is solved that purpose i used columns here

   #Array based code:

# @app.route('/test_model', methods=['POST'])
# def test():
#     test_data = request.get_json()
#     features = np.array([[
#         test_data['Ambient Temperature'],
#         test_data['Calibration'],
#         test_data['Unwanted substance deposition'],
#         test_data['Humidity'],
#         test_data['H2S Content'],
#         test_data['detected by']
#     ]])

    features = scaler.transform(features)
    # Predictions of the Standardize features using the trained model
    prediction = model.predict(features)[0]

    if prediction == 1:
        return jsonify({"message": "False Alarm, No Danger"})
    else:
        return jsonify({"message": "True Alarm, Danger"})


app.run(port=5050)
