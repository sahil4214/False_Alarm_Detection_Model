# False_Alarm_Detection_Model 

* After running the code in pycharm go to the postman and create link of your model to cennect with it  like : local server/model name : go to the row then select json and import json data 

This Model Developed for chemical industry which has installed alarms on their different section part they installed alarms but problem is that some time some alarm if gas not leakaged but false alarm tune so they need to call management team that having more cost to company so it cot. me and tell me to develop proper false alarm detection model.

#problem : 
False Alarm Detection
Problem Statement-This project was made for a chemical industry which had sensors installed in various parts of the factory to detect H2S gas which is hazardous to health. Every time one or multiple sensors detected the H2S leak, an emergency alarm rings to alert the workers. For every alarm, the industry calls a team which sanitizes the place and checks for the leak and this was a big cost to the company.
A few of the alarms that ring are not even hazardous. The company gave us the data for each alarm with a final column stating the alarm was dangerous or not.
Ambient Temperature	Calibration(days)	Unwanted substance deposition (0/1)	Humidity (%)	H2S Content(ppm)	Dangerous (0/1)

The data was first pre-processed and analysis libraries like Numpy and Pandas were used to make it ready to be utilized by a machine learning algorithm.
Problems like standard scaling, categorical data and missing values were handled with appropriate techniques.
Then, we used LogisticRegression model to make a classifier with first five column as independent columns and dangerous column as dependent/target column.
Now whenever, there is a leakage and the alarm rings, the data is sent to us and we predict if it is dangerous or not. If found dangerous then only the team is called to sanitize the place and fix the leak. This saved a lot of money for the company. 
#Solution :
	PROBLEM STATEMENT: 
•	This project was made for a chemical industry with sensors installed in various parts of the factory to detect H2S gas that is hazardous to health. Whenever one or multiple sensors detect the H2S leak, an emergency alarm rings to alert the workers. For every alarm, the industry calls a team that sanitizes the place and checks for the leak and this was a big cost to the company.


	IMPORTING NECESSARY LIBRARIES:

 


	CREATING FLASK APP. ALSO LOAD AND PREPROCESS THE DATA : 


 

	RENAMING COLUMNS :

 

	EXTRACTING FEATURES AND TARGET COLUMNS:

 

•	X: Features used for prediction. (INDEPENDENT VAR.)
•	y: Target variable indicating the spuriosity index. (DEPENDENT VAR.)

	SPLITTING THE DATA, STANDARDIZED FEATURES, TRAIN MODEL:

 
•	train_test_split: Splits the data into training and testing sets, with 30% of the data used for testing and 70% in training.
•	StandardScaler: Standardizes features by removing the mean and scaling to unit variance.
•	scaler.fit_transform: Fits the scaler on the training data and then transforms it.
•	scaler.transform: Transforms the testing data using the fitted scaler.
•	LogisticRegression: Initializes the Logistic Regression model.
•	model.fit: Trains the model using the training data.


	TRAINING THE MODEL:

 

•	@app.route('/train_model'): Defines a route for training the model.
•	train(): Function to train the model and return a success message.

	TESTING THE MODEL:

 

•	@app.route('/test_model', methods=['POST']): Defines a route for testing the model using POST requests.
•	test(): Function to handle the test request.
•	request.get_json(): Retrieves JSON data from the request.
•	pd.DataFrame(): Converts the test data into a DataFrame.
•	scaler.transform(): Standardizes the test data.
•	model.predict(): Predicts the outcome using the trained model.
•	jsonify(): Returns the prediction result in JSON format.

	RUNNING APP:

 


	TESTING WITH POSTMAN:
•	Test the Model:
Send a POST request to http://127.0.0.1:5000/test_model with the following JSON body:


 

 


•	FOR THE FIRST JSON DATA DURING TESTING I GET THE ANSWER: 
FALSE ALARM, NO DANGER 
•	FOR THE SECOND JSON DATA TESTING IN POSTMAN I GET THE  ANSWER: 
TRUE ALARM, DANGER

	QUERY THAT I HAVE: 
When I try to enter two testing JSON data entries at a time from Postman to check both data returned answers, I get an error.

 
	FINAL CODE:

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


