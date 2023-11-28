#!/usr/bin/env python
# coding: utf-8

# In[3]:


from flask import Flask, render_template_string, request
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify
import pandas as pd
from threading import Thread
import time
import pickle

app = Flask(__name__, static_url_path='/static')

# Load the pre-trained model
model = pickle.load(open('model.pkl','rb'))

# Load the scaler using pickle
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
    
html_template = """

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML Diabetes Prediction</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
        }
        .header {
            display: flex;
            align-items: center; /* Center vertically */
            justify-content: center; /* Center horizontally */
            background-color: #ffe5ec; /* Light pale pastel pink color */
            padding: 20px;
            text-align: center;
            height: 100vh;
        }

        .header img {
            max-width: 100%;
            height: auto;
            margin-right: 20px; /* Adjust as needed */
        }
        h1 {
            font-size: 60px; /* Adjust font size for the title */
            margin: 0; /* Remove default margin */
        }
        .text {
             text-align: left;
             max-width: 50%; /* Adjust as needed */
             color: #ff8fab;
        }
         h2 {
            text-align: center;
        }

        form {
            max-width: 400px;
            margin: 20px auto;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        }

        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
        }

        input {
            width: 100%;
            padding: 8px;
            margin-bottom: 15px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
        }

        button[type="submit"] {
            background-color: #4caf50;
            color: white;
            cursor: pointer;
        }

        button[type="submit"]:hover {
            background-color: #45a049;
        }

        p {
            margin-top: 20px;
            color: #333;
        }

       .button-container {
        margin-top: 20px; /* Add some space between the paragraph and the button */
    }

    button {
        background-color: #ff8fab;
        color: white;
        cursor: pointer;
        padding: 8px 16px; /* Adjust padding as needed */
        border: none;
        border-radius: 4px;
        margin-left: auto; /* This will push the button to the right */
    }

    </style>
</head>

<body>
    <div class="header">
        <img src="{{ url_for('static', filename='diabetes.png') }}" alt="Diabetes Image">
        <div class="text">
            <h1>ML Diabetes Prediction</h1>
            <p>Diabetes is a chronic disease affecting millions worldwide. It occurs when the pancreas fails to 
            produce sufficient insulin or when the body struggles to effectively utilize the insulin it generates
            . Our predictive model uses logistic regression to assess an individual's risk of developing diabetes based on 
            six key inputs. Our model aims to contribute to the early 
            detection and management of diabetes, thereby mitigating its potentially devastating effects on
            health.</p>
            <div class="button-container">
                <button type="button">Learn More</button>
            </div>
        </div>
    </div>


    <h2>Please enter the following data:</h2>

    <form method="post" action="/predict">
        <!-- dropdown select inputs -->
        <label for="input1">Gender:</label>
        <select name="input1" id="gender" required>
            <option value="Male">Male</option>
            <option value="Female">Female</option>
        </select>

        <!-- we should probably change some of the questions so that its more congruent with the dropdown select options
        (which are congruent how we encoded the dataframe for Log_Reg_Model) ex. "I have be diagnosed with hypertension: "-->

        <label for="input3">Hypertension:</label>
        <select name="input3" id="hypertension" required>
            <option value="True">True</option>
            <option value="False">False</option>
        </select>

        <label for="input4">Heart Disease:</label>
        <select name="input4" id="heartdisease" required>
            <option value="True">True</option>
            <option value="False">False</option>
        </select>

        <!-- text inputs -->
        <label for="input2">Age:</label>
        <input type="text" name="input2" required>

        <label for="input5">BMI:</label>
        <input type="text" name="input5" required>

        <label for="input6">HbA1c Level:</label>
        <input type="text" name="input6" required>

        <label for="input7">Blood Glucose Level:</label>
        <input type="text" name="input7" required>

        <!-- ChatGPT said that the button (used to be input) was not closed -->
        <button type="submit" value="Predict">Predict</button>
    </form>

    {% if prediction is defined %}
        <p>Prediction: {{ prediction }}</p>
    {% endif %}
</body>

</html>

"""

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # dropdown selects
            # get input values from the form and convert to what Log_Reg_Model sees
            if str(request.form['input1']) == "Male":
                input1 = 1 
            else:
                input1 = 0

            if str(request.form['input3']) == "True":
                input3 = 1 
            else:
                input3 = 0

            if str(request.form['input4']) == "True":
                input4 = 1 
            else:
                input4 = 0
            # input3 = str(request.form['input3'])
            # input4 = str(request.form['input4'])

            # text
            input2 = float(request.form['input2'])
            input5 = float(request.form['input5'])
            input6 = float(request.form['input6'])
            input7 = float(request.form['input7'])
            
            # Create a DataFrame with the input values
            input_data = pd.DataFrame({
                'age': [input2],
                'bmi': [input5],
                'HbA1c_level': [input6],
                'blood_glucose_level': [input7],
                'hypertension_1': [input3],
                'heart_disease_1': [input4],
                'gender_Male': [input1],
            })

            # Print the modified DataFrame
            print("Input Data:")
            print(input_data)

            # Rescale features using the loaded Min-Max scaler
            new_data_rescaled = scaler.transform(input_data)
            new_data_rescaled_df = pd.DataFrame(data=new_data_rescaled, columns=input_data.columns)

            print("Scaled Input Data:")
            print(new_data_rescaled)

            # Make a prediction using the pre-trained model
            prediction = model.predict(new_data_rescaled_df)
            print("Prediction:")
            print(prediction)


            # Display the prediction
            print("Prediction for the new input:", prediction[0])
            # Display the prediction on the web page
            return render_template_string(html_template, prediction=prediction[0])

        except ValueError as e:
            # Handle conversion errors
            return jsonify({"error": str(e)})

if __name__ == '__main__':

    flask_thread = Thread(target=app.run)
    flask_thread.start()

    # Allow some time for the Flask app to start
    time.sleep(2)

    # Additional code for other tasks if needed

    # Join the Flask thread to the main thread
    flask_thread.join()


# In[ ]:




