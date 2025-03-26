#############################################
# Setup
#############################################
from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import joblib
from datetime import datetime
import math
import pandas as pd


app = Flask(__name__)

# load for predictions
preprocessor = joblib.load('preprocessor.joblib')
model = tf.keras.models.load_model('flight_delays_sequential_3.keras')
#model = tf.keras.models.load_model('flight_delays.h5')


#############################################
# Routes
#############################################

@app.route('/', methods=['GET'])
def home():
    '''
    Render form on homepage
    '''
    return render_template('dummy.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    Get user inputs, preprocess inputs, give to model, and return prediction
    '''
    try:
        # parse input from form
        date = request.form.get('date')
        parsed_date = datetime.strptime(date, '%m/%d/%Y')

        scheduled_departure_time = request.form.get('departure_time')  #03:45 (ex)
        scheduled_am_pm = request.form.get('am_pm')  #am or pm
        datetime_str = f'{scheduled_departure_time} {scheduled_am_pm}'
        parsed_time = datetime.strptime(datetime_str, "%I:%M %p")
        total_minutes = parsed_time.hour * 60 + parsed_time.minute

        # dest_airport = request.form.get('dest_airport')
        # weather = request.form.get('weather')
        

        # convert cyclical variables
        month_sin = math.sin(2 * math.pi * (parsed_date.month / 12.0))
        month_cos = math.cos(2 * math.pi * (parsed_date.month / 12.0))

        day_sin = math.sin(2 * math.pi (parsed_date.day / 31.0))
        day_cos = math.cos(2 * math.pi (parsed_date.day / 31.0))

        dow_sin = math.sin(2 * math.pi * (parsed_date.weekday() / 7.0))
        dow_cos = math.cos(2 * math.pi * (parsed_date.weekday() / 7.0))

        minutes_sin = math.sin(2 * math.pi * total_minutes / 1440.0)
        minutes_cos = math.cos(2 * math.pi * total_minutes / 1440.0)



        # convert into a dict
        row_dict = {
            "Month (sin)": month_sin,
            "Month (cos)": month_cos,
            "Day (sin)": day_sin,
            "Day (cos)": day_cos,
            "Day of Week (sin)": dow_sin,
            "Day of Week (cos)": dow_cos,
            "Scheduled Departure Total Minutes (sin)": minutes_sin,
            "Scheduled Departure Total Minutes (cos)": minutes_cos,
        }

        # convert all variables into a df
        row_df = pd.DataFrame([row_dict])

        # convert inputs as needed
        user_input = [[dest_airport, weather]]

        # preprocessor
        #input_processed = preprocessor.transform(user_input)


        # run prediction to return
        #result = model.predict(input_processed)
        #return render_template('dummy.html', prediction=str(result[0]))
        return render_template('dummy.html', dest_airport)
    
    except Exception as e:
        # handle error
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # start flask app
    app.run(debug=True)