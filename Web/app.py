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
        # ----------------------------------------
        # parse input from form
        # ----------------------------------------
        #  time info
        date = request.form.get('date')
        parsed_date = datetime.strptime(date, '%m/%d/%Y')

        scheduled_departure_time = request.form.get('departure_time')  #03:45 (ex)
        scheduled_am_pm = request.form.get('am_pm')  #am or pm
        datetime_str = f'{scheduled_departure_time} {scheduled_am_pm}'
        parsed_time = datetime.strptime(datetime_str, "%I:%M %p")
        total_minutes = parsed_time.hour * 60 + parsed_time.minute

        flight_length_min = request.form.get('flight_length_min')  #scheduled flight length in minutes

        # Airline/Plane info
        carrier_code = request.form.get('carrier_code')  #airline
        dest_airport = request.form.get('dest_airport')
        manufacturer = request.form.get('manufacturer')
        plane_model = request.form.get('plane_model')

        aircraft_age = request.form.get('aircraft_age')
        aircraft_age_missing = 0  #default

        engine_type = request.form.get('engine_type')
        seat_cnt = request.form.get('seat_cnt')
        builder_certificated = request.form.get('builder_certificated')  #true or false

        # Origin: Weather Info
        precip_accum_1_hr = request.form.get('precip_accum_1_hr')
        precip_accum_6_hr = request.form.get('precip_accum_6_hr')
        air_temp = request.form.get('air_temp')
        dew_point_temp = request.form.get('dew_point_temp')
        relative_humidity = request.form.get('relative_humidity')
        wind_speed = request.form.get('wind_speed')
        wind_direction = request.form.get('wind_direction')
        wind_gust = request.form.get('wind_gust')
        visibility = request.form.get('visibility')

        ceiling = request.form.get('ceiling')
        ceiling_missing = 0  #default

        sea_level_pressure = request.form.get('sea_level_pressure')
        sea_level_pressure_missing = 0  #default


        # dest_airport = request.form.get('dest_airport')
        # weather = request.form.get('weather')
        
        # ----------------------------------------
        # convert cyclical variables
        # ----------------------------------------
        # time related 
        month_sin = math.sin(2 * math.pi * (parsed_date.month / 12.0))
        month_cos = math.cos(2 * math.pi * (parsed_date.month / 12.0))

        day_sin = math.sin(2 * math.pi (parsed_date.day / 31.0))
        day_cos = math.cos(2 * math.pi (parsed_date.day / 31.0))

        dow_sin = math.sin(2 * math.pi * (parsed_date.weekday() / 7.0))
        dow_cos = math.cos(2 * math.pi * (parsed_date.weekday() / 7.0))

        minutes_sin = math.sin(2 * math.pi * total_minutes / 1440.0)
        minutes_cos = math.cos(2 * math.pi * total_minutes / 1440.0)

        # wind direction (360 degrees)
        wind_rad = np.deg2rad(wind_direction)
        wind_dir_sin = np.sin(wind_rad)
        wind_dir_cos = np.cos(wind_rad)

        # ----------------------------------------
        # convert into a DF
        # ----------------------------------------
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
            "Scheduled Elapsed Time": int(flight_length_min),
            "Carrier Code": carrier_code,
            "Destination Airport": dest_airport,
            "Manufacturer": manufacturer,
            "Model": plane_model,
            "Aircraft Age": int(aircraft_age),
            "Aircraft Age Missing": int(aircraft_age_missing),
            "Type of Engine": engine_type,
            "Number of Seats": seat_cnt,
            "Builder Type Certificated": int(builder_certificated),
            "Precipitation Accumulation One Hour": float(precip_accum_1_hr),
            "Precipitation Accumulation Six Hours": float(precip_accum_6_hr),
            "Air Temperature": float(air_temp),
            "Dew Point Temperature": float(dew_point_temp),
            "Relative Humidity": float(relative_humidity),
            "Wind Speed": float(wind_speed),
            "Wind Direction (sin)": wind_dir_sin,
            "Wind Direction (cos)": wind_dir_cos,
            "Wind Gust": float(wind_gust),
            "Visibility": float(visibility),
            "Ceiling": int(ceiling),
            "Ceiling Missing": int(ceiling_missing),
            "Sea Level Pressure": float(sea_level_pressure),
            "Sea Level Pressure Missing": int(sea_level_pressure_missing)
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