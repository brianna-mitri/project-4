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
model = tf.keras.models.load_model('NN_model_binomial.keras')
#model = tf.keras.models.load_model('flight_delays.h5')


#############################################
# Routes
#############################################

@app.route('/', methods=['GET'])
def home():
    '''
    Render static home page
    '''
    return render_template('index.html')

# predict page: GET to show form, and POST for inference
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    Get user inputs, preprocess inputs, give to model, and return prediction
    '''
    if request.method == 'GET':
        # render form page
        return render_template('Predictive_Model.html')
    else: 
        # handle form submission
        try:
            # ----------------------------------------
            # parse input from form
            # ----------------------------------------
            #  time info
            departure_date = request.form.get('departure_date')
            parsed_date = datetime.strptime(departure_date, '%Y-%m-%d')
            # parsed_date = datetime.strptime(departure_date, '%m/%d/%Y')

            # departure time (minutes)
            scheduled_departure_time = request.form.get('departure_time')  #14:45 (ex)
            #scheduled_am_pm = request.form.get('am_pm')  #am or pm
            #datetime_str = f'{scheduled_departure_time} {scheduled_am_pm}'
            parsed_time = datetime.strptime(scheduled_departure_time, "%H:%M")
            
            # total flight length in minutes
            total_minutes = parsed_time.hour * 60 + parsed_time.minute
            # flight_length_min = request.form.get('flight_length_min')  #scheduled flight length in minutes
            arrival_time = request.form.get('arrival_time')
            parsed_arrival_time = datetime.strptime(arrival_time, "%H:%M")
            parsed_arrival_mins = parsed_arrival_time.hour * 60 + parsed_arrival_time.minute

            if parsed_arrival_mins < total_minutes:
                # add day's worth of minutes if arrival time is the next day
                parsed_arrival_mins += 24 * 60

            flight_length_min = parsed_arrival_mins - total_minutes  #get expected flight length

            # Airline/Plane info
            carrier_code = request.form.get('carrier_code').strip().upper()  #airline
            dest_airport = request.form.get('dest_airport').strip().upper()
            manufacturer = request.form.get('manufacturer').strip().upper()
            plane_model = request.form.get('plane_model').strip()

            aircraft_age = request.form.get('aircraft_age').strip()
            aircraft_age_missing = 0  #default

            engine_type = request.form.get('engine_type').strip()   #.upper???? 
            seat_cnt = request.form.get('seat_cnt').strip()
            builder_certificated = request.form.get('builder_certificated')  #true or false

            # Origin: Weather Info
            precip_accum_1_hr = request.form.get('precip_accum_1_hr').strip()
            precip_accum_6_hr = request.form.get('precip_accum_6_hr').strip()
            air_temp = request.form.get('air_temp').strip()
            dew_point_temp = request.form.get('dew_point_temp').strip()
            relative_humidity = request.form.get('relative_humidity').strip()
            wind_speed = request.form.get('wind_speed').strip()
            wind_direction = request.form.get('wind_direction').strip()
            wind_gust = request.form.get('wind_gust').strip()
            visibility = request.form.get('visibility').strip()

            ceiling = request.form.get('ceiling').strip()
            ceiling_missing = 0  #default

            sea_level_pressure = request.form.get('sea_level_pressure').strip()
            sea_level_pressure_missing = 0  #default

            # Destination: Weather Info
            dest_precip_accum_1_hr = request.form.get('dest_precip_accum_1_hr').strip()
            dest_precip_accum_6_hr = request.form.get('dest_precip_accum_6_hr').strip()
            dest_air_temp = request.form.get('dest_air_temp').strip()
            dest_dew_point_temp = request.form.get('dest_dew_point_temp').strip()
            dest_relative_humidity = request.form.get('dest_relative_humidity').strip()
            dest_wind_speed = request.form.get('dest_wind_speed').strip()
            dest_wind_direction = request.form.get('dest_wind_direction').strip()
            dest_wind_gust = request.form.get('dest_wind_gust').strip()
            dest_visibility = request.form.get('dest_visibility').strip()

            dest_ceiling = request.form.get('dest_ceiling').strip()
            dest_ceiling_missing = 0  #default

            dest_sea_level_pressure = request.form.get('dest_sea_level_pressure').strip()
            dest_sea_level_pressure_missing = 0  #default
            
            # ----------------------------------------
            # convert cyclical variables
            # ----------------------------------------
            # time related 
            month_sin = math.sin(2 * math.pi * (parsed_date.month / 12.0))
            month_cos = math.cos(2 * math.pi * (parsed_date.month / 12.0))

            day_sin = math.sin(2 * math.pi * (parsed_date.day / 31.0))
            day_cos = math.cos(2 * math.pi * (parsed_date.day / 31.0))

            dow_sin = math.sin(2 * math.pi * (parsed_date.weekday() / 7.0))
            dow_cos = math.cos(2 * math.pi * (parsed_date.weekday() / 7.0))

            minutes_sin = math.sin(2 * math.pi * total_minutes / 1440.0)
            minutes_cos = math.cos(2 * math.pi * total_minutes / 1440.0)

            # wind direction (360 degrees)
            wind_rad = np.deg2rad(float(wind_direction))
            wind_dir_sin = np.sin(wind_rad)
            wind_dir_cos = np.cos(wind_rad)

            dest_wind_rad = np.deg2rad(float(dest_wind_direction))
            dest_wind_dir_sin = np.sin(dest_wind_rad)
            dest_wind_dir_cos = np.cos(dest_wind_rad)

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
                "Sea Level Pressure Missing": int(sea_level_pressure_missing),
                "Destination Precipication Accumulation One Hour": float(dest_precip_accum_1_hr),
                "Destination Precipitation Six Hours": float(dest_precip_accum_6_hr),
                "Destination Air Temperature": float(dest_air_temp),
                "Destination Dew Point Temperature": float(dest_dew_point_temp),
                "Destination Relative Humidity": float(dest_relative_humidity),
                "Destination Wind Speed": float(dest_wind_speed),
                "Destination Wind Direction (sin)": dest_wind_dir_sin,
                "Destination Wind Direction (cos)": dest_wind_dir_cos,
                "Destination Wind Gust": float(dest_wind_gust),
                "Destination Visibility": float(dest_visibility),
                "Destination Ceiling": int(dest_ceiling),
                "Destination Ceiling Missing": int(dest_ceiling_missing),
                "Destination Sea Level Pressure": float(dest_sea_level_pressure),
                "Destination Sea Level Pressure Missing": int(dest_sea_level_pressure_missing)
            }

            # convert all variables into a df
            input_df = pd.DataFrame([row_dict])

            # ----------------------------------------
            # Preprocess
            # ----------------------------------------
            # preprocessor
            input_processed = preprocessor.transform(input_df)


            # run prediction to return
            result = model.predict(input_processed)  #delay probability
            delay_prob = result[0][0]

            if delay_prob >= 0.5:
                prediction = 'Delay.'
            else: 
                prediction = 'No delay!'

            return render_template('Predictive_Model.html', prediction=f"{prediction} {str(round(delay_prob * 100, 2))}% chance of delay")
            #return render_template('dummy.html', prediction=f"string{delay_prob}")
            #return render_template('dummy.html', prediction=date)

        except Exception as e:
            
            # handle error
            return render_template('Predictive_Model.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # start flask app
    app.run(debug=True)