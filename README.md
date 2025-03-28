# Predicting Flight Delays out of LAX
This project demonstrates supervised machine learning to predict if a flight out of LAX airport will be delayed based on factors such as day, time, aircraft model, and weather conditions. It is not representative of a consumer-level product, as the model uses information not readily available to passengers (such as aircraft specifications), but is instead directed toward the aviation industry for logistical purposes.

## Repository Directory
The repository is organized as follows. High-level folders contain further subdirectories to help with navigation.
|Folder|Description|
|---|---|
|Database|SQLite Database of modeling data|
|Machine_Learning|Machine learning models and results|
|Preprocessing|Jupyter Notebooks and outputs for preprocessing data|
|Resources|Source data files|
|Web|Web site files. Note that this site is not live. The Flask app `app.py` can be run locally to connect to the site.|
|Work|Collaborator work folders. ***Can be safely ignored.***|

### Important Files
The Jupyter Notebook with the **final model** can be found [here](https://github.com/brianna-mitri/project-4/blob/main/Machine_Learning/_Final/Flight_Delays_Binary_Neural_Net.ipynb).  
**Optimization** results in Excel format can be found [here](https://github.com/brianna-mitri/project-4/blob/main/Machine_Learning/Optimization%20Summary.xlsx).  
**Presentation** slides can be found [here](https://github.com/brianna-mitri/project-4/blob/main/Project_4-Group_1-Presentation-Predicting_Departure_Delays.pdf).  


### Zipped files
Note that several files (including the SQLite database) are zipped for this repository due to GitHub file size constraints. The Jupyter Notebooks will automatically read the zipped files, so there is no need to unzip them first.

# Analysis
## Overview

### Data
The data consists of 3 parts pulled from U.S. federal government web sites:
|Site|Data|
|---|---|
|[Bureau of Transportation Statistics](https://transtats.bts.gov/ONTIME/Departures.aspx)|Flights originating from LAX, 2020-2024|
|[Federal Aviation Administration](https://www.faa.gov/licenses_certificates/aircraft_certification/aircraft_registry/releasable_aircraft_download)|Aircraft data by tail number|
|[National Weather Service](https://www.weather.gov/lox/observations_historical)|Historical weather observations for LAX and destination airports|

Once compiled, the dataset consists of 829,906 rows. The full dataset (for visualization) has 71 columns. The modeling dataset has 48 columns.
## Results
### Data Preprocessing
Preprocessing consists of 3 stages:  
1. Cleaning weather data files with `1_cleaning_weather.ipynb`
   - Check for nulls and address them as appropriate
   - Data values are converted to machine learning-friendly formats
   - Create columns for origin (LAX) as well as destination (specific to flight)
2. Cleaning flight and aircraft data with `2_cleaning_flights.ipynb`
   - Flight and aircraft data are cleaned and merged by tail number
   - This data is then merged with the cleaned weather data
     - For LAX weather, the data was merged by datetime for the observation at or most recently before the scheduled departure time
     - For destination weather, the data merged similarly for the scheduled arrival time, based on the scheduled arrived time. This is a substitution for forecast data.
   - Two output files are created:
     - `flight_delays.zip` for creating visualizations with the full dataset
     - `modeling_data.zip` for machine learning (ML) models. This file lacks columns undesirable for ML (such as unique identifiers).
3. Splitting data with `3_splitting_data.ipynb`
   - Data is pre-split into training and testing data sets for efficiently running multiple models in a comparable fashion
     - `x_train.zip` & `y_train.zip` for training
     - `x_test.zip` & `y_test.zip` for testing
   - Optionally, data is rebalanced with `SMOTE`.
     - NOTE: The final product did not use this rebalancing as it did not produce better results.

### Machine Learning Models
Using the pre-split data, the following models were tested:

- Random Forest
- Logistic Regression
- K-Nearest Neighbors
- Neural Network

In addition to these, Support Vector Classification was attemped, but the model never successfully completed. As such, results from that model are omitted.

The models were tested against 3 configurations of target variables:

- Multiclass (6 classes)
- Multiclass (4 classes)
- Binary classification (Delayed / Not Delayed)

The multiclass models failed to achieve accuracy near the desired goal of 75%, so attention was directed to binary classification. The results of those tests are below.

#### Random Forest
Using default values for Random Forest, the model reached a testing accuracy of 0.8424, with training accuracy of 0.9999. This is overfitted, but hyperparameter adjustments to address the overfitting, determined by RandomSearchCV, resulted in underfitting.
|Configuration|Training Accuracy|Testing Accuracy|
|---|---|---|
|Default|0.9999|0.8424|
|"Optimized"|0.5970|0.5968|

#### Logistic Regression
The logistic regression model appears reasonable on its surface with 84% accuracy:

|Training Accuracy|Testing Accuracy|
|---|---|
|0.8397|0.8397|

But the Classification Report indicates otherwise with a recall of 0.00 for `Is Delayed` (1).

              precision    recall  f1-score   support

           0       0.84      1.00      0.91    139387
           1       0.48      0.00      0.00     26595

    accuracy                           0.84    165982

#### K-Nearest Neighbors
This model has accuracy similar to Logistic Regression, but with better prediction of delays.

|Training Accuracy|Testing Accuracy|
|---|---|
|0.8619|0.8223|

              precision    recall  f1-score   support

           0       0.85      0.95      0.90    139387
           1       0.36      0.13      0.20     26595

    accuracy                           0.82    165982

#### Neural Network
Using Tensorflow and automating the tuning of hyperparameters with Keras Tuner, these results for a neural network model were achieved:

|Training Accuracy|Testing Accuracy|
|---|---|
|0.8508|0.8369|

              precision    recall  f1-score   support

           0       0.85      0.96      0.90    139387
           1       0.40      0.13      0.19     26595

    accuracy                           0.83    165982

## Summary
Of the models tested, the one selected for the final product was the Neural Network model. This model performed similarly to K-Nearest Neighbors, but with slightly higher precision for `Is Delayed` and marginally better overall accuracy. Still, the model needs improvement through better selection of features (removing some, adding others) and tuning of hyperparameters.