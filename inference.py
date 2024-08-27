import joblib
import numpy as np
import pandas as pd

def inference_MagPrediction(data, model_path : str): 
    
    '''
     Input : np.array(),
     return np.array()
    '''
    
    mag_predictor_model = joblib.load(model_path)
    
    predicted_magnitude_value = mag_predictor_model.predict(data)
    
    return predicted_magnitude_value.tolist()[0]


def inference_LocationPrediction(data ,model_path : str) : 
    
    '''
    Input : np.array(), 
    
    return -> two dimentional list of latitude and longitude.
    
    return value example : [[ 31.5524,-121.34242 ]]
        access first element - list[0][0] (latitude)
        access second element - list[0][1] (longitude)
    '''
    
    location_predictor_model = joblib.load(model_path)
    
    predicted_Latitude_Longitude = location_predictor_model.predict(data)
    
    return predicted_Latitude_Longitude.tolist()


# result = inference_MagPrediction(data=np.array([[ 3.866370e+01,-1.196733e+02,8.000000e-02,2.400000e+01,1.960000e+02,7.700000e+01,1.800000e-01]]),model_path=r'D:\Earthquake-prediction-using-Machine-learning-models-main\models\MagPred_random_forest_regressor_200_estimators_minSampLeaf_5_minSampleSplit6_oob_True.pkl')

# print(result)