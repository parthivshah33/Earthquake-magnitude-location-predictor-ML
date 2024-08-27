import numpy as np

def preprocess_MagPrediction(data):
    
    '''
        Input Order : Latitude(deg), Longitude(deg), Depth(km), No_of_Stations, Gap, Close, RMS
    '''
    data = np.array(data)
    return data


def preprocess_LocationPrediction(data) :
    
    '''
        Input Order : Depth(km), Magnitude(ergs), No_of_Stations, Gap, Close, RMS
    '''
    data = np.array(data)
    return data