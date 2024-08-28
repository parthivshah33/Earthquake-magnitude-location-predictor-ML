from matplotlib.pyplot import margins
import streamlit as st
from preprocess import preprocess_MagPrediction, preprocess_LocationPrediction
from inference import inference_MagPrediction, inference_LocationPrediction


# Set page layout to wide mode
st.set_page_config(layout="wide")

# Sidebar for choosing the option
st.sidebar.write("### Choose an Option")
option = st.sidebar.radio("What to predict?:", ["Magnitude", "Location Cordinates"])

prediction = None

# Display the appropriate form based on the selected option
st.write("Earthquake related Predictions")

if option == "Magnitude":
    st.write("Magnitude Prediction")
    latitude = st.number_input("Latitude", format="%.10f")
    longitude = st.number_input("Longitude", format="%.10f")
    depth = st.number_input("Depth", format="%.10f")
    no_of_stations = st.number_input("No of Stations", format="%.10f")
    gap = st.number_input("Gap", format="%.10f")
    close = st.number_input("Close", format="%.10f")
    rms = st.number_input("RMS", format="%.10f")
    submit_button = st.button("Predict")

    if submit_button:
        # Example prediction logic (replace with actual logic)
        values = [[latitude, longitude, depth, no_of_stations, gap, close, rms]]
        
        data_preprocessed = preprocess_MagPrediction(values)
        print(data_preprocessed)
        prediction = inference_MagPrediction(data_preprocessed, model_path=r'D:\Earthquake-prediction-ML\models\MagPred_random_forest_regressor_200_estimators_minSampLeaf_5_minSampleSplit6_oob_True.pkl')
        
        st.success("Form 1 Submitted Successfully!")

elif option == "Location Cordinates":
    st.write("Location Prediction")
    depth = st.number_input("Depth(km)", format="%.10f")
    magnitude = st.number_input("Magnitude(ergs)", format="%.10f")
    no_of_stations = st.number_input("No of Stations", format="%.10f")
    gap = st.number_input("Gap", format="%.10f")
    close = st.number_input("Close", format="%.10f")
    rms = st.number_input("RMS", format="%.10f")
    submit_button = st.button("Predict")

    if submit_button:
        # Example prediction logic (replace with actual logic)
        
        values = [[depth, magnitude, no_of_stations, gap, close, rms]]
        data_preprocessed = preprocess_LocationPrediction(values)
        print(data_preprocessed)
        prediction = inference_LocationPrediction(data_preprocessed, model_path=r'D:\Earthquake-prediction-ML\models\location_predictor.pkl')

        print("----------------------------- :::::::",prediction)
        st.success("Form 2 Submitted Successfully!")

# Display the answer box if a prediction is available
if prediction is not None:
    st.text_area("Prediction:", value=f"{str(prediction)}", height=50)