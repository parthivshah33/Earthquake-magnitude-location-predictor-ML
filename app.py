from flask import Flask, render_template, request, redirect, url_for
from preprocess import preprocess_MagPrediction, preprocess_LocationPrediction
from inference import inference_MagPrediction, inference_LocationPrediction

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select', methods=['POST'])
def select():
    selection = request.form.get('selection')
    if selection == 'Magnitude':
        return redirect(url_for('magnitude'))
    elif selection == 'Location':
        return redirect(url_for('location'))
    # Add other conditions for "Future EQ" if needed
    return 'Not implemented yet'


@app.route('/magnitude', methods=['GET', 'POST'])
def magnitude():
    if request.method == 'POST':
        try:
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            depth = float(request.form['depth'])
            no_of_stations = float(request.form['no_of_stations'])
            gap = float(request.form['gap'])
            close = float(request.form['close'])
            rms = float(request.form['rms'])
        except ValueError:
            return render_template('magnitude.html', error="All inputs must be valid floats.")

        data_magnitude = [[latitude, longitude, depth, no_of_stations, gap, close, rms]]
        data_preprocessed = preprocess_MagPrediction(data_magnitude)
        data_inference = inference_MagPrediction(data_preprocessed, model_path=r'D:\Earthquake-prediction-ML\models\MagPred_random_forest_regressor_200_estimators_minSampLeaf_5_minSampleSplit6_oob_True.pkl')

        return render_template('magnitude.html', latitude=latitude, longitude=longitude, depth=depth,
                               no_of_stations=no_of_stations, gap=gap, close=close, rms=rms,
                               prediction=data_inference, prediction_type="Magnitude")
    return render_template('magnitude.html')


@app.route('/location', methods=['GET', 'POST'])
def location():
    if request.method == 'POST':
        try:
            depth = float(request.form['depth'])
            magnitude = float(request.form['magnitude'])
            no_of_stations = float(request.form['no_of_stations'])
            gap = float(request.form['gap'])
            close = float(request.form['close'])
            rms = float(request.form['rms'])
        except ValueError:
            return "All inputs must be valid floats."

        data_location = [[depth, magnitude, no_of_stations, gap, close, rms]]
        data_preprocessed = preprocess_LocationPrediction(data_location)
        print(data_preprocessed)
        data_inference = inference_LocationPrediction(data_preprocessed, model_path=r'D:\Earthquake-prediction-ML\models\location_predictor.pkl')

        return render_template('location.html', depth=depth, magnitude=magnitude,
                               no_of_stations=no_of_stations, gap=gap, close=close, rms=rms,
                               prediction=data_inference, prediction_type="Location")
    return render_template('location.html')

@app.route('/future_eq', methods=['GET', 'POST'])
def future_eq():
    return render_template("future_eq.html")

# if __name__ == '__main__':
#     app.run(debug=True)
