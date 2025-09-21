from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
try:
    model_path = 'model_new.pkl'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
except Exception as e:
    print("Error loading model:", e)
    model = None

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Model not loaded properly!")

    try:
        # Extract data from form
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]

        # Debugging logs
        print("Received features:", features)
        print("Shape passed to model:", np.array(final_features).shape)

        # Make prediction
        prediction = model.predict(final_features)
        print("Model prediction:", prediction)

        output = 'Eligible for Loan' if prediction[0] == 1 else 'Not Eligible for Loan'

        return render_template('index.html', prediction_text=f'Prediction: {output}')

    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
