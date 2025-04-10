from flask import Flask, render_template, request, jsonify
from model import CropPredictor
import logging

app = Flask(__name__)
predictor = CropPredictor()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handle the prediction form and results"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        # Get form data
        data = {
            'location': request.form.get('location', ''),
            'temperature': float(request.form.get('temperature', 0)),
            'humidity': float(request.form.get('humidity', 0)),
            'rainfall': float(request.form.get('rainfall', 0)),
            'nitrogen': float(request.form.get('nitrogen', 0)),
            'phosphorus': float(request.form.get('phosphorus', 0)),
            'potassium': float(request.form.get('potassium', 0))
        }
        
        # Log the received data
        logger.info(f"Received prediction request with data: {data}")
        
        # Make prediction
        predicted_crop = predictor.predict_crop(
            temperature=data['temperature'],
            humidity=data['humidity'],
            rainfall=data['rainfall'],
            n=data['nitrogen'],
            p=data['phosphorus'],
            k=data['potassium']
        )
        
        # Get crop requirements for comparison
        crop_requirements = predictor.get_crop_requirements(predicted_crop)
        
        return render_template('result.html', 
                             prediction=predicted_crop,
                             requirements=crop_requirements,
                             data=data)
                             
    except ValueError as e:
        logger.error(f"Invalid input data: {str(e)}")
        return render_template('predict.html', 
                             error="Please check your input values and try again.")
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return render_template('predict.html', 
                             error="An error occurred. Please try again later.")

if __name__ == '__main__':
    app.run(debug=True, port=8000)
