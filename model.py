import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

class CropPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.setup_logging()
        self.train_model()

    def setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_model(self):
        """Train the machine learning model using the crop dataset"""
        try:
            # Load and prepare the dataset
            data = pd.read_csv('data/crop_data.csv')
            
            # Separate features and target
            X = data[['N', 'P', 'K', 'temperature', 'humidity', 'rainfall']]
            y = data['label']

            # Initialize label encoder
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)

            # Initialize scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )

            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.model.fit(X_train, y_train)

            # Calculate and log accuracy
            train_accuracy = self.model.score(X_train, y_train)
            test_accuracy = self.model.score(X_test, y_test)
            
            self.logger.info(f"Model trained successfully!")
            self.logger.info(f"Training accuracy: {train_accuracy:.2f}")
            self.logger.info(f"Testing accuracy: {test_accuracy:.2f}")

            # Save the model and scaler
            joblib.dump(self.model, 'data/crop_model.joblib')
            joblib.dump(self.scaler, 'data/scaler.joblib')
            joblib.dump(self.label_encoder, 'data/label_encoder.joblib')

        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            self.model = joblib.load('data/crop_model.joblib')
            self.scaler = joblib.load('data/scaler.joblib')
            self.label_encoder = joblib.load('data/label_encoder.joblib')
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.train_model()

    def predict_crop(self, temperature, humidity, rainfall, n, p, k):
        """
        Predict the most suitable crop based on given parameters.
        
        Args:
            temperature (float): Temperature in Celsius
            humidity (float): Humidity percentage
            rainfall (float): Annual rainfall in mm
            n (float): Nitrogen content ratio
            p (float): Phosphorus content ratio
            k (float): Potassium content ratio
            
        Returns:
            str: Name of the most suitable crop
        """
        try:
            # Validate input ranges
            if not (0 <= temperature <= 50 and 
                   0 <= humidity <= 100 and 
                   0 <= rainfall <= 500 and
                   0 <= n <= 100 and 
                   0 <= p <= 100 and 
                   0 <= k <= 100):
                raise ValueError("Input parameters out of valid range")

            # Prepare input data
            input_data = np.array([[n, p, k, temperature, humidity, rainfall]])
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Make prediction
            prediction_encoded = self.model.predict(input_scaled)
            
            # Decode prediction
            predicted_crop = self.label_encoder.inverse_transform(prediction_encoded)[0]
            
            return predicted_crop

        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def get_crop_requirements(self, crop_name):
        """
        Get the ideal requirements for a specific crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Dictionary containing ideal requirements for the crop
        """
        try:
            # Load the dataset
            data = pd.read_csv('data/crop_data.csv')
            
            # Filter data for the specific crop
            crop_data = data[data['label'] == crop_name]
            
            if crop_data.empty:
                return None
            
            # Calculate average requirements
            requirements = {
                'temp': (
                    float(crop_data['temperature'].mean() - 2),
                    float(crop_data['temperature'].mean() + 2)
                ),
                'humidity': (
                    float(crop_data['humidity'].mean() - 5),
                    float(crop_data['humidity'].mean() + 5)
                ),
                'rainfall': (
                    float(crop_data['rainfall'].mean() - 20),
                    float(crop_data['rainfall'].mean() + 20)
                )
            }
            
            return requirements

        except Exception as e:
            self.logger.error(f"Error getting crop requirements: {str(e)}")
            return None

if __name__ == "__main__":
    # Test the model
    predictor = CropPredictor()
    test_prediction = predictor.predict_crop(
        temperature=25.0,
        humidity=65.0,
        rainfall=150.0,
        n=45,
        p=50,
        k=55
    )
    print(f"Test prediction: {test_prediction}")
