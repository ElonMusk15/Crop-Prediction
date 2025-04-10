class CropPredictor:
    def __init__(self):
        # Define crop requirements (simplified for demonstration)
        self.crop_requirements = {
            'rice': {'temp': (20, 35), 'humidity': (60, 90), 'rainfall': (150, 300)},
            'wheat': {'temp': (15, 25), 'humidity': (50, 70), 'rainfall': (75, 150)},
            'corn': {'temp': (20, 30), 'humidity': (50, 80), 'rainfall': (100, 200)},
            'cotton': {'temp': (25, 35), 'humidity': (40, 60), 'rainfall': (60, 150)},
            'sugarcane': {'temp': (25, 35), 'humidity': (70, 90), 'rainfall': (150, 300)}
        }

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

            # Simple scoring system for each crop
            scores = {}
            
            for crop, requirements in self.crop_requirements.items():
                temp_range = requirements['temp']
                humidity_range = requirements['humidity']
                rainfall_range = requirements['rainfall']
                
                # Calculate score based on how well conditions match requirements
                temp_score = 1 if temp_range[0] <= temperature <= temp_range[1] else 0
                humidity_score = 1 if humidity_range[0] <= humidity <= humidity_range[1] else 0
                rainfall_score = 1 if rainfall_range[0] <= rainfall <= rainfall_range[1] else 0
                
                # Additional score based on NPK values (simplified)
                npk_score = (n + p + k) / 300  # Normalize NPK sum
                
                # Calculate total score
                total_score = (temp_score + humidity_score + rainfall_score) * 0.8 + npk_score * 0.2
                scores[crop] = total_score

            # Return the crop with highest score
            best_crop = max(scores.items(), key=lambda x: x[1])[0]
            return best_crop

        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

    def get_crop_requirements(self, crop_name):
        """
        Get the ideal requirements for a specific crop.
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Dictionary containing ideal requirements for the crop
        """
        return self.crop_requirements.get(crop_name, None)
