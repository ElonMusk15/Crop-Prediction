
Built by https://www.blackbox.ai

---

```markdown
# Crop Prediction App

## Project Overview
The Crop Prediction App is a simple web application designed to predict which crop is most suitable for a specific location based on various environmental parameters such as temperature, humidity, and rainfall. The app utilizes a basic scoring algorithm to evaluate crop suitability, providing farmers and agricultural enthusiasts with valuable insights to optimize crop selection for their local conditions.

## Installation
To set up the Crop Prediction App on your local machine, follow the steps below:

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd crop-prediction-app
   ```

2. **Install the required dependencies:**
   You will need to have Python and pip installed. Then, install Flask if you haven't already:
   ```bash
   pip install Flask
   ```

3. **Run the application:**
   Start the Flask server by running:
   ```bash
   python app.py
   ```
   The application will be available at `http://127.0.0.1:8000/`.

## Usage
1. Open your web browser and navigate to `http://127.0.0.1:8000/`.
2. Fill out the prediction form on the `/predict` page with the required parameters:
   - Temperature (in Celsius)
   - Humidity (in percentage)
   - Rainfall (in mm)
   - Nitrogen, Phosphorus, and Potassium content ratios
3. Submit the form to view the predicted crop along with its ideal growing conditions.

## Features
- **Crop Prediction:** Based on user-provided environmental conditions.
- **Crop Requirement Lookup:** View the ideal growing conditions for the predicted crop.
- **User-friendly Interface:** Simple HTML forms for data entry and result display.
- **Error Handling:** Informative error messages for invalid inputs.

## Dependencies
The application relies on the following Python package:
- Flask: A micro web framework for Python.

Here’s the `requirements.txt` for your convenience:
```
Flask==2.0.1
```

## Project Structure
```
crop-prediction-app/
│
├── app.py                # Main application file that runs the Flask server
├── model.py              # Contains the CropPredictor class with crop prediction logic
├── templates/            # Directory for HTML template files
│   ├── index.html        # Home page template
│   ├── predict.html      # Prediction input form template
│   └── result.html       # Result display page template
└── requirements.txt       # List of Python package dependencies
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to contribute to this project by submitting issues or pull requests!
```