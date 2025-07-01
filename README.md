# Formula-1-Prediction-Model

<h1 align="center" style="font-weight: bold;">Formula 1 Prediction Model 🏎️</h1>


<p align="center">F1 Prediction Dashboard is an interactive web app that predicts Formula 1 race outcomes using machine learning, real qualifying data, and live weather conditions.

## Features
- Race Time Prediction: Uses an XGBoost regression model trained on historical qualifying and race data to predict median race lap times for each driver.

- Live & Hypothetical Modes: Analyze real races (2023–2024) or generate hypothetical scenarios for the 2025 season with customizable weather and track conditions.

- Weather Integration: Fetches live weather data via WeatherAPI to enhance prediction accuracy.

- Data Visualization: Presents results and podium predictions with interactive Plotly charts.

- User-Friendly Interface: Built with Streamlit for an intuitive, responsive dashboard experience.

## How It Works
The backend fetches and processes qualifying and race data using FastF1.

The model is trained and saved with train_and_save_model.py using features like sector times and weather.

The Streamlit app (streamlit_app.py) loads the model, collects user inputs, and displays predictions.

</p>



<h2 id="technologies">💻 Technologies</h2>

Here is a list of the main technologies used in your F1 Advanced Prediction Dashboard project:

- Streamlit – for building the interactive web dashboard.

- Pandas – for data manipulation and analysis.

- NumPy – for numerical operations and random data -generation.

- Joblib – for saving and loading the trained machine learning model.

- Requests – for making HTTP requests to fetch live weather data.

- FastF1 – for accessing and processing Formula 1 session data (qualifying, race, weather).

- Plotly – for interactive data visualization within the dashboard.

- XGBoost – as the core machine learning algorithm for race time prediction.

- Scikit-learn – for model evaluation and data splitting (train/test split).

- WeatherAPI – (via HTTP requests) for integrating live weather data into predictions

<h2 id="started">🚀 Getting started</h2>

### Step 1 
Clone the repository:
```bash
git clone https://github.com/aryannverse/Formula-1-Prediction-Model.git
```

### Step 2
Install all the libraries mentioned in the 'Requirements.txt'
```bash
pip install -r Requirements.txt
```

### Step 3
In the 'Streamlit_app.py' file, insert your WeatherAPI key in line 76
```python
params = {"key": "", "q": location} #insert your key here from http://api.weatherapi.com
```

## Running the app:
Enter this line in the terminal to launch the dashboard in your browser:
```bash
streamlit run 'streamlit_app.py'
```
