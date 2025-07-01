
import streamlit as st
import pandas as pd
import joblib
import requests
import fastf1
import plotly.express as px
import numpy as np

st.set_page_config(
    layout="wide",
    page_title="F1 Advanced Prediction Dashboard",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        color: #EAEAEA;
    }
    .stApp {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "f1_prediction_model.joblib"

@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {MODEL_PATH}. Please run the training script first.")
        return None

@st.cache_data
def get_qualifying_data(year, race_name):
    try:
        session = fastf1.get_session(year, race_name, 'Q')
        session.load(telemetry=False, weather=False, messages=False)
        
        laps = session.laps
        fastest_laps_per_driver = laps.loc[laps.groupby('Driver')['LapTime'].idxmin()]

        if fastest_laps_per_driver.empty:
            st.warning(f"No fastest lap data found for {year} {race_name}.")
            return pd.DataFrame()

        qualifying_data = fastest_laps_per_driver[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()

        for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
            qualifying_data[f"{col} (s)"] = pd.to_timedelta(qualifying_data[col]).dt.total_seconds()

        qualifying_data.rename(columns={
            "LapTime (s)": "QualifyingTime_s",
            "Sector1Time (s)": "QualiSector1_s",
            "Sector2Time (s)": "QualiSector2_s",
            "Sector3Time (s)": "QualiSector3_s"
        }, inplace=True)

        return qualifying_data[["Driver", "QualifyingTime_s", "QualiSector1_s", "QualiSector2_s", "QualiSector3_s"]]

    except Exception as e:
        st.error(f"Error loading data for {year} {race_name}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def get_weather_data(location="Shanghai"):
    params = {"key": "", "q": location} # Replace with your WeatherAPI key under "key"
    try:
        response = requests.get("http://api.weatherapi.com/v1/current.json", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None

def generate_hypothetical_2025_data(year, race_name):
    drivers_2025 = [
        "Max Verstappen", "Charles Leclerc", "Lando Norris", "George Russell", "Oscar Piastri",
        "Lewis Hamilton", "Carlos Sainz Jr.", "Sergio Pérez", "Fernando Alonso", "Esteban Ocon",
        "Pierre Gasly", "Yuki Tsunoda", "Alexander Albon", "Logan Sargeant", "Nico Hülkenberg",
        "Kevin Magnussen", "Valtteri Bottas", "Guanyu Zhou", "Daniel Ricciardo", "Liam Lawson"
    ]

    historical_data = get_qualifying_data(year, race_name)

    if not historical_data.empty:
        mean_q_time = historical_data["QualifyingTime_s"].mean()
        std_q_time = historical_data["QualifyingTime_s"].std()
        mean_s1 = historical_data["QualiSector1_s"].mean()
        std_s1 = historical_data["QualiSector1_s"].std()
        mean_s2 = historical_data["QualiSector2_s"].mean()
        std_s2 = historical_data["QualiSector2_s"].std()
        mean_s3 = historical_data["QualiSector3_s"].mean()
        std_s3 = historical_data["QualiSector3_s"].std()

        performance_order = {
            "Max Verstappen": 1, "Charles Leclerc": 2, "Lando Norris": 3, "George Russell": 4, "Oscar Piastri": 5,
            "Lewis Hamilton": 6, "Carlos Sainz Jr.": 7, "Sergio Pérez": 8, "Fernando Alonso": 9, "Esteban Ocon": 10,
            "Pierre Gasly": 11, "Yuki Tsunoda": 12, "Alexander Albon": 13, "Logan Sargeant": 14, "Nico Hülkenberg": 15,
            "Kevin Magnussen": 16, "Valtteri Bottas": 17, "Guanyu Zhou": 18, "Daniel Ricciardo": 19, "Liam Lawson": 20
        }
        drivers_2025.sort(key=lambda x: performance_order.get(x, 99))

        qualifying_times = []
        sector_1_times = []
        sector_2_times = []
        sector_3_times = []

        np.random.seed(42) 

        for i, driver in enumerate(drivers_2025):

            bias_factor = (i - len(drivers_2025) / 2) * 0.005

            q_time = np.random.normal(loc=mean_q_time + bias_factor * std_q_time, scale=std_q_time * 0.5)
            s1_time = np.random.normal(loc=mean_s1 + bias_factor * std_s1, scale=std_s1 * 0.5)
            s2_time = np.random.normal(loc=mean_s2 + bias_factor * std_s2, scale=std_s2 * 0.5)
            s3_time = np.random.normal(loc=mean_s3 + bias_factor * std_s3, scale=std_s3 * 0.5)

            q_time = max(q_time, 60)
            s1_time = max(s1_time, 15)
            s2_time = max(s2_time, 20)
            s3_time = max(s3_time, 10)

            qualifying_times.append(q_time)
            sector_1_times.append(s1_time)
            sector_2_times.append(s2_time)
            sector_3_times.append(s3_time)

    else:
        st.warning(f"No historical data for {year} {race_name}. Generating generic hypothetical data.")
        np.random.seed(42)
        base_q_time = 88.0
        qualifying_times = base_q_time + np.random.uniform(-0.5, 2.0, len(drivers_2025))
        sector_1_times = qualifying_times / 3.1 + np.random.uniform(-0.1, 0.2, len(drivers_2025))
        sector_2_times = qualifying_times / 2.9 + np.random.uniform(-0.1, 0.2, len(drivers_2025))
        sector_3_times = qualifying_times / 3.0 + np.random.uniform(-0.1, 0.2, len(drivers_2025))

    hypothetical_data = pd.DataFrame({
        "Driver": drivers_2025,
        "QualifyingTime_s": qualifying_times,
        "QualiSector1_s": sector_1_times,
        "QualiSector2_s": sector_2_times,
        "QualiSector3_s": sector_3_times
    })
    return hypothetical_data

def make_predictions(model, features_df, weather_inputs):
    if model is None or features_df.empty:
        return None

    prediction_input = features_df.copy()
    for key, value in weather_inputs.items():
        prediction_input[key] = value

    feature_cols = ["QualifyingTime_s", "QualiSector1_s", "QualiSector2_s", "QualiSector3_s", "TrackTemp", "AirTemp", "Humidity", "WindSpeed"]
    X_pred = prediction_input[feature_cols].fillna(0)

    predictions = model.predict(X_pred)
    features_df['PredictedRaceTime (s)'] = predictions
    return features_df.sort_values(by="PredictedRaceTime (s)", ascending=True)

def main():
    st.title("🏎️ F1 Advanced Prediction Dashboard")
    st.caption("Interactive predictions powered by XGBoost, FastF1, and WeatherAPI")

    model = load_model()
    if model is None:
        return

    st.sidebar.header("🔮 Prediction Mode")
    prediction_mode = st.sidebar.radio("Choose a mode:", ["Live Race Data", "Hypothetical 2025 Race"])

    if prediction_mode == "Live Race Data":
        st.sidebar.header("🏁 Race Selection")
        year = st.sidebar.selectbox("Year", [2024, 2023], index=0)
        race_name = st.sidebar.selectbox("Race", ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Chinese Grand Prix", "Miami Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix", "Abu Dhabi Grand Prix"], index=0)
        qualifying_data = get_qualifying_data(year, race_name)
        if qualifying_data.empty:
            st.warning("Could not load qualifying data. Please select another race.")
            return
    else: 
        st.sidebar.header("🏎️ Hypothetical 2025 Race")
        hypothetical_year = st.sidebar.selectbox("Base Year for Track Data", [2024, 2023], index=0)
        hypothetical_race_name = st.sidebar.selectbox("Choose a Track", ["Bahrain Grand Prix", "Saudi Arabian Grand Prix", "Australian Grand Prix", "Chinese Grand Prix", "Miami Grand Prix", "Monaco Grand Prix", "Spanish Grand Prix", "Canadian Grand Prix", "Austrian Grand Prix", "British Grand Prix", "Hungarian Grand Prix", "Belgian Grand Prix", "Dutch Grand Prix", "Italian Grand Prix", "Singapore Grand Prix", "Japanese Grand Prix", "Qatar Grand Prix", "United States Grand Prix", "Mexico City Grand Prix", "Brazilian Grand Prix", "Las Vegas Grand Prix", "Abu Dhabi Grand Prix"], index=0)
        qualifying_data = generate_hypothetical_2025_data(hypothetical_year, hypothetical_race_name)

    st.sidebar.header("🌦️ Weather & Track Conditions")
    if prediction_mode == "Live Race Data":
        weather_location = st.sidebar.text_input("Weather Location", race_name.replace(" Grand Prix", ""))
    else:
        weather_location = st.sidebar.text_input("Weather Location", "Shanghai") 
    weather_data = get_weather_data(weather_location)

    default_air_temp, default_humidity, default_wind_speed = 22.0, 60.0, 10.0
    if weather_data:
        st.sidebar.write(f"**Live Weather in {weather_data['location']['name']}**")
        st.sidebar.write(f"🌡️ {weather_data['current']['temp_c']}°C | 💧 {weather_data['current']['humidity']}% | 🌬️ {weather_data['current']['wind_kph']} kph")
        default_air_temp = weather_data['current']['temp_c']
        default_humidity = weather_data['current']['humidity']
        default_wind_speed = weather_data['current']['wind_kph']

    track_temp = st.sidebar.slider("Track Temperature (°C)", 0.0, 60.0, default_air_temp + 5, 0.5)
    air_temp = st.sidebar.slider("Air Temperature (°C)", -10.0, 50.0, default_air_temp, 0.5)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, float(default_humidity), 1.0)
    wind_speed = st.sidebar.slider("Wind Speed (kph)", 0.0, 40.0, default_wind_speed, 0.5)

    weather_inputs = {"TrackTemp": track_temp, "AirTemp": air_temp, "Humidity": humidity, "WindSpeed": wind_speed}
    predictions_df = make_predictions(model, qualifying_data.copy(), weather_inputs)

    if predictions_df is not None and not predictions_df.empty:
        st.header("🏆 Predicted Podium")
        podium_df = predictions_df.rename(columns={"PredictedRaceTime (s)": "PredictedRaceTime"})
        top_3 = podium_df.head(3)
        cols = st.columns(3)
        for i, driver in enumerate(top_3.itertuples()):
            with cols[i]:
                st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #444; border-radius: 8px; background-color: #272727;'>"
                            f"<h3>{["🥇", "🥈", "🥉"][i]}</h3>"
                            f"<h2>{driver.Driver}</h2>"
                            f"<p style='font-size: 1.2em;'><b>{driver.PredictedRaceTime:.3f}s</b></p>"
                            f"</div>", unsafe_allow_html=True)

        st.header("📊 Detailed Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted vs. Qualifying Times")
            fig = px.scatter(predictions_df, x="QualifyingTime_s", y="PredictedRaceTime (s)", text="Driver", title="Pace Analysis", color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Time Delta to Leader")
            leader_time = predictions_df["PredictedRaceTime (s)"].min()
            predictions_df["Delta to Leader (s)"] = predictions_df["PredictedRaceTime (s)"] - leader_time
            fig2 = px.bar(predictions_df.sort_values("Delta to Leader (s)"), x="Driver", y="Delta to Leader (s)", title="Gap to Predicted Winner", color="Delta to Leader (s)", color_continuous_scale="reds")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Model Feature Importance")
        importance_data = pd.DataFrame({'feature': model.feature_names_in_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        fig3 = px.bar(importance_data, x='importance', y='feature', orientation='h', title="What Matters to the Model?")
        st.plotly_chart(fig3, use_container_width=True)

    else:
        st.warning("No predictions available. Check model and inputs.")

if __name__ == "__main__":
    main()
