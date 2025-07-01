import pandas as pd
import numpy as np
import fastf1
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

MODEL_PATH = "f1_prediction_model.joblib"

def get_race_data_for_training(year, race_name):
    try:

        qualifying_session = fastf1.get_session(year, race_name, 'Q')
        race_session = fastf1.get_session(year, race_name, 'R')
        qualifying_session.load(telemetry=False, weather=True, messages=False)
        race_session.load(telemetry=False, weather=True, messages=False)

        q_laps_all = qualifying_session.laps
        q_laps = q_laps_all.loc[q_laps_all.groupby('Driver')['LapTime'].idxmin()]

        q_results = q_laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
        q_results.rename(columns={
            "LapTime": "QualifyingTime",
            "Sector1Time": "QualiSector1",
            "Sector2Time": "QualiSector2",
            "Sector3Time": "QualiSector3"
        }, inplace=True)

        r_laps = race_session.laps.pick_wo_box()
        median_lap_times = r_laps.groupby("Driver")["LapTime"].median().reset_index()
        median_lap_times.rename(columns={"LapTime": "MedianRaceLap"}, inplace=True)

        training_data = pd.merge(q_results, median_lap_times, on="Driver")

        for col in ["QualifyingTime", "QualiSector1", "QualiSector2", "QualiSector3", "MedianRaceLap"]:
            training_data[f"{col}_s"] = training_data[col].dt.total_seconds()

        weather_data = race_session.weather_data
        training_data["TrackTemp"] = weather_data["TrackTemp"].mean()
        training_data["AirTemp"] = weather_data["AirTemp"].mean()
        training_data["Humidity"] = weather_data["Humidity"].mean()
        training_data["WindSpeed"] = weather_data["WindSpeed"].mean()

        return training_data

    except Exception as e:
        print(f"Could not process data for {year} {race_name}: {e}")
        return pd.DataFrame()

def train_and_save_model():
    print("Starting model training...")
    fastf1.Cache.enable_cache("f1_cache")

    races_to_train = [
        (2023, "Bahrain Grand Prix"), (2023, "Saudi Arabian Grand Prix"),
        (2023, "Australian Grand Prix"), (2023, "Azerbaijan Grand Prix"),
        (2023, "Miami Grand Prix"), (2023, "Monaco Grand Prix"),
        (2023, "Spanish Grand Prix"), (2023, "Canadian Grand Prix"),
        (2023, "Austrian Grand Prix"), (2023, "British Grand Prix"),
        (2024, "Bahrain Grand Prix"), (2024, "Saudi Arabian Grand Prix"),
        (2024, "Australian Grand Prix"), (2024, "Japanese Grand Prix"),
        (2024, "Chinese Grand Prix"), (2024, "Miami Grand Prix")
    ]

    full_training_data = pd.concat(
        [get_race_data_for_training(year, race) for year, race in races_to_train],
        ignore_index=True
    ).dropna()

    if full_training_data.empty:
        print("No training data could be compiled. Aborting.")
        return

    features = [
        "QualifyingTime_s", "QualiSector1_s", "QualiSector2_s", "QualiSector3_s",
        "TrackTemp", "AirTemp", "Humidity", "WindSpeed"
    ]
    target = "MedianRaceLap_s"

    X = full_training_data[features]
    y = full_training_data[target]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training new XGBoost model with early stopping...")
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print(f"Saving new, improved model to {MODEL_PATH}...")
    joblib.dump(model, MODEL_PATH)
    print("Model training complete and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()