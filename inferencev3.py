import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load the model, scaler, and columns
model_path = r"C:\Users\ramna\Desktop\Bike Demand Analyzer\models\xgboost_regressor_r2_0_94_v3.pkl"
model = pickle.load(open(model_path, "rb"))

sc_dump_path = r"C:\Users\ramna\Desktop\Bike Demand Analyzer\models\sc.pkl"
scaler = pickle.load(open(sc_dump_path, "rb"))

column_path = r"C:\Users\ramna\Desktop\Bike Demand Analyzer\models\columns.pkl"
columns = pickle.load(open(column_path, "rb"))

def preprocess_input(input_data):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Convert "Date" to "year", "month", and "day"
    input_df['Date'] = pd.to_datetime(input_df['Date'], format="%d-%m-%Y")
    input_df['year'] = input_df['Date'].dt.year
    input_df['month'] = input_df['Date'].dt.month
    input_df['day'] = input_df['Date'].dt.day_name()  # Day name for weekdays_weekend

    # Create the weekdays_weekend column
    input_df['weekdays_weekend'] = input_df['day'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Drop the 'Date', 'day', and 'year' columns as they are not needed for prediction
    input_df = input_df.drop(['Date', 'day', 'year'], axis=1)

    # One-hot encode the categorical features
    categorical_features = ['Hour', 'Seasons', 'Holiday', 'Functioning_Day', 'month', 'weekdays_weekend']
    for col in categorical_features:
        dummies = pd.get_dummies(input_df[col], prefix=col, drop_first=True)
        input_df = pd.concat([input_df, dummies], axis=1)
        input_df = input_df.drop([col], axis=1)

    # Ensure all expected columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df

def make_prediction(input_data):
    # Preprocess the input
    processed_input = preprocess_input(input_data)

    # Scale the input features
    scaled_input = scaler.transform(processed_input)

    # Make the prediction (predicting the square root of the bike count in scaled form)
    predicted_scaled_sqrt_value = model.predict(scaled_input).reshape(-1, 1)

    # Inverse transform the prediction to get the original scale
    predicted_sqrt_value = scaler.inverse_transform(np.concatenate([predicted_scaled_sqrt_value, scaled_input[:, 1:]], axis=1))[:, 0].reshape(-1, 1)

    # Square the prediction to get the original bike count
    predicted_value = (predicted_sqrt_value ** 2).tolist()

    # Return the predicted value
    return round(predicted_value[0][0])

# Define the Gradio interface with enhanced UI
def gradio_interface(hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall, seasons, holiday, functioning_day, date):
    input_data = {
        'Hour': hour,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_speed': wind_speed,
        'Visibility': visibility,
        'Solar_Radiation': solar_radiation,
        'Rainfall': rainfall,
        'Snowfall': snowfall,
        'Seasons': seasons,
        'Holiday': holiday,
        'Functioning_Day': functioning_day,
        'Date': date
    }
    return make_prediction(input_data)

# Create the Gradio interface with additional description and styling
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Slider(minimum=0, maximum=23, step=1, label="Hour"),
        gr.Number(label="Temperature (°C)"),
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Wind Speed (m/s)"),
        gr.Number(label="Visibility (meters)"),
        gr.Number(label="Solar Radiation"),
        gr.Number(label="Rainfall (mm)"),
        gr.Number(label="Snowfall (mm)"),
        gr.Dropdown(choices=['Spring', 'Summer', 'Autumn', 'Winter'], label="Seasons"),
        gr.Dropdown(choices=['Holiday', 'No Holiday'], label="Holiday"),
        gr.Dropdown(choices=['Yes', 'No'], label="Functioning Day"),
        gr.Textbox(label="Date (DD-MM-YYYY)", placeholder="01-12-2017")
    ],
    outputs=gr.Textbox(label="Predicted Rented Bike Count"),
    title="Bike Demand Prediction",
    description="Enter the details below to predict the number of rented bikes. This model estimates bike rental demand based on weather conditions and time of the day.",
    theme="default"  # You can choose different themes or create a custom theme
)

# Launch the interface
iface.launch()
