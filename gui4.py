import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox  # Corrected import
from tkinter import ttk  # For modern widgets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Function to forecast future values
def forecast_future(data, model, scaler, future_steps):
    input_data = data[-time_step:].reshape(1, time_step, 1)
    future_predictions = []

    for _ in range(future_steps):
        next_value = model.predict(input_data, verbose=0)
        future_predictions.append(next_value[0, 0])
        next_value_reshaped = next_value.reshape(1, 1, 1)
        input_data = np.append(input_data[:, 1:, :], next_value_reshaped, axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# GUI Functions
def load_data():
    global data, scaled_load, scaled_price, scaler_load, scaler_price, time_step

    load_file = filedialog.askopenfilename(title="Select Load Data CSV", filetypes=[("CSV Files", "*.csv")])
    price_file = filedialog.askopenfilename(title="Select Price Data CSV", filetypes=[("CSV Files", "*.csv")])

    load_data = pd.read_csv(load_file, parse_dates=['Date'], index_col='Date')
    price_data = pd.read_csv(price_file, parse_dates=['Date'], index_col='Date')

    data = pd.merge(load_data, price_data, left_index=True, right_index=True)

    scaler_load = MinMaxScaler(feature_range=(0, 1))
    scaler_price = MinMaxScaler(feature_range=(0, 1))

    scaled_load = scaler_load.fit_transform(data['Load'].values.reshape(-1, 1))
    scaled_price = scaler_price.fit_transform(data['Price'].values.reshape(-1, 1))

    time_step = 24  # Look-back window for LSTM

    messagebox.showinfo("Data Loaded", "Load and Price data have been successfully loaded.")

def predict_and_plot():
    global model_load, model_price

    try:
        model_load = load_model('load_forecasting_model.keras')
        model_price = load_model('price_forecasting_model.keras')
    except Exception as e:
        messagebox.showerror("Error", f"Error loading models: {e}")
        return

    future_steps = int(future_steps_entry.get())

    future_load = forecast_future(scaled_load, model_load, scaler_load, future_steps)
    future_price = forecast_future(scaled_price, model_price, scaler_price, future_steps)

    actual_load = scaler_load.inverse_transform(scaled_load[-100:])
    actual_price = scaler_price.inverse_transform(scaled_price[-100:])

    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, 101), actual_load, label="Actual Load (Last 100)", color="blue")
    plt.plot(np.arange(101, 101 + future_steps), future_load, label="Forecasted Load", color="red")
    plt.title("Electricity Load Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(1, 101), actual_price, label="Actual Price (Last 100)", color="blue")
    plt.plot(np.arange(101, 101 + future_steps), future_price, label="Forecasted Price", color="red")
    plt.title("Electricity Price Forecasting")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()

    plt.tight_layout()
    plt.show()

# GUI Setup
app = tk.Tk()
app.title("Electricity Load and Price Forecasting")
app.geometry("800x600")  # Set larger window size
app.configure(bg="#f0f8ff")  # Light blue background

# Title Label
title_label = tk.Label(
    app,
    text="Electricity Load and Price Forecasting",
    font=("Arial", 24, "bold"),
    bg="#f0f8ff",
    fg="#4682b4"
)
title_label.pack(pady=20)

# Buttons and Inputs
style = ttk.Style()
style.configure("TButton", font=("Arial", 14), padding=10)

load_data_button = ttk.Button(app, text="Load Data", command=load_data)
load_data_button.pack(pady=20)

future_steps_label = tk.Label(
    app,
    text="Enter Number of Future Steps to Forecast:",
    font=("Arial", 14),
    bg="#f0f8ff",
    fg="#333333"
)
future_steps_label.pack(pady=10)

future_steps_entry = ttk.Entry(app, font=("Arial", 14), width=10)
future_steps_entry.pack(pady=10)

predict_button = ttk.Button(app, text="Predict and Plot", command=predict_and_plot)
predict_button.pack(pady=30)

# Run the GUI
app.mainloop()

