import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# Configure TensorFlow memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("TensorFlow configured to allow memory growth on GPU.")
    except Exception as e:
        print(f"Error configuring TensorFlow memory growth: {e}")
else:
    print("No GPU found. TensorFlow will use CPU.")

# Load the dataset from /root/Stock/sample.csv
file_path = '/root/Stock/sample.csv'
df = pd.read_csv(file_path)

# Create the directory if it doesn't exist
model_save_path = '/root/Stock/Models'
os.makedirs(model_save_path, exist_ok=True)
print(f"Model save path: {model_save_path}")

# Function to preprocess data for a specific symbol
def preprocess_data(symbol_df, sequence_length=60):
    data = symbol_df[['Date', 'Close']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Normalize the Close prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Create sequences for LSTM
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)

    # Reshape for LSTM [samples, time steps, features]
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x, y, scaler

# Function to create the LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=input_shape))  # Reduced units
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False))  # Reduced units
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting the next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train and predict for each symbol
unique_symbols = df['Symbol'].unique()
print(f"Unique symbols found: {unique_symbols}")

for symbol in unique_symbols:
    print(f"\nProcessing symbol: {symbol}")
    symbol_df = df[df['Symbol'] == symbol].sort_values('Date')  # Ensure data is sorted by date
    
    # Preprocess the data
    x, y, scaler = preprocess_data(symbol_df)
    print(f"Data preprocessed for symbol: {symbol}. Shape of x: {x.shape}, y: {y.shape}")

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Data split into training and testing sets for symbol: {symbol}")

    # Create the LSTM model
    model = create_lstm_model((x_train.shape[1], 1))
    print(f"LSTM model created for symbol: {symbol}")

    # Define callbacks
    checkpoint_filepath = f'{model_save_path}/lstm_model_{symbol}.keras'  # Change .h5 to .keras
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_best_only=True,
        monitor='loss',
        mode='min',
        verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Train the model with reduced batch size and added callbacks
    try:
        history = model.fit(
            x_train, y_train,
            epochs=50,
            batch_size=16,  # Reduced batch size
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        print(f"Training completed for symbol: {symbol}. Model saved at {checkpoint_filepath}")
    except Exception as e:
        print(f"Failed to train model for symbol: {symbol}. Error: {e}")

# Final message
print("\nAll symbols have been processed.")

