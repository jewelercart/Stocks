import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load the dataset from /root/Stock/sample.csv
file_path = '/root/Stock/sample.csv'
df = pd.read_csv(file_path)

# Create the directory if it doesn't exist
model_save_path = '/root/Stock/Models'
os.makedirs(model_save_path, exist_ok=True)

# Function to preprocess data for a specific symbol
def preprocess_data(symbol_df, sequence_length):
    # Select relevant columns
    data = symbol_df[['Date', 'Close']].copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Normalize the Close prices
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Return the scaled data and the scaler for future use
    return scaled_data, scaler

# Function to create a data generator for batch training
def data_generator(data, sequence_length, batch_size):
    while True:
        x_batch, y_batch = [], []
        for _ in range(batch_size):
            idx = np.random.randint(sequence_length, len(data))
            x_batch.append(data[idx-sequence_length:idx, 0])
            y_batch.append(data[idx, 0])
        x_batch = np.array(x_batch)
        y_batch = np.array(y_batch)
        yield np.reshape(x_batch, (x_batch.shape[0], x_batch.shape[1], 1)), y_batch

# Function to create the LSTM model with reduced units
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(30, return_sequences=True, input_shape=input_shape))  # Reduced from 50 to 30 units
    model.add(Dropout(0.2))
    model.add(LSTM(30, return_sequences=False))  # Reduced from 50 to 30 units
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Predicting the next price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 1: Calculate the average Close price for each symbol
avg_close_per_symbol = df.groupby('Symbol')['Close'].mean().sort_values(ascending=False)

# Step 2: Select the top 30 symbols based on highest average Close price
top_30_symbols = avg_close_per_symbol.head(30).index

# Set sequence length and batch size
sequence_length = 30  # Reduced from 60 to 30 days
batch_size = 16  # Reduced batch size to 16
epochs = 20  # Reduced number of epochs to 20

# Step 3: Train and predict for each of the top 30 symbols
for symbol in top_30_symbols:
    symbol_df = df[df['Symbol'] == symbol]
    
    # Preprocess the data
    scaled_data, scaler = preprocess_data(symbol_df, sequence_length)

    # Create training and testing sets using data generator
    train_size = int(len(scaled_data) * 0.8)  # 80% for training
    test_size = len(scaled_data) - train_size
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]  # Include the last part for sequence continuity

    # Create the data generators for training and testing
    train_generator = data_generator(train_data, sequence_length, batch_size)
    test_generator = data_generator(test_data, sequence_length, batch_size)

    # Create and compile the LSTM model
    model = create_lstm_model((sequence_length, 1))

    # Train the model using the data generator
    steps_per_epoch = len(train_data) // batch_size
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

    # Save the model to /root/Stock/Models
    model.save(f'{model_save_path}/lstm_model_{symbol}.h5')
    print(f'Model saved for symbol: {symbol}')

