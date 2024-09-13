import os
import pandas as pd
import yfinance as yf
import mysql.connector  # Import the MySQL connector for MariaDB
from datetime import datetime

# Load the CSV data
stock = pd.read_csv("/root/Stock/stock_details_5_years.csv")

# Get the unique company symbols
stock_list = stock['Company'].unique()

# Initialize an empty list to store the stock data
stock_data_list = []

# Define the date range
start_date = "2018-12-31"
end_date = datetime.today()

# Loop through each symbol and download the data
for symbol in stock_list:
    try:
        # Download stock data for the current symbol
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        
        # Add a column for the stock symbol
        stock_data['Symbol'] = symbol
        
        # Reset the index to get 'Date' as a column
        stock_data.reset_index(inplace=True)
        
        # Append the DataFrame to the list
        stock_data_list.append(stock_data)
    except Exception as e:
        print(f"Error downloading data for {symbol}: {e}")

# Combine all the DataFrames into one
combined_df = pd.concat(stock_data_list, ignore_index=True)

# Keep the original column names for the DataFrame
common_colnames = ["Date", "Open", "High", "Low", "Close", "Volume", "Adj Close", "Symbol"]
combined_df.columns = common_colnames

# Rename 'Adj Close' to 'AdjClose' only for insertion into MariaDB
df_for_sql = combined_df.rename(columns={"Adj Close": "AdjClose"})

# Save the combined data to a CSV file
df_for_sql.to_csv("/root/Stock/sample.csv", index=False)

# Connect to the MariaDB server (without specifying a database initially)
conn = mysql.connector.connect(
    host="localhost",  # Your MariaDB server hostname
    user="remote_user",       # Your MariaDB username
    password="stocks123",        # Your MariaDB password (empty in this case)
    database="",
    port = 3307
)

# Create a cursor object to interact with the server
cursor = conn.cursor()

# Create the database if it doesn't exist
cursor.execute("CREATE DATABASE IF NOT EXISTS Stocks;")
print("Database 'Stocks' checked or created.")

# Select the database to use
cursor.execute("USE Stocks;")

# Check if the table 'Stocks' exists
cursor.execute("""
    SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'Stocks' AND table_name = 'Stocks';
""")
table_exists = cursor.fetchone()[0]

# If the table does not exist, create it with a primary key
if not table_exists:
    cursor.execute("""
        CREATE TABLE Stocks (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Date VARCHAR(255),
            Open FLOAT,
            High FLOAT,
            Low FLOAT,
            Close FLOAT,
            Volume INT,
            AdjClose FLOAT,
            Symbol VARCHAR(255)
        );
    """)
    print("Table 'Stocks' created.")
else:
    # Empty the table if it exists
    cursor.execute("TRUNCATE TABLE Stocks;")
    print("Table 'Stocks' cleared.")

# Define the chunk size for insertion
chunk_size = 1000  # Adjust the chunk size based on your needs

# Insert the data into the MariaDB table in chunks
for start in range(0, len(df_for_sql), chunk_size):
    end = start + chunk_size
    chunk = df_for_sql.iloc[start:end]
    data_tuples = [tuple(row) for row in chunk.values]
    cursor.executemany("""
        INSERT INTO Stocks (Date, Open, High, Low, Close, Volume, AdjClose, Symbol)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
    """, data_tuples)

print("Data inserted into 'Stocks' table.")

# Commit the changes and close the connection
conn.commit()
conn.close()

# Print the first few rows of the combined DataFrame
print(combined_df.head())

# Optionally, display the current working directory
print(os.getcwd())

