import mysql.connector
import pandas as pd

def fetch_irrigation_data():
    """
    Connects to the MySQL database, retrieves data from the 'irrigation_data' table, 
    and returns it as a Pandas DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame containing irrigation data.
    """
    # Database connection details
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "root",
        "database": "agriculture_db",
        "port": 3306
    }

    try:
        # Connect to MySQL
        connection = mysql.connector.connect(**db_config)
        
        if connection.is_connected():
            print("✅ Connected to MySQL!")

            # Query the table
            query = "SELECT * FROM irrigation_data;"
            
            # Load data into Pandas DataFrame
            df = pd.read_sql(query, connection)
            
            print("📊 Data Loaded Successfully!")
            print(df.head())  # Display first few rows
            
            return df

    except mysql.connector.Error as e:
        print(f"❌ Error: {e}")
        return None

    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("🔌 Connection Closed.")
