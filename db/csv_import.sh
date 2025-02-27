#!/bin/bash

# Set variables
CONTAINER_NAME="mysql-container"
CSV_FILE=$(find . -maxdepth 1 -name "*.csv" | head -n 1)
DEST_PATH="/var/lib/mysql-files/"

# Check if CSV file exists
if [[ -z "$CSV_FILE" ]]; then
    echo "❌ No CSV file found in the current directory."
    exit 1
fi

echo "✅ Found CSV file: $CSV_FILE"

# Ensure MySQL is running
if ! docker ps --format '{{.Names}}' | grep -q "$CONTAINER_NAME"; then
    echo "❌ MySQL container is not running!"
    exit 1
fi

# Create database and table if they don't exist
echo "📂 Checking if database and table exist..."
docker exec -i "$CONTAINER_NAME" mysql -u root -proot -e "
CREATE DATABASE IF NOT EXISTS agriculture_db;
USE agriculture_db;
CREATE TABLE IF NOT EXISTS irrigation_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    crop VARCHAR(50),
    season VARCHAR(10),
    altitude VARCHAR(10),
    soil_type VARCHAR(50),
    water_requirement_mm_day INT,
    irrigation_strategy VARCHAR(255),
    total_water_requirement_m3 BIGINT
);"

# Copy CSV file into the MySQL Docker container
echo "🚀 Copying CSV file into Docker container..."
docker cp "$CSV_FILE" "$CONTAINER_NAME":"$DEST_PATH"

# Import CSV into MySQL
echo "📂 Importing CSV into MySQL..."
docker exec -i "$CONTAINER_NAME" mysql -u root -proot --local-infile=1 -e "
USE agriculture_db;
LOAD DATA INFILE '${DEST_PATH}$(basename "$CSV_FILE")'
INTO TABLE irrigation_data
FIELDS TERMINATED BY ',' 
ENCLOSED BY '\"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(crop, season, altitude, soil_type, water_requirement_mm_day, irrigation_strategy, total_water_requirement_m3);
"


echo "✅ Data import complete!"

# Verify import
echo "📊 Verifying data import..."
docker exec -i "$CONTAINER_NAME" mysql -u root -proot -e "
USE agriculture_db;
SELECT COUNT(*) FROM irrigation_data;
"