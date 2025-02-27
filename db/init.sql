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
);