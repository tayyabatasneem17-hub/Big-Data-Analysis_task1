
# Step 1: Install necessary packages (if not installed)
!pip install pyspark matplotlib seaborn pandas

# Step 2: Import Libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, month, year
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Step 3: Start Spark Session
spark = SparkSession.builder \
    .appName("BigData_Analysis_Task") \
    .getOrCreate()

# Step 4: Create a Synthetic Big Dataset with more variance
data = []
start_date = datetime(2019, 1, 1)

# Generate 100,000 rows with seasonal bias and varied fares
for i in range(100000):
    # Seasonal pickup: more rides in summer (months 6-8) and winter (12-1)
    month_offset = random.choices(
        population=[1,2,3,4,5,6,7,8,9,10,11,12],
        weights=[5,5,5,5,5,20,20,20,5,5,5,5],
        k=1
    )[0]
    day_offset = random.randint(0, 27)
    pickup_time = datetime(2019, month_offset, 1) + timedelta(days=day_offset, minutes=random.randint(0, 1440))
    
    passenger_count = random.randint(1, 6)
    trip_distance = round(random.uniform(1, 50), 2)
    # More variance in fare
    fare_amount = round(trip_distance * random.uniform(1.0, 5.0), 2)
    
    data.append((pickup_time, passenger_count, trip_distance, fare_amount))

columns = ["tpep_pickup_datetime", "passenger_count", "trip_distance", "fare_amount"]
df = spark.createDataFrame(data, columns)

# Step 5: Data Cleaning
df = df.filter(col("trip_distance") > 0).filter(col("fare_amount") > 0)

# Step 6: Analysis Queries & Visualization

# 1. Average fare by passenger count
avg_fare = df.groupBy("passenger_count").agg(avg("fare_amount").alias("avg_fare"))
avg_fare.show()
avg_fare_pd = avg_fare.toPandas()

plt.figure(figsize=(8,5))
sns.barplot(x="passenger_count", y="avg_fare", data=avg_fare_pd, palette="viridis")
plt.title("Average Fare by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Average Fare ($)")
plt.savefig("avg_fare_by_passengers.png")
plt.show()

# 2. Top 5 busiest pickup months
df = df.withColumn("year", year(col("tpep_pickup_datetime")))
df = df.withColumn("month", month(col("tpep_pickup_datetime")))

pickup_trends = df.groupBy("year", "month").count().orderBy(col("count").desc())
pickup_trends.show(5)

pickup_trends_pd = pickup_trends.toPandas()
pickup_trends_pd['month_year'] = pickup_trends_pd['month'].astype(str) + '-' + pickup_trends_pd['year'].astype(str)

top5_months = pickup_trends_pd.head(5)
plt.figure(figsize=(8,5))
sns.barplot(x="month_year", y="count", data=top5_months, palette="magma")
plt.title("Top 5 Busiest Pickup Months")
plt.xlabel("Month-Year")
plt.ylabel("Number of Pickups")
plt.savefig("top5_busiest_months.png")
plt.show()

# 3. Longest trips (Top 5 by distance)
longest_trips = df.orderBy(col("trip_distance").desc()).select(
    "tpep_pickup_datetime", "passenger_count", "trip_distance", "fare_amount"
).limit(5)
longest_trips.show()

longest_trips_pd = longest_trips.toPandas()
# Add trip id for clear x-axis labels
longest_trips_pd['trip_id'] = range(1, len(longest_trips_pd)+1)

plt.figure(figsize=(8,5))
sns.barplot(x='trip_id', y='trip_distance', data=longest_trips_pd, palette="coolwarm")
plt.xticks(ticks=range(5), labels=longest_trips_pd['trip_id'])
plt.title("Top 5 Longest Trips by Distance")
plt.xlabel("Trip ID")
plt.ylabel("Trip Distance (miles)")
plt.savefig("longest_trips.png")
plt.show()

# Step 7: Stop Spark Session
spark.stop()
