
# dowload the parguet files from the cloudfront and store the files in the data folder, if the data folder does not exist, create one
import requests
import os
import sys

# Jan url pull
jan_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet'
response_jan = requests.get(jan_url)    
feb_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-02.parquet'
response_feb = requests.get(feb_url)
mar_url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-03.parquet'
response_mar = requests.get(mar_url)

# check if the data folder exists, if not create one
if not os.path.exists('data'):
    os.makedirs('data')

# write the parquet files to the data folder
with open('data/green_tripdata_2023-01.parquet', 'wb') as file:
    file.write(response_jan.content)
with open('data/green_tripdata_2023-02.parquet', 'wb') as file:
    file.write(response_feb.content)
with open('data/green_tripdata_2023-03.parquet', 'wb') as file:
    file.write(response_mar.content)    
    
# make this file executable from the command line
if __name__ == '__main__':
    pass    
   
    
