import requests 

ride = {
    "PUlocationID": 10,
    "DOlocationID": 50,
    "trip_distance": 10,
}
url = "http://0.0.0.0:9696/predict"

responce = requests.post(url, json = ride)
print(responce.json())