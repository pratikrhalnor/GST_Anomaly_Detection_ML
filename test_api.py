import requests

url = "http://127.0.0.1:5000/predict_batch"
files = {'file': open('data/batch_2025-10-05.csv', 'rb')}

response = requests.post(url, files=files)
print(response.status_code)
print(response.json())
