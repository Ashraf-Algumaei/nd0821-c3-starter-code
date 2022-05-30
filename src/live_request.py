"""
Script used to send a sample request to the API and prints the response back
"""
import requests

response = requests.post('/url/to/query/')

print(response.status_code)
print(response.json())