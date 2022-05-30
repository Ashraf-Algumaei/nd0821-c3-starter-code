"""
Script used to send a sample request to the API and prints the response back
"""
import requests

user_request = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "educationNum": 9,
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capitalGain": 5000,
    "capitalLoss": 0,
    "hoursPerWeek": 45,
    "nativeCountry": "United-States"
}

resp = requests.get('https://udacity-project3-ashraf.herokuapp.com/')

print(resp.status_code)
print(resp.json())