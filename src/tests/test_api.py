from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting": "Hello to Udacity ML DevOps - Project 3"}


def test_post_lower_than_50k():
    user_request = {
        "age": 23,
        "workclass": "State-gove",
        "fnlgt": 77516,
        "education": "Bachelors",
        "educationNum": 13,
        "maritalStatus": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capitalGain": 2174,
        "capitalLoss": 0,
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    }

    resp = client.post("/predict", json=user_request)
    assert resp.status_code == 200
    assert resp.json() == {"Salary": "Less than or equal to $50k"}


def test_post_higher_than_50k():
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

    resp = client.post("/predict", json=user_request)
    assert resp.status_code == 200
    assert resp.json() == {"Salary": "Higher than $50k"}
