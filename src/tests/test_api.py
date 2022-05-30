from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_path():
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json() == {"greeting": "Hello to Udacity ML DevOps - Project 3"}