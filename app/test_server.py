from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from app.server import app

client = TestClient(app)

def test_get_map():
    response = client.get("/map")
    assert response.status_code == 200
    assert "map" in response.json()

def test_reset_valid():
    response = client.post("/reset", json={"width": 10, "height": 10})
    assert response.status_code == 200
    assert "state" in response.json()

def test_reset_invalid():
    response = client.post("/reset", json={"width": -1, "height": -1})
    assert response.status_code == 422

def test_get_state():
    response = client.get("/state")
    assert response.status_code == 200
    assert "state" in response.json()

def test_step_valid():
    response = client.post("/step", json={"action": 1})
    assert response.status_code == 200
    assert "state" in response.json()

def test_step_invalid():
    response = client.post("/step", json={"action": 5})
    assert response.status_code == 422