# ---------------- Tests ----------------
from fastapi.testclient import TestClient
from game import app

client = TestClient(app)

def test_start_new_game():
    response = client.post("/start_new_game", json={"secret": [1, 2, 3, 4]})
    assert response.status_code == 200
    data = response.json()
    assert "game_id" in data


def test_check_guess():
    # Start new game
    response = client.post("/start_new_game", json={"secret": [1, 2, 3, 4]})
    game_id = response.json()["game_id"]

    # Correct guess
    response = client.post("/check", json={"game_id": game_id, "guess": [1, 2, 3, 4]})
    data = response.json()
    assert data["bulls"] == 4
    assert data["cows"] == 0


def test_check_after_win():
    # Start new game
    response = client.post("/start_new_game", json={"secret": [1, 2, 3, 4]})
    game_id = response.json()["game_id"]

    # Win the game
    client.post("/check", json={"game_id": game_id, "guess": [1, 2, 3, 4]})

    # Try guessing again
    response = client.post("/check", json={"game_id": game_id, "guess": [1, 2, 3, 4]})
    assert response.status_code == 200
    # assert response.json()["detail"] == "Game already won. No further guesses allowed."


def test_invalid_secret():
    response = client.post("/start_new_game", json={"secret": [0, 2, 3, 4]})
    assert response.status_code == 400
    assert "must contain integers from 1 to 9" in response.json()["detail"]


def test_invalid_guess():
    response = client.post("/start_new_game", json={"secret": [1, 2, 3, 4]})
    game_id = response.json()["game_id"]

    response = client.post("/check", json={"game_id": game_id, "guess": [10, 2, 3, 4]})
    assert response.status_code == 400
    assert "must contain integers from 1 to 9" in response.json()["detail"]
