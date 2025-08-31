from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, List, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, namedtuple

import json
import uuid
import uvicorn
import sys
import argparse
import requests
import logging
import random
import sqlite3
import time


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        # Log request details
        req_body = await request.body()
        logger.info(f"Request: {request.method} {request.url} {req_body.decode()}")

        response = await call_next(request)

        # Log response details
        # Log Response Body
        # Note: Reading response body requires careful handling to not consume the stream
        # This example assumes a simple text response; for streaming responses, it's more complex.
        res_body = b""
        async for chunk in response.body_iterator:
            res_body += chunk
        logger.info(f"Response: {res_body.decode()}")

        # Recreate response for client
        return Response(content=res_body, media_type=response.media_type, status_code=response.status_code, headers=response.headers)

app = FastAPI()
app.add_middleware(LoggingMiddleware)

GameLog = namedtuple("GameLog", ["action", "content"])
games = defaultdict(list)  # game_id -> List[GameLog]
log_event = lambda game_id, action, content: games[game_id].append(GameLog(action=action, content=content))


def is_valid(code: List[int]) -> tuple[bool, str]:
    """Check if the code (secret or guess) is valid.
    Args:
        code (List[int]): The code to check.
    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating validity and an error message (if any).
                            True for valid and False for not.
    """

    if len(code) != 4:
        return False, f"Validation failed: must have exactly 4 integers"
    if not all(isinstance(x, int) and 1 <= x <= 9 for x in code):
        return False, f"Validation failed: must contain integers from 1 to 9"
    # check only one occurance of each item
    if len(set(code)) != 4:
        return False, f"Validation failed: must contain unique integers"
    return True, ""


class StartGameRequest(BaseModel):
    secret: List[int]  # e.g. [1,2,3,4]

class StartGameResponse(BaseModel):
    game_id: str

@app.post("/start_new_game", response_model=StartGameResponse)
def start_new_game(req: StartGameRequest):
    flag, msg = is_valid(req.secret)
    if not flag:
        logger.error(f"Failed to start new game: {msg}")
        raise HTTPException(status_code=400, detail=msg)
    game_id = str(uuid.uuid4())
    log_event(game_id, "start_new_game", {"secret": req.secret})
    return StartGameResponse(game_id=game_id)


class CheckRequest(BaseModel):
    game_id: str
    guess: List[int]

class CheckResponse(BaseModel):
    game_id: str
    bulls: int
    cows: int

@app.post("/check", response_model=CheckResponse)
def check(req: CheckRequest):
    if req.game_id not in games:
        logger.error(f"Game ID {req.game_id} not found")
        raise HTTPException(status_code=404, detail="Game not found")
    guess = req.guess
    flag, msg = is_valid(guess)
    if not flag:
        log_event(req.game_id, "tries", {"guess": guess, "is_valid": False, "error": msg})
        raise HTTPException(status_code=400, detail=msg)

    # Compute bulls and cows
    secret = games[req.game_id][0].content["secret"]
    bulls = sum(s == g for s, g in zip(secret, guess))
    cows = sum(min(secret.count(d), guess.count(d)) for d in set(guess)) - bulls

    log_event(req.game_id, "tries", {"guess": guess, "is_valid": True, "bulls": bulls, "cows": cows})
    return CheckResponse(game_id=req.game_id, bulls=bulls, cows=cows)


class ScoreRequest(BaseModel):
    game_id: str

class ScoreResponse(BaseModel):
    game_id: str
    win: bool # whether the last guess receives 4 bulls
    num_tries: int
    num_invalid: int # number of invalid guesses
    secret: list[int]

@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    if req.game_id not in games:
        log_event(req.game_id, "score", {"error": "Game not found"})
        raise HTTPException(status_code=404, detail="Game not found")

    events = games[req.game_id]
    # Analyze events to compute score
    tries = [evt for evt in events if evt.action == "tries"]
    secret = events[0].content["secret"]
    if not tries:
        win, num_tries, num_invalid = False, 0, 0
    else:
        win = tries[-1].content["is_valid"] and tries[-1].content["bulls"] == 4
        num_tries = len(tries)
        num_invalid = sum(1 for evt in tries if not evt.content["is_valid"])

    score = dict(win=win, num_tries=num_tries, num_invalid=num_invalid, secret=secret)
    log_event(req.game_id, "score", score)
    return ScoreResponse(game_id=req.game_id, **score)


@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ---------------- Client SDK ----------------
class BullsAndCowsClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def start_new_game(self, secret: List[int]) -> str:
        resp = requests.post(f"{self.base_url}/start_new_game", json={"secret": secret})
        resp.raise_for_status()
        game_id = resp.json()["game_id"]
        return game_id

    def check(self, game_id: str, guess: List[int]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/check", json={"game_id": game_id, "guess": guess})
        resp.raise_for_status()
        result = resp.json()
        return result

    def score(self, game_id: str) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/score", json={"game_id": game_id})
        resp.raise_for_status()
        result = resp.json()
        return result

    def health_check(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            status = resp.json().get("status")
            return status == "healthy"
        except requests.RequestException as e:
            return False

    def wait_until_health(self, timeout: float = 10, step: float = 1) -> bool:
        # Wait until the server is healthy or the timeout is reached
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.health_check():
                logger.debug("Client: Server is healthy")
                return True
            logger.debug(f"Client: Server is not healthy, waiting {step} seconds...")
            time.sleep(step)
        logger.error("Client: Timeout reached, server is not healthy")
        return False

    @staticmethod
    def generate_a_secret() -> list[int]:
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        s = numbers[:4]
        assert is_valid(s)[0]
        return s


def interactive_mode(base_url: str):
    # here to run an interactive mode client
    # start the server directly, use the following command:
    # uvicorn game:app --host 0.0.0.0 --port 8000

    # print the description
    print("Welcome to the Bulls and Cows game!")
    print("Enter 4 digit number as your guess, separated by spaces.")
    print("Enter 0 to end the game and see the final score.")
    # now enter the game loop
    client = BullsAndCowsClient(base_url)
    while True:
        # start a new game with randomly generated secret
        secret = client.generate_a_secret()
        game_id = client.start_new_game(secret)
        while True:
            user_input = input("Enter your guess: ")
            if user_input == "0":
                score = client.score(game_id)
                print("Final score:", score)
                break
            # remove all letters not in 1~9 by iterate all letters in the string
            guess = [int(c) for c in user_input if c in "123456789"]
            try:
                feedback = client.check(game_id, guess)
                b, c = feedback["bulls"], feedback["cows"]
                print(f"Bulls={b}, Cows={c} ({b}A{c}B)")
            except Exception as e:
                print(f"Error occurred: {e}")
    sys.exit(0)

