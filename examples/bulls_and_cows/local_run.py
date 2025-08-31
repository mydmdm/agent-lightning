import os
import sys
import json
import time
import argparse
import asyncio

from dataclasses import dataclass, field
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import Optional, TextIO

from game import BullsAndCowsClient, interactive_mode
from agent import agent_function, completion
from utils import DaemonServer, dump_jsonl, load_jsonl, safe_open, completion


@dataclass
class AgentRunner:
    # --- Required Configuration ---
    # game
    game_server_url: str
    # model and engine
    openai_base_url: str
    model: str

    # --- Optional Configuration ---
    # game
    game_seeds: list[dict] = field(init=False)
    # engine and model
    engine: str = "vllm"
    gen_parameters: dict = field(default_factory=dict)
    # environmental settings
    dotenv_file: str = None
    # running
    game_client: BullsAndCowsClient = field(init=False)
    logger_filename: str = None
    loggers: list[TextIO] = field(default_factory=list, init=False)

    def __post_init__(self):
        # Validate the configuration values
        if self.engine != "vllm":
            raise ValueError(f"Unsupported engine: {self.engine}")
        # configure env vars for OpenAI-compatible clients
        os.environ["OPENAI_API_BASE"] = self.openai_base_url
        if self.dotenv_file:
            load_dotenv(self.dotenv_file)
        else:
            os.environ["OPENAI_API_KEY"] = ""
        # create game client
        self.game_client = BullsAndCowsClient(self.game_server_url)
        if self.logger_filename:
            with safe_open(self.logger_filename, "w", encoding="utf-8") as f:
                pass # just to create/clear the file
            self.loggers.append(open(self.logger_filename, "a", encoding="utf-8"))

    def close(self):
        for logger in self.loggers:
            logger.close()

    @property
    def game_port(self) -> int:
        return urlparse(self.game_server_url).port

    @property
    def llm_engine_port(self) -> int:
        return urlparse(self.openai_base_url).port

    @property
    def openai_api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY", None)

    @property
    def vllm_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.llm_engine_port),
        ]
        if self.openai_api_key:
            cmd.append("--api-key")
            cmd.append(self.openai_api_key)
        return cmd

    @property
    def game_cmd(self) -> list[str]:
        # uvicorn game:app --host 0.0.0.0 --port 8000
        cmd = [
            sys.executable, "-m", "uvicorn", "game:app",
            "--host", "0.0.0.0",
            "--port", str(self.game_port),
        ]
        return cmd

    # runner method

    def run_sample(self, secret):
        game_id = self.game_client.start_new_game(secret)
        flag, tries = agent_function(self.model, self.game_client, game_id)
        score = self.game_client.score(game_id)
        dic = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "game_id": game_id,
            "secret": secret,
            "flag": flag,
            "tries": tries,
            "score": score
        }
        for fn in self.loggers:
            fn.write(json.dumps(dic) + "\n")
        return dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    # create sub command: play -c <config_file>
    play_parser = subparsers.add_parser("play", help="Run the game in interactive mode, user can play it in-person")

    # create sub command: run -c <config_file> -n <num>
    run_parser = subparsers.add_parser("run", help="Run the game in a batch mode (num_games in config_file)")
    run_parser.add_argument("-n", "--num", type=int, default=None, help="Number of games to play")

    # create sub command: vllm -c <config_file>
    vllm_parser = subparsers.add_parser("vllm")

    # add config_file for all subcommands
    for p in [play_parser, run_parser, vllm_parser]:
        p.add_argument("-c", "--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        agr = AgentRunner(**config)

    if args.command == "play":
        with DaemonServer(agr.game_cmd, "Game Server") as game_server:
            assert agr.game_client.wait_until_health(timeout=20, step=2)
            interactive_mode(agr.game_server_url)
        sys.exit(0)

    if args.command == "run":
        # test llm inference available
        print(completion(agr.model, "Tell me a joke", chat=False, max_tokens=2000))
        # Handle the run command
        with DaemonServer(agr.game_cmd, "Game Server") as game_server:
            assert agr.game_client.wait_until_health(timeout=20, step=2)
            for i in range(args.num):
                secret = agr.game_client.generate_a_secret()
                dic = agr.run_sample(secret)
                print(dic)
        agr.close()
        sys.exit(0)

    if args.command == "vllm":
        # start vllm server and wait until user interrupt
        with DaemonServer(agr.vllm_cmd, "vLLM Server") as v:
            while True:
                time.sleep(10)
