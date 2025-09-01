import os
import sys
import json
import time
import argparse

from dotenv import load_dotenv
from typing import List, Dict, Any, TextIO
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass, asdict

from game import BullsAndCowsClient, interactive_mode, generate_a_secret
from utils import completion, extract_text_between_tags, DaemonServer, safe_open
from agent_config import AgentRunConfig


@dataclass
class Trial:
    guess: list[int]
    is_valid: bool
    bulls: int = None
    cows: int = None

    # write a method to convert to string
    def __str__(self):
        guess_str = " ".join(str(c) for c in self.guess) if isinstance(self.guess, list) else str(self.guess)
        if self.is_valid:
            return f"<guess>{guess_str}</guess> -> bulls={self.bulls}, cows={self.cows}"
        else:
            return f"<guess>{guess_str}</guess> -> invalid guess"


class BullsAndCowAgent:

    def __init__(self, config: AgentRunConfig):
        self.config = config
        # initialize the essential components
        # --- game client
        self.game_client = BullsAndCowsClient(config.game_server_url)
        # --- llm api
        os.environ["OPENAI_API_BASE"] = config.openai_base_url
        if config.dotenv_file:
            load_dotenv(config.dotenv_file)
        # --- agent function
        self.template_env = Environment(loader=FileSystemLoader('prompts'))
        self.prompt_for_new_guess = self.template_env.get_template(config.prompt_for_new_guess)
        # setup loggers
        self.loggers: list[TextIO] = []
        if config.logger_filename:
            with safe_open(config.logger_filename, "w", encoding="utf-8"):
                pass
            self.loggers.append(open(config.logger_filename, "a", encoding="utf-8"))

    def call_llm(self, input: str | list) -> str:
        return completion(
            self.config.model,
            input,
            chat=False if self.config.use_legacy_completion else True,
            max_tokens=self.config.max_response_length,
            **self.config.gen_parameters
        )

    def play_a_game(self, game_id) -> Dict[str, Dict|List]:
        tries = [] # type: list[Trial]
        for _ in range(self.config.max_tries):
            # render the prompt template
            prompt = self.prompt_for_new_guess.render(tries=[str(t) for t in tries])

            # Get model response
            response = self.call_llm(prompt)
            print("\n\n", prompt, response, "\n\n")
            guess = None

            try:
                guess_str = extract_text_between_tags(response, "guess")
                # remove all letters not in 1~9 by iterate all letters in the string
                guess = [int(c) for c in guess_str if c in "123456789"]
                feedback = self.game_client.check(game_id, guess)
                b, c = feedback["bulls"], feedback["cows"]
                tries.append(Trial(guess=guess, is_valid=True, bulls=b, cows=c))
                if b == 4: # win
                    break
            except Exception:
                if guess is not None:
                    tries.append(Trial(guess=guess, is_valid=False))
        score = self.game_client.score(game_id)
        return {
            "tries": [asdict(t) for t in tries],
            "score": score
        }

    def run(self, num: int):
        for i in range(num):
            game_id = self.game_client.start_new_game([])
            dic = self.play_a_game(game_id)
            print(dic)
            for logger in self.loggers:
                logger.write(json.dumps(dic) + "\n")


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

    config = AgentRunConfig.from_json_file(args.config)

    if args.command == "play":
        with DaemonServer(
            config.game_cmd,
            "Game Server",
            ready_text=config.game_ready_text
        ) as game_server:
            interactive_mode(config.game_server_url)
        sys.exit(0)

    if args.command == "run":
        # test llm inference available
        agent = BullsAndCowAgent(config)
        print(agent.call_llm("Tell me a joke about cats."))
        # Handle the run command
        with DaemonServer(
            config.game_cmd,
            "Game Server",
            ready_text=config.game_ready_text
        ) as game_server:
            agent.run(args.num)
            agent.close()
        sys.exit(0)

    if args.command == "vllm":
        # start vllm server and wait until user interrupt
        with DaemonServer(config.vllm_cmd, "vLLM Server") as v:
            while True:
                time.sleep(10)
