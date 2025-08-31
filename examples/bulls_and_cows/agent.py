import os
from typing import List, Dict, Any
from jinja2 import Environment, FileSystemLoader
from dataclasses import dataclass, asdict

from game import BullsAndCowsClient
from utils import completion, extract_text_between_tags


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


def agent_function(
        model: str,
        game_client: BullsAndCowsClient,
        game_id: str,
        max_tries: int = 20,
        prompt_template: str = "simple.md.jinja",
    ) -> tuple[bool, list[dict]]:
    # load templates from the `prompts` folder
    env = Environment(loader=FileSystemLoader('prompts'))
    template = env.get_template(prompt_template)

    tries = [] # type: list[Trial]

    for _ in range(max_tries):
        # render the prompt template
        prompt = template.render(tries=[str(t) for t in tries])

        # Get model response
        response = completion(
            model, prompt,
            chat=False,
            max_tokens=2000,
        )

        print(response)
        guess = None

        try:
            guess_str = extract_text_between_tags(response, "guess")
            # remove all letters not in 1~9 by iterate all letters in the string
            guess = [int(c) for c in guess_str if c in "123456789"]
            feedback = game_client.check(game_id, guess)
            b, c = feedback["bulls"], feedback["cows"]
            tries.append(Trial(guess=guess, is_valid=True, bulls=b, cows=c))
            if b == 4: # win
                return True, [asdict(t) for t in tries]
        except Exception:
            if guess is not None:
                tries.append(Trial(guess=guess, is_valid=False))

    return False, [asdict(t) for t in tries]
