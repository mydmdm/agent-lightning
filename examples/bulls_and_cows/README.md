# Bulls and Cows Dry-Run Example

This example is a dry-run (inference only, without training) of an agent to solve [Bulls and Cows](https://en.wikipedia.org/wiki/Bulls_and_cows) game, to demonstrate how Agent Lightning manage the workflow and capture the necessary data for future training.

## Game Rules

**Bulls and Cows** is a classic code-breaking game of logic and deduction.

- The **server** secretly chooses a 4-digit sequence, where each digit is between **1–9**.
- The **player** tries to guess this sequence.
- After each guess, the server returns two numbers:
  - **Bulls** – digits that are correct **and** in the right position.
  - **Cows** – digits that are correct but in the **wrong position**.

The goal is to find the secret sequence using as few guesses as possible.

**Example**
- Secret: `4271`
- Guess: `1234`
- Result: `1 Bull, 2 Cows` (also represented as `1A2B` in some implementations)
  - (The `2` is a bull, `4` and `1` are cows).

We build a simple game server in `game.py`, which allows players to start a new game, guess numbers, and receive feedback on their guesses. If you're interested to have a try in person, use the interactive mode by
```bash
python local_run.py play -c config.json
```

## Agent

An agent powered by large language models (LLMs) can be used to play the Bulls and Cows game. The agent can analyze the game state, make decisions about which guesses to try, and learn from the feedback it receives.
The agent is implemented in `agent.py` with prompts stored in the folder `./prompts/`.