
import os
import json
import requests
import subprocess
import httpx
import asyncio

from contextlib import contextmanager, AbstractContextManager
from typing import Any, Iterable, List, Dict


@contextmanager
def safe_open(filepath: str, mode: str = "w", *args, **kwargs):
    """Open a file safely for writing.
    If the parent folder doesn't exist, create it first.
    Works just like built-in open().
    """
    if "w" in mode or "a" in mode or "x" in mode:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, mode, *args, **kwargs) as f:
        yield f


def dump_jsonl(filepath: str, data: Iterable[dict[str, Any]]):
    """Serialize a list of dicts to a JSONL file."""
    with safe_open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> list[dict[str, Any]]:
    """Deserialize a JSONL file into a list of dicts."""
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


class DaemonServer(AbstractContextManager):
    """Base class for starting and managing a daemon subprocess."""

    def __init__(self, cmd: list[str], name: str):
        self.cmd = cmd
        self.name = name
        self.process: subprocess.Popen | None = None

    def __enter__(self):
        self.process = subprocess.Popen(self.cmd)
        print(f"[{self.name}] Started (PID={self.process.pid}), command: \n>>> {' '.join(self.cmd)}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            print(f"[{self.name}] Terminated (PID={self.process.pid})")
        return False  # don't suppress exceptions


# def completion(model: str, input: List[Dict[str, Any]] | str, chat: bool = True, **kwargs) -> str:
#     """Call an OpenAI-compatible Chat Completions API, with the following environment variables:
#     - OPENAI_API_BASE: The base URL for the API (e.g. "https://api.openai.com/v1" or "http://localhost:8000/v1")
#     - OPENAI_API_KEY: The API key for authentication

#     Args:
#         model (str): model name that can be accepted by the API
#         input (List[Dict[str, Any]] | str): the input that to be used as input of models, format determined by chat
#             - when chat is True, input should be a list of chat messages, each a dict with {"role": str, "content": str}
#             - when chat is False, input should be a string
#         chat (bool): whether use the chat/completions interface.

#     Returns:
#         The assistant's reply as a string.
#     """
#     payload = {
#         "model": model,
#         "stream": False,
#     }
#     endpoint = f"{os.environ['OPENAI_API_BASE']}".rstrip("/")
#     if chat:
#         if not endpoint.endswith("chat/completions"):
#             endpoint = f"{endpoint}/chat/completions"
#         payload["messages"] = input
#     else:
#         if not endpoint.endswith("/completions"):
#             endpoint = f"{endpoint}/completions"
#         payload["prompt"] = input

#     headers = {
#         "Content-Type": "application/json",
#     }
#     if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]: # key is not empty
#         headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"

#     payload.update(**kwargs)

#     response = requests.post(endpoint, headers=headers, json=payload)
#     response.raise_for_status()  # raises if error

#     data = response.json()
#     return data["choices"][0]["message"]["content"] if chat else data["choices"][0]["text"]


def extract_text_between_tags(s: str, tag: str) -> str:
    # only extract the first item with this tag
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    start_idx = s.find(start_tag)
    end_idx = s.find(end_tag)
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        raise ValueError(f"Tag <{tag}> not found or malformed in string.")
    return s[start_idx + len(start_tag):end_idx].strip()



async def completion_async(
    model: str,
    input: List[Dict[str, Any]] | str,
    chat: bool = True,
    stream: bool = False,
    **kwargs
) -> str:
    """Async call to an OpenAI-compatible API using httpx.

    Args:
        model: Model name.
        input: Messages (chat=True) or prompt (chat=False).
        chat: Whether to use /chat/completions endpoint.
        stream: Whether to stream tokens.
        kwargs: Extra payload options.

    Returns:
        The assistant's reply as a string.
    """
    base = os.environ["OPENAI_API_BASE"].rstrip("/")
    endpoint = f"{base}/chat/completions" if chat else f"{base}/completions"

    headers = {"Content-Type": "application/json"}
    if os.getenv("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"

    payload = {"model": model, "stream": stream}
    if chat:
        payload["messages"] = input
    else:
        payload["prompt"] = input
    payload.update(kwargs)

    async with httpx.AsyncClient(timeout=None) as client:
        if stream:
            # Handle streaming responses
            reply_chunks: List[str] = []
            async with client.stream("POST", endpoint, headers=headers, json=payload) as r:
                async for line in r.aiter_lines():
                    if line and line.startswith("data: "):
                        data = line[len("data: "):]
                        if data.strip() == "[DONE]":
                            break
                        chunk = json.loads(data)
                        delta = chunk["choices"][0].get("delta" if chat else "text", {})
                        if chat and "content" in delta:
                            reply_chunks.append(delta["content"])
                        elif not chat and delta:
                            reply_chunks.append(delta)
            return "".join(reply_chunks)
        else:
            # Regular non-streaming call
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"] if chat else data["choices"][0]["text"]


def completion(
    model: str,
    input: List[Dict[str, Any]] | str,
    chat: bool = True,
    stream: bool = False,
    **kwargs
) -> str:
    """Sync wrapper around completion_async.

    If already inside an event loop (e.g. Jupyter or FastAPI),
    use asyncio.run_coroutine_threadsafe to avoid RuntimeError.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We are in an async environment already
        future = asyncio.run_coroutine_threadsafe(
            completion_async(model, input, chat=chat, stream=stream, **kwargs),
            loop
        )
        return future.result()
    else:
        # Normal sync context
        return asyncio.run(completion_async(model, input, chat=chat, stream=stream, **kwargs))