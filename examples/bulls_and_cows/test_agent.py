import json
import time
import requests_mock

from utils import completion, DaemonServer
from local_run import AgentRunner


def get_fake_response(data: str, chat: bool):
    if chat:
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": data},
                    "finish_reason": "stop"
                }
            ]
        }
    else:
        return {
            "id": "cmpl-456",
            "object": "text_completion",
            "created": 1234567891,
            "model": "gpt-3.5-legacy",
            "choices": [
                {
                    "index": 0,
                    "text": data,
                    "finish_reason": "stop"
                }
            ]
        }

with open("config.json") as f:
    config = json.load(f)
# replace with testing config
config["openai_base_url"] = "https://mock.openai.com/v1"
agr = AgentRunner(**config)


def test_chat_completion():
    with requests_mock.Mocker() as m:
        # test chat and completion interface
        test_data = "Hello!"
        m.post(agr.openai_base_url + "/chat/completions", json=get_fake_response(test_data, chat=True))
        m.post(agr.openai_base_url + "/completions", json=get_fake_response(test_data, chat=False))

        response = completion(agr.model, "Tell me something", chat=True)
        assert response == test_data

        response = completion(agr.model, "Tell me something", chat=False)
        assert response == test_data


def test_agent_function():
    with requests_mock.Mocker(real_http=True) as m, DaemonServer(agr.game_cmd, "Game Server") as g:
        test_data = "<think>How can I make a new guess</think><guess>1 2 3 4</guess>"
        m.post(agr.openai_base_url + "/chat/completions", json=get_fake_response(test_data, chat=True))
        m.post(agr.openai_base_url + "/completions", json=get_fake_response(test_data, chat=False))

        assert agr.game_client.wait_until_health(timeout=20, step=2)

        dic = agr.run_sample(secret=[1,2,3,4])
        assert dic["secret"] == [1,2,3,4]
        for key, val in {"win": True, "num_tries": 1, "num_invalid": 0, "secret": [1,2,3,4]}.items():
            assert dic["score"][key] == val

        dic = agr.run_sample(secret=[5, 6, 7, 8])
        assert dic["secret"] == [5, 6, 7, 8]
        for key, val in {"win": False, "num_invalid": 0, "secret": [5, 6, 7, 8]}.items():
            assert dic["score"][key] == val