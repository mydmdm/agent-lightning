import os
import sys
import json
import time

from urllib.parse import urlparse
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, model_validator, PrivateAttr
from dotenv import dotenv_values, load_dotenv


class AgentRunConfig(BaseModel):
    """The configuration to run both agent and game (environment)."""

    # --- Required Configuration ---
    game_server_url: str
    openai_base_url: str
    model: str

    # --- Optional Configuration ---
    # for agent runtime configuration
    max_tries: int = 20 # number of max tries allowed in a game
    prompt_for_new_guess: str = "simple.md.jinja"
    # specific for llm api
    use_legacy_completion: bool = False
    max_response_length: int = 2000
    gen_parameters: Dict = Field(default_factory=dict)

    engine: str = "vllm"
    dotenv_file: Optional[str] = None
    logger_filename: Optional[str] = None

    @model_validator(mode="after")
    def setup_runtime(self):
        if self.engine != "vllm":
            raise ValueError(f"Unsupported engine: {self.engine}")
        if self.dotenv_file:
            load_dotenv(self.dotenv_file)
        return self

    # --- Class constructors ---
    @classmethod
    def from_json_file(cls, path: str | Path) -> "AgentRunConfig":
        """Load config from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return cls(**data)

    # --- Properties ---
    # game server setting
    @property
    def game_port(self) -> int:
        return urlparse(self.game_server_url).port

    @property
    def game_ready_text(self) -> str:
        return "Application startup complete."

    @property
    def game_cmd(self) -> List[str]:
        return [
            sys.executable, "-m", "uvicorn", "game:app",
            "--host", "0.0.0.0",
            "--port", str(self.game_port),
        ]

    # inference engine setting

    @property
    def llm_engine_port(self) -> int:
        return urlparse(self.openai_base_url).port

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY", None)

    @property
    def vllm_cmd(self) -> List[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.llm_engine_port),
        ]
        if self.openai_api_key:
            cmd.extend(["--api-key", self.openai_api_key])
        return cmd



