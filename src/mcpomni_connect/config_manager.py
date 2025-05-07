import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from dotenv import load_dotenv
from mcpomni_connect.utils import logger


@dataclass
class ConfigManager:
    """Manages configuration and environment variables for the MCP client."""

    config: dict = field(default_factory=dict)
    _instance: Optional["ConfigManager"] = None

    def __new__(cls, *args, **kwargs):
        """实现单例模式。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __post_init__(self) -> None:
        """Initialize configuration with environment variables."""
        load_dotenv()
        self.config["llm_api_key"] = os.getenv("LLM_API_KEY")

        if not self.config.get("llm_api_key"):
            raise ValueError("LLM_API_KEY not found in environment variables")

        config_data = self.load_config("servers_config.json")
        if config_data:
            self.config.update(config_data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

    def load_config(self, file_path: str, merge: bool = False) -> dict:
        """Load server configuration from JSON file.

        Args:
            file_path: Path to the configuration file
            merge: If True, merge the loaded config into self.config

        Returns:
            The loaded configuration dictionary
        """
        config_path = Path(file_path)
        logger.info(f"Loading configuration from: {config_path.name}")
        if config_path.name.lower() != "servers_config.json":
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}, it should be 'servers_config.json'"
            )
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
            if merge:
                self.config.update(config_data)
            return config_data
