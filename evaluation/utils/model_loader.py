import os
import logging
from typing import List, Dict
from langchain_openai import ChatOpenAI


logger = logging.getLogger("evaluation")


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        logger.info(f"Loading remote chat model: {model}")

        self.chat = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=1.0
        )
        self.model, self.tok = None, None

    def __call__(self, chat_msgs: List[Dict]) -> str:
        return self.chat.invoke(chat_msgs).content



