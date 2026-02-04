import os
import sys
import logging
import asyncio

# Allow running this file directly (python llm/client_manager.py)
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.models import ModelSelector, build_chat_request_kwargs


class ClientManager:

    def __init__(self, args):
        self.args = args
        self.mode: str = args.mode
        self.model: str = args.model

        selector = ModelSelector(mode=self.mode, model_name=self.model)
        self.client = getattr(selector, "client", None)

    def chat_completion(
        self,
        messages: list,
    ):

        request_kwargs = build_chat_request_kwargs(
            messages=messages,
            model=self.model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            timeout=self.args.timeout,
        )

        response = self.client.chat.completions.create(**request_kwargs)

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()

        return None

    async def chat_completion_async(
        self,
        messages: list,
    ):

        request_kwargs = build_chat_request_kwargs(
            messages=messages,
            model=self.model,
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            timeout=self.args.timeout,
        )


        response = await asyncio.to_thread(
            self.client.chat.completions.create, **request_kwargs
        )

        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()

        return None

    async def aclose(self):
        return None