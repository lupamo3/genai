from __future__ import annotations
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

class LLMClient:
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: Optional[str] = None,
        chatbot_name: str = "Atlas",
        default_system_prompt: Optional[str] = None,
        ) -> None:
        self.model = model
        self.provider = self._infer_provider(model)
        self.chatbot_name = chatbot_name.strip() or "Atlas"
        self.default_system_prompt = (
            default_system_prompt
            or (
                f"You are {self.chatbot_name}, a concise, friendly AI assistant. "
                f"Your name is {self.chatbot_name}. Always refer to yourself as {self.chatbot_name}. "
                "Prefer step-by-step clarity. Avoid unrelated topics."
            )
        )
        # Late import to avoid hard dependency if keys absent
        if self.provider == "openai":
            from openai import OpenAI
            self.api_key = api_key or env.openai_api_key
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is required for gpt-* models.")
            self.client = OpenAI(api_key=self.api_key)
        else:
            from groq import Groq
            self.api_key = api_key or env.groq_api_key
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is required for llama-* models.")
            self.client = Groq(api_key=self.api_key)


        self.history: List[Dict[str, str]] = []

    def _infer_provider(self, model: str) -> str:
        return "openai" if model.startswith("gpt-") else "groq"


    def clear(self) -> None:
        self.history = []


    def meta_answer(self, prompt: str) -> Optional[str]:
        t = prompt.strip().lower()
        if t in {"what is your name", "what's your name", "who are you"}:
            return f"My name is {self.chatbot_name}."
        if "what is your model" in t or "model version" in t:
            prov = "OpenAI" if self.provider == "openai" else "Groq"
            return f"I'm {self.chatbot_name}, running on {prov} model `{self.model}`."
        return None

    def chat(
        self,
        user_message: str,
        temperature: float = 0.5,
        max_tokens: int = 1024,
        system_prompt: Optional[str] = None,
    ) -> str:
    # Short-circuit meta
        meta = self.meta_answer(user_message)
        if meta:
            self.history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": meta},
            ])
            return meta

        sys_msg = system_prompt if system_prompt else self.default_system_prompt
        msgs: List[Dict[str, str]] = [{"role": "system", "content": sys_msg}] + self.history + [
            {"role": "user", "content": user_message},
        ]


        if self.provider == "openai":
            # GPT-5 note: some variants expect max_completion_tokens
            is_gpt5 = self.model.startswith("gpt-5")
            params = {
                "model": self.model,
                "messages": msgs,
            }
            if not is_gpt5:
                params["temperature"] = temperature
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens
            resp = self.client.chat.completions.create(**params)
            text = resp.choices[0].message.content
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = resp.choices[0].message.content


        self.history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": text},
        ])
        return text