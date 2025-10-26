"""
Simple LLM Application using Groq and OpenAI API
"""
from appconfig import env_config
from typing import List, Dict

class LLMApp:

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        chatbot_name: str = "Atlas",
        default_system_prompt: str | None = None,
    ):
        
        """
        Initialize the LLM application (supports Groq + OpenAI)
        
        Args:
            api_key: Provider API key (if None, reads from env var)
            model: Model to use for completions
            chatbot_name: A friendly assistant identity
            default_system_prompt: Fallback system prompt if none is supplied
        """

        self.model = model
        self.provider = self._infer_provider(model)
        self.chatbot_name = chatbot_name
        self.default_system_prompt = (
            default_system_prompt
            or (
                f"You are {chatbot_name}, a concise, helpful, and friendly AI assistant. "
                f"Your name is {chatbot_name}. You must refer to yourself as {chatbot_name} at all times; "
                "never call yourself ChatGPT or any other name. Prefer clear step-by-step answers, "
                "cite assumptions, and ask only one clarifying question if critically necessary. "
                "Avoid introducing unrelated topics."
            )
        )
        self.conversation_history: List[Dict[str, str]] = []

        # Configure API key & client by provider
        if self.provider == "groq":
            from groq import Groq
            self.api_key = api_key or env_config.groq_api_key
            if not self.api_key:
                raise ValueError("Groq API key must be provided or set in `GROQ_API_KEY`.")
            self.client = Groq(api_key=self.api_key)
        else:
            # OpenAI
            from openai import OpenAI
            self.api_key = api_key or env_config.openai_api_key
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided or set in `OPENAI_API_KEY`.")
            self.client = OpenAI(api_key=self.api_key)

    def chat(self, user_message, system_prompt=None, temperature=0.5, max_tokens=1024):
        """
        Send a message and get a response
        
        Args:
            user_message: The user's message
            system_prompt: Optional system prompt to set context
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens in response
            
        Returns:
            The assistant's response text
        """

        # Build messages with persona-aware system prompt
        messages: List[Dict[str, str]] = []

        # Add system prompt if provided
        persona = system_prompt if system_prompt else self.default_system_prompt
        messages.append({"role": "system", "content": persona})

        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Meta overrides for name/model questions (deterministic, avoids drift)
        text = (user_message or "").strip()
        lower = text.lower()
        if lower in {"what is your name", "what's your name", "who are you"}:
            return f"My name is {self.chatbot_name}."
        if "what is your model" in lower or "model version" in lower:
            prov = "OpenAI" if self.provider == "openai" else "Groq"
            return f"I'm {self.chatbot_name}, running on {prov} model `{self.model}`."

        # Add current user's message
        messages.append({"role": "user", "content": text})

        # Make LLM call
        # Make LLM call (provider-specific)
        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            assistant_message = response.choices[0].message.content
        else:
            # --- OpenAI ---
            is_gpt5 = self.model.startswith("gpt-5")

            # Build parameters
            params = {
                "model": self.model,
                "messages": messages,
            }

            # GPT-5 only supports default temperature = 1, so skip custom values
            if not is_gpt5:
                params["temperature"] = temperature

            # GPT-5 uses `max_completion_tokens`; older models use `max_tokens`
            if is_gpt5:
                params["max_completion_tokens"] = max_tokens
            else:
                params["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**params)
            assistant_message = response.choices[0].message.content


        # Update in-memory history
        self.conversation_history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ])

        return assistant_message
    
    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []

    @staticmethod
    def _infer_provider(model: str) -> str:
        """
        Very simple detector: if model starts with 'gpt-' it's OpenAI, otherwise assume Groq.
        You can make this mapping stricter if needed.
        """
        if model.startswith("gpt-"):
            return "openai"
        return "groq"

if __name__=="__main__":

    # Initialize the app
    app = LLMApp()

    # while True:
    message = input(f"What do you want to ask: ")
    response = app.chat(message)
    print(f"\nAssistant Response: {response}\n")
