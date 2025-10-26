"""
Streamlit Frontend for Groq + OPENAI LLM Application
"""
import streamlit as st
from main import LLMApp
import os

# page configuration
st.set_page_config(
    page_title="Simple LLM Chat Application (Groq + OpenAI)",
    page_icon="🤖",
    layout="centered"
)

# initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm_app" not in st.session_state:
    st.session_state.llm_app = None
if "last_config" not in st.session_state:
    st.session_state.last_config = {}

# Title and description
st.title("Groq + OpenAI LLM Chat Application")
st.markdown(
    "Chat with open-source (Groq) or OpenAI models. Enter API keys in the sidebar. "
    "Groq keys: https://console.groq.com • OpenAI keys: https://platform.openai.com"
)
# Implement sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # API key input
    # API keys (optional if present in .env)
    groq_api_key = st.text_input("Groq API Key", type="password", help="Optional if GROQ_API_KEY is set in .env")
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY", "")

    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Optional if OPENAI_API_KEY is set in .env")
    if not openai_api_key:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

    # Model selection
    model = st.selectbox(
        "Model",
        [
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gpt-4o",
            "gpt-4o-mini-search-preview",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        help="Select the model to use"
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Select a value to control response randomness. Higher values make output more random."
    )

    max_tokens = st.slider(
        "Max Tokens (uses Max Completion Tokens for GPT-5)",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="For GPT-5 models this is passed as max_completion_tokens."
    )


    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="You are a helpful assistant...",
        help = "Set the context and behaviour of the assistant"
    )

    chatbot_name = st.text_input(
        "Chatbot Name",
        value="Atlas",
        help="Give your assistant a friendly identity"
    )


    # Clear chat button
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.llm_app:
            st.session_state.llm_app.clear_history()

        st.rerun()

# if st.session_state.llm_app is None:
#     try:
#         st.session_state.llm_app = LLMApp(api_key=api_key, model=model)
#     except Exception as e:
#         st.error(f"Error initializing LLM App: {str(e)}")

def _pick_api_key(model_name: str) -> str | None:
    """Return correct key by model/provider."""
    if model_name.startswith("gpt-"):
        return openai_api_key or None
    return groq_api_key or None

# (Re)initialize the app if any config affecting persona/provider changed
current_config = {
    "model": model,
    "chatbot_name": chatbot_name.strip(),
    "system_prompt": (system_prompt or "").strip(),
}

def _needs_reinit(prev: dict, curr: dict) -> bool:
    if not prev:
        return True
    # Re-init if model/provider OR persona fields changed
    return any(prev.get(k) != curr.get(k) for k in ["model", "chatbot_name", "system_prompt"])

if _needs_reinit(st.session_state.last_config, current_config) or st.session_state.llm_app is None:
    key = _pick_api_key(model)
    try:
        st.session_state.llm_app = LLMApp(
            api_key=key,
            model=model,
            chatbot_name=current_config["chatbot_name"] or "Atlas",
            default_system_prompt=(current_config["system_prompt"] or None),
        )
        st.session_state.last_config = current_config
        # When changing persona/model, clear UI history to avoid mixing contexts
        st.session_state.messages = []
    except Exception as e:
        st.error(f"Error initializing LLM App: {str(e)}")

# Provider badge
provider = "OpenAI" if model.startswith("gpt-") else "Groq"
st.caption(f"**Provider:** {provider} • **Model:** {model}")


# display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # if not api_key:
    #     st.warning("Please enter your Groq API key in the sidebar")
     # Validate API key for chosen provider
    active_key = _pick_api_key(model)
    if not active_key:
        st.warning("Please enter a valid API key for the selected provider in the sidebar (.env also supported).")
 
    else:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"{prompt}"
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # get assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        system_prompt=system_prompt if system_prompt else None,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{response}"
                        }
                    )
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


