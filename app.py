import os
from pathlib import Path
import streamlit as st
from groq import Groq


class GroqAPI:
    def __init__(self, model_name: str):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name

    def _response(self, message):
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=message,
            temperature=0,
            max_tokens=8192,
            stream=True,
            stop=None,
        )

    def response_stream(self, message):
        for chunk in self._response(message):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class Message:
    system_prompt: str = (
        """あなたは優しいAIです。以下のユーザの入力に日本語で返答して下さい。"""
    )

    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "system",
                    "content": self.system_prompt,
                }
            ]

    def add(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})

    def display_chat_history(self):
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def display_stream(self, generater):
        with st.chat_message("assistant"):
            return st.write_stream(generater)


class ModelSelector:
    def __init__(self):
        self.models = ["llama3-8b-8192", "llama3-70b-8192"]

    def select(self):
        with st.sidebar:
            st.sidebar.title("groq chat")
            return st.selectbox("", self.models)


def main():
    user_input = st.chat_input("質問などを入力してください")
    model = ModelSelector()
    selected_model = model.select()

    message = Message()

    if user_input:
        llm = GroqAPI(selected_model)

        message.add("user", user_input)
        message.display_chat_history()

        response = message.display_stream(
            generater=llm.response_stream(st.session_state.messages)
        )
        message.add("assistant", response)


if __name__ == "__main__":
    main()
