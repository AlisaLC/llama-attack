import os
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from attacks.gcg import GCG

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def generate(messages: list[dict[str, str]]):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=128,
        stream=True,
    )

st.title("GCG Attack")

expected = st.text_input("Expected")

if prompt := st.chat_input("Query"):
    loss_plot = st.empty()
    user_message = st.empty()
    ai_message = st.empty()
    user_message.container()
    ai_message.container()
    user_message.chat_message("user").markdown(prompt)

    fig, ax = plt.subplots()
    loss_values = []
    
    ax.set_title("Loss Over Time")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")

    for i, (suffix, loss) in enumerate(GCG(prompt, expected)):
        user_message.empty()
        ai_message.empty()
        chat = [{"role": "user", "content": prompt + suffix}]
        user_message.chat_message("user").markdown(prompt + suffix)
        
        stream = generate(chat)
        response = ai_message.chat_message("assistant").write_stream(stream)

        loss_values.append(loss)

        ax.clear()
        ax.plot(loss_values, label='Loss')
        ax.set_title("Loss Over Time")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()

        loss_plot.pyplot(fig)
