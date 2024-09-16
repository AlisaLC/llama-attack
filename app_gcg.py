import os
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from attacks.gcg import GCG_qwen2_vl, GCG_llama
from models import get_llama, get_qwen2_vl, generate_qwen2_vl

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_model = os.getenv("OPENAI_MODEL")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def generate_llama(messages: list[dict[str, str]], max_tokens=128):
    return client.chat.completions.create(
        model=openai_model,
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
    )


st.title("GCG Attack")

expected = st.text_area("Expected")

model_name = st.selectbox(
    "Model",
    ("Llama 3.1", "Qwen2-VL"),
)

if prompt := st.chat_input("Query"):
    loss_plot = st.empty()
    user_message = st.empty()
    ai_message = st.empty()
    user_message.container()
    ai_message.container()
    user_message.chat_message("user").markdown(prompt)

    fig, ax = plt.subplots(figsize=(6, 3))
    loss_values = []

    ax.set_title("Loss Over Time")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")

    if model_name == "Llama 3.1":
        model, tokenizer = get_llama()
        gcg_func = GCG_llama
    elif model_name == "Qwen2-VL":
        model, tokenizer = get_qwen2_vl()
        gcg_func = GCG_qwen2_vl

    for i, (suffix, loss) in enumerate(gcg_func(model, tokenizer, prompt, expected)):
        user_message.empty()
        ai_message.empty()
        chat = [{"role": "user", "content": prompt + suffix}]
        user_message.chat_message("user").markdown(prompt + suffix)

        if model_name == "Llama 3.1":
            stream = generate_llama(chat)
            response = ai_message.chat_message(
                "assistant").write_stream(stream)
        elif model_name == "Qwen2-VL":
            output = generate_qwen2_vl(model, tokenizer, chat)
            response = ai_message.chat_message("assistant").markdown(output)

        loss_values.append(loss)

        ax.clear()
        ax.plot(loss_values, label='Loss')
        ax.set_title("Loss Over Time")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.legend()

        loss_plot.pyplot(fig)
    
    user_message.empty()
    ai_message.empty()
    chat = [{"role": "user", "content": prompt + suffix}]
    user_message.chat_message("user").markdown(prompt + suffix)

    if model_name == "Llama 3.1":
        stream = generate_llama(chat, max_tokens=1024)
        response = ai_message.chat_message(
            "assistant").write_stream(stream)
    elif model_name == "Qwen2-VL":
        output = generate_qwen2_vl(model, tokenizer, chat, max_tokens=1024)
        response = ai_message.chat_message("assistant").markdown(output)