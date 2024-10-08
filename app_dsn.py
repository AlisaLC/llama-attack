import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from attacks.dsn import DSN_qwen2_vl, DSN_llama
from models import get_llama, get_qwen2_vl, generate_qwen2_vl, generate_llama

load_dotenv()

st.title("DSN Attack")

with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Model",
        ("Llama 3.2", "Qwen2-VL"),
    )

    expected = st.text_area("Expected")

    negative = st.text_area("Refusal")

    alpha = st.slider(
        "Alpha (10^x)", min_value=-10.0, max_value=10.0, value=3.0, step=0.1
    )

    num_steps = st.slider(
        "Number of Steps", min_value=100, max_value=2000, value=500, step=50
    )
    search_width = st.slider(
        "Search Width", min_value=64, max_value=1024, value=512, step=64
    )
    batch_size = st.slider(
        "Batch Size", min_value=32, max_value=256, value=128, step=32
    )
    topk = st.slider(
        "Top-K", min_value=64, max_value=512, value=256, step=64
    )

if prompt := st.chat_input("Query"):
    col1, col2, col3 = st.columns([2, 1, 0.5])

    with col1:
        user_message = st.empty()
        ai_message = st.empty()

        user_message.container()
        ai_message.container()
        user_message.chat_message("user").write(prompt)

    with col2:
        loss_plot = st.empty()

    fig, ax = plt.subplots(figsize=(5, 4))
    loss_values = []

    ax.set_title("Loss Over Time")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")

    if model_name == "Llama 3.2":
        model, tokenizer = get_llama()
        dsn_func = DSN_llama
    elif model_name == "Qwen2-VL":
        model, tokenizer = get_qwen2_vl()
        dsn_func = DSN_qwen2_vl

    for i, (suffix, loss) in enumerate(
        dsn_func(
            model,
            tokenizer,
            prompt,
            expected,
            negative,
            alpha=10 ** alpha,
            num_steps=num_steps,
            search_width=search_width,
            batch_size=batch_size,
            topk=topk,
        )
    ):
        user_message.empty()
        ai_message.empty()
        chat = [{"role": "user", "content": prompt + suffix}]
        user_message.chat_message("user").write(prompt + suffix)

        if model_name == "Llama 3.2":
            output = generate_llama(model, tokenizer, chat)
            response = ai_message.chat_message("assistant").markdown(output)
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

        with col2:
            loss_plot.pyplot(fig)

    user_message.empty()
    ai_message.empty()
    chat = [{"role": "user", "content": prompt + suffix}]
    user_message.chat_message("user").markdown(prompt + suffix)

    if model_name == "Llama 3.2":
        output = generate_llama(model, tokenizer, chat, max_tokens=1024)
        response = ai_message.chat_message("assistant").markdown(output)
    elif model_name == "Qwen2-VL":
        output = generate_qwen2_vl(model, tokenizer, chat, max_tokens=1024)
        response = ai_message.chat_message("assistant").markdown(output)
