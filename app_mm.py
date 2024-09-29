import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from attacks.mm import multi_modal_attack_qwen2_vl, multi_modal_attack_llama
from models import get_qwen2_vl, generate_qwen2_vl_with_image, get_llama, generate_llama_with_image
from PIL import Image

load_dotenv()

st.title("Multi Modal Attack")

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

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    learning_rate = st.slider(
        "Learning Rate", min_value=0.001, max_value=0.1, value=0.001, step=0.001, format='%g',
    )

    num_steps = st.slider(
        "Number of Steps", min_value=100, max_value=2000, value=500, step=50
    )

if prompt := st.chat_input("Query"):
    img = Image.open(uploaded_image)

    if model_name == "Llama 3.2":
        model, processor = get_llama()
        attack_func = multi_modal_attack_llama
        gen_func = generate_llama_with_image
    elif model_name == "Qwen2-VL":
        model, processor = get_qwen2_vl()
        attack_func = multi_modal_attack_qwen2_vl
        gen_func = generate_qwen2_vl_with_image

    col1, col2, col3 = st.columns([2, 1, 0.5])

    with col1:
        user_message = st.empty()
        ai_message = st.empty()

        user_message.container()
        ai_message.container()
        user_message.chat_message("user").write(prompt)

    with col2:
        image = st.empty()
        loss_plot = st.empty()

    loss_values = []

    attack_generator = attack_func(model, processor, img, prompt, expected, negative, num_steps=num_steps, lr=learning_rate, alpha=10 ** alpha)

    for step, (updated_img, loss) in enumerate(attack_generator):
        user_message.empty()
        ai_message.empty()
        image.image(updated_img, caption=f"Image at Step {step + 1}", use_column_width=True)
        loss_values.append(loss)
        fig, ax = plt.subplots()
        ax.plot(loss_values, label="Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        loss_plot.pyplot(fig)
        user_message.chat_message("user").write(prompt)
        ai_message.chat_message("assistant").markdown(
            gen_func(
                model,
                processor,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                updated_img,
            )
        )

    final_col1, final_col2 = st.columns(2)
    with final_col1:
        st.image(updated_img, caption="Final Image", use_column_width=True)
    with final_col2:
        fig, ax = plt.subplots()
        ax.plot(loss_values, label="Final Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    user_message.chat_message("user").write(prompt)
    ai_message.chat_message("assistant").markdown(
        gen_func(
            model,
            processor,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            updated_img,
            1024,
        )
    )
