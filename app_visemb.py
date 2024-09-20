import streamlit as st
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from attacks.visemb import similar_visual_embedding_attack
from models import get_qwen2_vl, generate_qwen2_vl_with_image
from PIL import Image

load_dotenv()

st.title("Visual Embedding Attack")

with st.sidebar:
    st.header("Settings")

    model_name = st.selectbox(
        "Model",
        ("Qwen2-VL",),
    )

    query = st.text_area("Query", value="describe the image.")

    st.write("both images must have the same size")

    uploaded_safe_image = st.file_uploader("Upload Safe Image", type=["jpg", "jpeg", "png"], help='width and height must be divisible by 28')
    uploaded_unsafe_image = st.file_uploader("Upload Unsafe Image", type=["jpg", "jpeg", "png"], help='width and height must be divisible by 28')

    learning_rate = st.slider(
        "Learning Rate", min_value=0.001, max_value=0.1, value=0.001, step=0.001, format='%g',
    )

    num_steps = st.slider(
        "Number of Steps", min_value=100, max_value=2000, value=500, step=50
    )

if uploaded_safe_image and uploaded_unsafe_image:
    img_safe = Image.open(uploaded_safe_image)
    img_unsafe = Image.open(uploaded_unsafe_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_safe, caption="Safe Image", use_column_width=True)
    with col2:
        st.image(img_unsafe, caption="Unsafe Image", use_column_width=True)

    if model_name == "Qwen2-VL":
        model, processor = get_qwen2_vl()

    safe_image_placeholder = st.empty()

    loss_plot_placeholder = st.empty()

    loss_values = []

    attack_generator = similar_visual_embedding_attack(model, processor, img_safe, img_unsafe, num_steps=num_steps, lr=learning_rate)

    for step, (updated_img_safe, loss) in enumerate(attack_generator):
        safe_image_placeholder.image(updated_img_safe, caption=f"Safe Image at Step {step + 1}", use_column_width=True)
        loss_values.append(loss)
        fig, ax = plt.subplots()
        ax.plot(loss_values, label="Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        loss_plot_placeholder.pyplot(fig)

    final_col1, final_col2 = st.columns(2)
    with final_col1:
        st.image(updated_img_safe, caption="Final Safe Image", use_column_width=True)
    with final_col2:
        fig, ax = plt.subplots()
        ax.plot(loss_values, label="Final Loss")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.legend()
        st.pyplot(fig)

    st.chat_message("user").write(query)
    st.chat_message("assistant").markdown(
        generate_qwen2_vl_with_image(
            model,
            processor,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": query},
                    ],
                }
            ],
            updated_img_safe,
            1024,
        )
    )
