import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image
from PIL import Image

# --- PAGE CONFIGURATION ---
st.set_page_config(layout="wide")


# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_model():
    # Automatically detect if a GPU is available (Nvidia, Apple Silicon, or fallback to CPU)
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32  # CPU requires float32

    # Load base Stable Diffusion 1.4
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype
    ).to(device)

    # Load your custom LoRA weights from the local 'model' folder
    pipeline.load_lora_weights("./LoRA Weights", weight_name="pytorch_lora_weights.safetensors")

    return pipeline


# --- UI LAYOUT ---
st.title("Architectural Scale Model Image Gen-AI")
st.markdown("Enter a description to generate custom physical scale models using a fine-tuned AI.")

# Load the pipeline
with st.spinner("Loading AI model... (this takes a moment on startup)"):
    pipeline = load_model()

# --- TWO-COLUMN MAIN LAYOUT ---
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.subheader("Generation Settings")

    prompt = st.text_area(
        "Prompt",
        value="A physical scale model of a modern museum, archmodel style, wooden base, studio lighting",
        height=120
    )

    negative_prompt = st.text_input(
        "Negative Prompt (Things to avoid)",
        value="photorealistic, real life building, people, blurry, low resolution, messy background"
    )

    num_images = st.slider("Images to Generate", min_value=1, max_value=4, value=2)
    inference_steps = st.slider("Quality (Inference Steps)", min_value=20, max_value=50, value=30)
    guidance_scale = st.slider("Prompt Adherence (Guidance Scale)", min_value=5.0, max_value=15.0, value=7.5)

    generate_btn = st.button("Generate Architecture Models", type="primary", use_container_width=True)

with right_col:
    st.subheader(" Output Gallery")

    # Placeholder or container for results
    output_container = st.container()

    if generate_btn:
        if prompt:
            with st.spinner(f"Rendering {num_images} image(s)... Please wait."):
                results = pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=num_images
                ).images

                with output_container:
                    # If generating multiple images, stack them nicely or put them in sub-columns
                    for idx, img in enumerate(results):
                        st.image(img, caption=f"Variation {idx + 1}", use_container_width=True)
                st.success("Generation complete!")
        else:
            with output_container:
                st.warning("Please enter a prompt first.")
    else:
        with output_container:
            st.info(
                "Configure your settings on the left and click **Generate Architecture Models** to see results here.")