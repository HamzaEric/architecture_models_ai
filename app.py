import spaces  
import gradio as gr
import torch
from diffusers import AutoPipelineForText2Image


if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Loading model on {device}...")
pipeline = AutoPipelineForText2Image.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=dtype
).to(device)


pipeline.load_lora_weights("./LoRA Weights", weight_name="pytorch_lora_weights.safetensors")
print("Model and LoRA loaded successfully!")



@spaces.GPU(duration=60)
def generate_architectural_models(prompt, negative_prompt, num_images, inference_steps, guidance_scale):
    if not prompt:
        # Triggers a pop-up warning in Gradio if the prompt is empty
        raise gr.Error("Please enter a prompt first.")
        
    results = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images
    ).images
    
    return results


# --- TWO-COLUMN GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Architectural Scale Model Image Gen-AI") as demo:
    gr.Markdown("# 🏛️ Architectural Scale Model Image Gen-AI")
    gr.Markdown("Enter a description to generate custom physical scale models using a fine-tuned AI.")
    
    with gr.Row():
        # LEFT COLUMN (Generation Settings)
        with gr.Column(scale=1):
            gr.Markdown("### Generation Settings")
            
            prompt = gr.Textbox(
                label="Prompt",
                value="A physical scale model of a modern museum, archmodel style, wooden base, studio lighting",
                lines=4
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt (Things to avoid)",
                value="photorealistic, real life building, people, blurry, low resolution, messy background"
            )
            
            num_images = gr.Slider(
                label="Images to Generate", min_value=1, max_value=4, value=2, step=1
            )
            
            inference_steps = gr.Slider(
                label="Quality (Inference Steps)", min_value=20, max_value=50, value=30, step=1
            )
            
            guidance_scale = gr.Slider(
                label="Prompt Adherence (Guidance Scale)", min_value=5.0, max_value=15.0, value=7.5, step=0.1
            )
            
            generate_btn = gr.Button("🎨 Generate Architecture Models", variant="primary")
            
        # RIGHT COLUMN (Output Gallery)
        with gr.Column(scale=1):
            gr.Markdown("### Output Gallery")
            
            output_gallery = gr.Gallery(
                label="Generated Models",
                show_label=False,
                columns=2,
                object_fit="contain",
                height="auto"
            )
            
    # Connect the button click to the inference function and update the gallery
    generate_btn.click(
        fn=generate_architectural_models,
        inputs=[prompt, negative_prompt, num_images, inference_steps, guidance_scale],
        outputs=[output_gallery]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
