
# Archi AI: Architectural Scale Model Generator

Archi AI is an interactive AI web application that transforms architectural concepts into realistic, physical-style scale models. Rather than producing standard photorealistic buildings, the pipeline enforces a tangible "archmodel" aesthetic—complete with wooden bases, precise geometry, and studio lighting.

##  How It Works

This project is built on **PyTorch** and the **Hugging Face Diffusers** library. 

1. **Base Model:** Uses `CompVis/stable-diffusion-v1-4` as the foundational text-to-image generator.
2. **LoRA Fine-Tuning:** Injects custom Low-Rank Adaptation (LoRA) weights to specialize the outputs strictly toward scale-model architectural photography.
3. **ZeroGPU Acceleration:** The backend is deployed on Hugging Face Spaces using the `@spaces.GPU` decorator. This grants the Gradio frontend dynamic access to enterprise-grade NVIDIA GPUs, delivering rapid, on-demand inference without the overhead of a 24/7 dedicated instance.

##  Features

* **Custom Prompting:** Define the exact style, building type, and environment for your scale model.
* **Granular Output Control:** Sliders to adjust the number of generated variations, inference steps (quality), and guidance scale (prompt adherence).
* **Responsive UI:** A clean, two-column interface built with Gradio 6.0 that allows for instant side-by-side gallery comparisons.

## Tech Stack
* **Machine Learning:** PyTorch, Hugging Face Diffusers, PEFT, Accelerate

* **Interface:** Gradio 6.24.0

* **Infrastructure:** Hugging Face ZeroGPU (Python 3.12)
