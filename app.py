import gradio as gr
import torch
from diffusers import ZImagePipeline
import os
import time
import random
import psutil

# --- CONFIGURATION ---
BASE_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
print("🚀 Starting Z-Image-Turbo GUI...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device detected: {device}")

if torch.cuda.get_device_capability()[0] >= 10:
    print("✨ RTX 50-Series (Blackwell) detected! Unleashing power.")

print(f"Loading model: {MODEL_ID} ...")
try:
    # We load in bfloat16 mode for maximum speed
    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    ).to(device)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Note: Did you run 'pip install git+https://github.com/huggingface/diffusers'?")
    exit()

# --- FUNCTIONS ---

def shutdown_server():
    """Kill switch."""
    print("Shutting down...")
    os._exit(0)

def open_output_folder():
    """Opens the output folder in Windows Explorer."""
    path = os.path.abspath(OUTPUT_DIR)
    os.startfile(path)

def get_system_stats():
    """System Monitor (CPU/RAM/VRAM)."""
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    
    vram_display = "N/A"
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = total - free
        used_gb = used / (1024**3)
        total_gb = total / (1024**3)
        percent = (used / total) * 100
        vram_display = f"{used_gb:.1f}GB / {total_gb:.1f}GB ({percent:.0f}%)"
    
    return f"""
    <div style="display: flex; gap: 24px; font-family: monospace; font-size: 20px; color: #eee; font-weight: bold; align-items: center;">
        <span>🖥️ CPU: {cpu}%</span>
        <span>🧠 RAM: {ram}%</span>
        <span>🎮 VRAM: {vram_display}</span>
    </div>
    """

def generate_image(prompt, steps, seed, width, height):
    if not prompt:
        return None, "Error: Please enter a prompt."
    
    if seed == -1:
        seed = random.randint(0, 2147483647)
    
    print(f"🎨 Gen: '{prompt}' | Steps: {steps} | Seed: {seed}")
    
    generator = torch.Generator(device=device).manual_seed(int(seed))
    
    start_time = time.time()
    
    try:
        image = pipe(
            prompt=prompt,
            num_inference_steps=int(steps),
            guidance_scale=0.0,  
            width=int(width),
            height=int(height),
            max_sequence_length=1024,
            generator=generator
        ).images[0]
        
        duration = time.time() - start_time
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"Z-Image_{timestamp}_seed{seed}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        image.save(save_path)
        return image, f"✅ Done in {duration:.2f}s! Saved to: {save_path}"
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, f"Error: {str(e)}"

# --- GUI LAYOUT ---
with gr.Blocks(title="Z-Image-Turbo Local") as demo:
    
    with gr.Row(variant="compact"):
        gr.Markdown("# ⚡ Z-Image-Turbo (RTX 5090 Edition)")
        system_stats = gr.HTML(value="Loading stats...")

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt", 
                placeholder="Describe your image... (e.g., 'A futuristic cyberpunk city with neon lights, realistic, 8k')",
                lines=3
            )
            
            with gr.Accordion("⚙️ Advanced Settings", open=True):
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=50, value=8, step=1, label="Steps (4-8 recommended)")
                    seed = gr.Slider(minimum=-1, maximum=2147483647, value=-1, step=1, label="Seed (-1 = Random)")
                
                with gr.Row():
                    width = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Width")
                    height = gr.Slider(minimum=512, maximum=2048, value=1024, step=64, label="Height")

            generate_btn = gr.Button("⚡ Generate Image", variant="primary")
            
            with gr.Row():
                open_folder_btn = gr.Button("📂 Open Output Folder")
                exit_btn = gr.Button("🛑 Shutdown", variant="stop")

        with gr.Column(scale=2):
            output_img = gr.Image(label="Result", type="pil")
            status_text = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown(
                """
                If you find this tool helpful, feel free to support me by following my 
                <a href="https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=5d3AbCKgR3GemCemctb8FA" target="_blank" style="color: #1DB954; font-weight: bold; text-decoration: none;">Spotify</a> 
                profile. Every follower counts!
                """
            )

    # --- EVENTS ---
    timer = gr.Timer(1.0)
    timer.tick(fn=get_system_stats, outputs=system_stats)

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, steps, seed, width, height], # time_shift removed
        outputs=[output_img, status_text]
    )
    
    open_folder_btn.click(fn=open_output_folder)
    exit_btn.click(fn=shutdown_server)

if __name__ == "__main__":
    demo.launch(inbrowser=True)