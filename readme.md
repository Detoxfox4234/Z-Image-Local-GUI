## Z-Image-Turbo Local GUI (Windows) ⚡

A lightning-fast local Gradio Web-UI for the Z-Image-Turbo model. Generate photorealistic images in milliseconds using the power of your local GPU.

**Optimized for NVIDIA RTX 50-Series (Blackwell) & CUDA 12.8.**

![Example Image](preview.png)

*(Generated locally in seconds)*

---

## ✨ Features

- ⚡ **Turbo Speed**: Generates high-quality images in just 4-8 steps.
- 📊 **Real-Time Monitor**: Live display of CPU, RAM, and VRAM usage.
- 🖼️ **Full Control**: Adjustable Resolution (up to 2048x2048), Steps, and Seed.
- 💾 **Auto-Save**: Results are automatically saved to the `outputs` folder.
- 📂 **Quick Access**: Open the output folder directly from the UI.
- 🛑 **VRAM Friendly**: Includes a Shutdown button to instantly close the server.

---

## ⚙️ Installation

### 1. Prerequisites

- Python 3.10 or 3.11 installed.
- Git installed.
- High-end NVIDIA GPU (RTX 3090/4090 or RTX 5090 recommended).

### 2. Clone the Repository

Open PowerShell or Terminal and run:

```bash
git clone https://github.com/Detoxfox4234/Z-Image-Local-GUI.git
cd Z-Image-Local-GUI
```

### 3. Create a Virtual Environment

Highly recommended to keep dependencies clean.

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

We need the latest Diffusers version from source for Z-Image support.

#### For RTX 5090 (Blackwell) Users:

You need PyTorch Nightly for CUDA 12.8 support.

```bash
# 1. Install PyTorch Nightly
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 2. Install Diffusers from Source (Required!)
pip install git+https://github.com/huggingface/diffusers

# 3. Install other requirements
pip install transformers accelerate sentencepiece protobuf gradio psutil huggingface_hub
```

#### For older GPUs (RTX 30/40 series):

Standard installation is usually sufficient:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/huggingface/diffusers
pip install transformers accelerate sentencepiece protobuf gradio psutil huggingface_hub
```

---

## 🚀 Usage

1. **Activate your environment:**

```bash
.\venv\Scripts\activate
```

2. **Run the app:**

```bash
python app.py
```

3. **Open your browser** at `http://127.0.0.1:7860`.

> Alternatively, you can simply use the "start_z_image_lokal_gui.bat" file

> **Note:** On the first run, the model (~6GB) will be downloaded automatically from Hugging Face.

---

## 🔗 Credits

- **Model**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- **Library**: [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

---

## 🤝 Support

This is a free open-source project. I don't ask for donations.
However, if you want to say "Thanks", check out my profile on **[Spotify](https://open.spotify.com/artist/7EdK2cuIo7xTAacutHs9gv?si=4AqQE6GcQpKJFeVk6gJ06g)**.
A follow or a listen is the best way to support me! 🎧


