# AI-Powered Storytelling Assistant

## 📌 Project Overview
The **AI-Powered Storytelling Assistant** is an interactive AI application that combines multiple AI models to create immersive storytelling experiences. Users can upload an image, and the system will detect objects, generate a story, summarize it into key scenes, create AI-generated images for each scene, and convert the story into speech.

## ✨ Features
1. **Object Detection** (YOLOv8) - Identifies objects in an uploaded image.
2. **Story Generation** (GPT-4o-mini) - Creates a unique story based on the detected objects.
3. **Story Summarization** (BART-Large) - Extracts key scenes from the generated story.
4. **Image Generation** (Stable Diffusion) - Creates AI-generated images for each scene.
5. **Text-to-Speech (TTS)** (Google TTS) - Converts the final story into audio.

## 🛠️ Tech Stack
- **Backend**: Python, Gradio
- **Machine Learning Models**:
  - YOLOv8 (Object Detection)
  - OpenAI GPT-4o-mini (Story Generation)
  - BART-Large (Summarization)
  - Stable Diffusion (Image Generation)
  - Google TTS (Speech Synthesis)
- **Deployment**: Hugging Face Spaces

## 🚀 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Powered-Storytelling-Assistant.git
cd AI-Powered-Storytelling-Assistant
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Application
```bash
python app.py
```

### 4️⃣ Deploy on Hugging Face
- Save your OpenAI API key on Hugging Face Secrets as `OPENAI_API_KEY`
- Upload your files to Hugging Face Spaces and deploy.

## 🎮 How to Use
1. Upload an image.
2. The system detects objects and generates a story.
3. The story is summarized into key scenes.
4. AI-generated images are created for each scene.
5. The full story is converted into speech for an immersive experience.

## 📸 Example Output
- **Input**: Image with objects (e.g., a dog and a ball)
- **Generated Story**: "Once upon a time, a dog found a magical ball..."
- **AI Images**: AI-generated scenes of the story.
- **Audio Output**: A narrated version of the story.

## 📜 License
This project is open-source under the MIT License.

## 🤝 Contributing
Feel free to fork the repo, submit issues, or contribute via pull requests!

---
### 💡 Future Improvements
- Implement real-time voice input for interactive storytelling.
- Add more customization options for generated images.
- Improve text-to-speech with different voice styles.

Happy Coding! 🚀
