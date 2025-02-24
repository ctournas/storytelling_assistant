import os
import gradio as gr
import torch
import nltk
from openai import OpenAI
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from ultralytics import YOLO
from gtts import gTTS
from PIL import Image
import numpy as np
from nltk.tokenize import sent_tokenize
from IPython.display import Audio

# Βεβαιωθείτε ότι το API Key υπάρχει
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("⚠️ OpenAI API Key is missing! Add it as a Secret in Hugging Face Spaces.")

# OpenAI Client
client = OpenAI(api_key=api_key)

# Φόρτωση μοντέλων
yolo_model = YOLO("yolov8s.pt")
stable_diffusion = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
nltk.download("punkt")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def detect_objects(image_path):
    results = yolo_model(image_path)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]
            detected_objects.append(label)
    return detected_objects

def generate_story(detected_objects):
    story_prompt = f"Write a short story based on the following objects: {', '.join(detected_objects)}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": story_prompt}],
        max_tokens=200
    )
    return response.choices[0].message.content

def summarize_story(story):
    summary = summarizer(story, max_length=100, do_sample=False)[0]['summary_text']
    scenes = sent_tokenize(summary)
    return scenes

def generate_images(story):
    scenes = summarize_story(story)
    prompts = [f"Highly detailed, cinematic scene: {scene}, digital art, 4K, realistic lighting" for scene in scenes]
    images = []
    for prompt in prompts:
        image = stable_diffusion(prompt).images[0]
        images.append(image)
    return images

def text_to_speech(story):
    tts = gTTS(text=story, lang="en", slow=False)
    audio_file_path = "story_audio.mp3"
    tts.save(audio_file_path)
    return audio_file_path

def full_pipeline(image_path):
    detected_objects = detect_objects(image_path)
    story = generate_story(detected_objects)
    scenes = summarize_story(story)
    images = generate_images(scenes)
    audio = text_to_speech(story) 

demo = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Generated Story"),
        gr.Textbox(label="Story Scenes"),
        gr.Gallery(label="Generated Images"),
        gr.Audio(label="Story Audio"),
    ],
    title="AI-Powered Storytelling Assistant",
    description="Upload an image, and the AI will detect objects, generate a story, create images, and narrate the story."
)

if __name__ == "__main__":
    demo.launch()


