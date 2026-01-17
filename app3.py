from fastai.vision.all import *
import gradio as gr
import torch

# Force CPU and add error handling
try:
    learn = load_learner('melanoma_classifier.pkl', cpu=True)
except Exception as e:
    print(f"Error loading model: {e}")
    # Try with pickle_module parameter
    import pickle
    learn = load_learner('melanoma_classifier.pkl', cpu=True, pickle_module=pickle)

categories = ['Benign', 'Melanoma']

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return {categories[i]: float(probs[i]) for i in range(len(categories))}

examples = ['benign.jpg', 'melanoma.jpg']

intf = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    examples=examples
)

intf.launch()