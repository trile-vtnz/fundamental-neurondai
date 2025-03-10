import gradio as gr
import cv2
import numpy as np

# Define function to process images
def edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Create Gradio interface
iface = gr.Interface(
    fn=edge_detection,  # Processing function
    inputs=gr.Image(type="numpy"),  # Image input
    outputs=gr.Image(type="numpy"),  # Processed image output
    title="Edge Detection App",
    description="Upload an image and apply an edge detection filter using OpenCV."
)

# Launch the app
iface.launch()
