import google.generativeai as genai
from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np

# Load environment variables and configure Google Generative AI
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to get AI response
def get_genai_response(input_text, extracted_text, prompt):
    try:
        response = model.generate_content([input_text, extracted_text, prompt])
        # Check if the response has valid parts
        if hasattr(response, 'candidates') and response.candidates:
            # Assuming we want the first candidate's text
            return response.candidates[0].text if hasattr(response.candidates[0], 'text') else "No valid text found in the response."
        else:
            return "The AI could not generate a valid response."
    except Exception as e:
        st.error(f"Error with AI generation: {e}")
        return "There was an error processing your request."

# Function to extract text from the uploaded image using OCR
def extract_text_from_image(image):
    # Convert the PIL Image to an OpenCV image (BGR format)
    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(image_array)
    return extracted_text

# Streamlit configuration
st.set_page_config(page_title="BS Extractor")
st.header("Bank Statement Extractor")

# User input for the prompt
input_text = st.text_input("Input Prompt:", key="input")

# File uploader for bank statement image
uploaded_file = st.file_uploader("Choose a bank statement image", type=['jpg', 'png', 'jpeg'])
image = ""

# Display uploaded image
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Button for submission
button = st.button("Tell me about the image?")

# Prompt text
prompt = """
    You are an expert in understanding bank statements. We will upload an image of a bank statement,
    and you will have to answer any questions based on the uploaded bank statement.
"""

# Processing the request after clicking the button
if button and input_text:
    if uploaded_file is not None:
        # Extract text from the uploaded image
        extracted_text = extract_text_from_image(image)

        if extracted_text:  # Check if any text was extracted
            # Get AI-generated response
            response = get_genai_response(input_text, extracted_text, prompt)

            # Display the response
            st.subheader("The response is:")
            st.write(response)
        else:
            st.error("No text was extracted from the image.")
    else:
        st.error("Please upload a valid image file!")




