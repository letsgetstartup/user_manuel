#!./venv/bin/python
import os
import glob
import time
from google import genai
from google.genai import types
from pypdf import PdfReader

# Configuration
API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
MANUALS_DIR = "manuals"
IMAGES_DIR = "images"

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def main():
    print("Starting Error Solution App...")
    
    # Initialize Client (Defaulting to AI Studio/Generative Language API)
    # The provided API Key appears to be a Vertex AI Bound key. 
    # For this code to work, the "Generative Language API" must be enabled in the project 'manualai-481406'
    # and the API Key must allow access to it.
    client = genai.Client(api_key=API_KEY)
    
    # 1. Load Manuals Text
    print("Processing manuals...")
    manual_files = glob.glob(os.path.join(MANUALS_DIR, "*.pdf"))
    if not manual_files:
        print("No manuals found in", MANUALS_DIR)
        return

    manuals_text = ""
    for manual_path in manual_files:
        print(f"Extracting text from {manual_path}...")
        text = extract_text_from_pdf(manual_path)
        if text:
            manuals_text += f"\n\n--- Content from {os.path.basename(manual_path)} ---\n{text}"

    if not manuals_text:
        print("No text extracted from manuals.")

    # 2. Process Image
    image_files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGES_DIR, "*.png"))
    if not image_files:
        print("No images found in", IMAGES_DIR)
        return
    
    target_image = image_files[0]
    print(f"Analyzing image: {target_image}")
    
    with open(target_image, "rb") as f:
        image_bytes = f.read()

    # 3. Generate Solution
    print("Generating solution...")
    
    prompt = "Analyze this image which shows an error. Using the provided manuals context below, identify the error and provide a step-by-step solution to fix it.\n\nManuals Context:\n" + manuals_text[:1000000] 
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=prompt),
                        types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    ]
                )
            ]
        )
        
        print("\n" + "="*20 + " SOLUTION " + "="*20 + "\n")
        print(response.text)
        print("\n" + "="*50)

    except Exception as e:
        print(f"Error responding: {e}")
        print("\nTroubleshooting: If you see a 401 'API keys are not supported', please enable the 'Generative Language API' in your Google Cloud Project and ensure the API Key restrictions allow it.")

if __name__ == "__main__":
    main()