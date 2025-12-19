import vertexai
from vertexai.generative_models import GenerativeModel
import os

API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
PROJECT_ID = "manualai-481406"
LOCATION = "me-west1" 

print(f"Testing Vertex AI SDK with Key. Project: {PROJECT_ID}, Location: {LOCATION}")

try:
    # Initialize Vertex AI with API Key
    vertexai.init(project=PROJECT_ID, location=LOCATION, api_key=API_KEY)
    
    print("Vertex AI initialized. Loading model...")
    model = GenerativeModel("gemini-1.5-flash")
    
    print("Generating content...")
    response = model.generate_content("Hello, can you hear me?")
    
    print("Response received:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")
