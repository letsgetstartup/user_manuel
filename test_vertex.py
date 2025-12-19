from google import genai
import os

# Details extracted from screenshots
API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
PROJECT_ID = "manualai-481406"
LOCATION = "me-west1" 

print(f"Testing key for Project: {PROJECT_ID}, Location: {LOCATION}")

try:
    # Try initializing for Vertex AI
    # Note: 'vertexai=True' usually enables Vertex mode.
    # We pass api_key explicitly.
    client = genai.Client(
        vertexai=True, 
        project=PROJECT_ID, 
        location=LOCATION, 
        api_key=API_KEY
    )
    
    print("Client initialized. Generating content...")
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents="Hello, are you working?"
    )
    print("Response received:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")
