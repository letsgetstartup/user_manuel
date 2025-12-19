from google import genai
import os

API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
PROJECT_ID = "manualai-481406"
LOCATION = "me-west1" 

print(f"Testing key for Project: {PROJECT_ID}, Location: {LOCATION} with custom api_base")

try:
    # Explicitly set the api_base to the Vertex AI Regional Endpoint
    client = genai.Client(
        api_key=API_KEY,
        http_options={
            'api_version': 'v1beta1',
            'base_url': f'https://{LOCATION}-aiplatform.googleapis.com' 
        }
    )
    
    # Vertex AI "Express" / API Key access often uses the same path structure as AI Studio?
    # Or does it use the Vertex path?
    # If we point base_url to Vertex, we probably need to use the Vertex model format.
    
    print("Generating content...")
    response = client.models.generate_content(
        model=f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-1.5-flash",
        contents="Hello, do you work?"
    )
    print("Response received:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")
