from google import genai
import os

# Details extracted from screenshots
API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
PROJECT_ID = "manualai-481406"
LOCATION = "me-west1" 

print(f"Testing key for Project: {PROJECT_ID}, Location: {LOCATION}")

try:
    # Try initializing with http_options to point to Vertex endpoint
    # The Vertex AI endpoint for Gemini is typically: 
    # https://{LOCATION}-aiplatform.googleapis.com
    # And the path often includes /v1beta1/projects/{project}/locations/{location}/publishers/google/models/...
    
    # However, 'google-genai' SDK might handle this if we just set the api_base?
    
    # Let's try to simulate what "Express" mode does.
    # If we use api_key, it defaults to 'generativelanguage.googleapis.com'.
    # We want to force it to 'me-west1-aiplatform.googleapis.com'.
    
    client = genai.Client(
        api_key=API_KEY,
        http_options={
            'api_version': 'v1beta1',
            # This might be tricky because the SDK constructs paths differently for Vertex vs AI Studio.
            # Vertex: POST /v1beta1/projects/{PROJECT}/locations/{LOCATION}/publishers/google/models/{MODEL}:generateContent
            # AI Studio: POST /v1beta/models/{MODEL}:generateContent
        }
    )
    
    # If the SDK strictly separates them, we might be stuck unless we find the right option.
    # Let's try to just use `vertexai=True` BUT set the Credentials manually? No, API key is not a credential object.
    
    # Alternate attempt: Use `google-generativeai` SDK (the OLD one) and configure the base_url?
    # Or just use `requests` to debug the exact endpoint?
    
    print("Attempting with default client but checking model...")
    # Maybe extracting the project/location into the model name works?
    # models/projects/manualai-481406/locations/me-west1/publishers/google/models/gemini-1.5-flash
    
    full_model_name = f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-1.5-flash"
    
    response = client.models.generate_content(
        model=full_model_name,
        contents="Hello, are you working?"
    )
    print("Response received:")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")
