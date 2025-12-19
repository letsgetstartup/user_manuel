from google.cloud import aiplatform
import google.auth
import os

API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
PROJECT_ID = "manualai-481406"
LOCATION = "me-west1" 

print(f"Testing Vertex AI SDK with Key (alternative import). Project: {PROJECT_ID}, Location: {LOCATION}")

try:
    # Try using aiplatform.init with api_key if supported
    # Note: older versions might not support api_key in init directly like 'vertexai.init'
    # But 1.38+ should.
    
    # If explicit 'vertexai' module is missing, we try this:
    aiplatform.init(project=PROJECT_ID, location=LOCATION, api_key=API_KEY)
    
    # Check if we can use GenerativeModel from generic place?
    # Usually it's vertexai.generative_models
    # Or google.cloud.aiplatform.generative_models? (Unlikely)
    
    # Let's try to import vertexai again inside here, maybe it was a path fluke?
    import vertexai
    from vertexai.generative_models import GenerativeModel
    
    model = GenerativeModel("gemini-1.5-flash")
    print("Generating content...")
    response = model.generate_content("Hello?")
    print(response.text)

except Exception as e:
    print(f"Error: {e}")
    # Fallback: check dir
    import google.cloud
    print("dir(google.cloud):", dir(google.cloud))
