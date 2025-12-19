import google.generativeai as genai

API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
genai.configure(api_key=API_KEY)

try:
    print("Listing models...")
    for m in genai.list_models():
        print(m.name)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
