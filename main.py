"""
PROJECT DOCUMENTATION: Universal Guide - Sprint 1 (Caching Layer)

Overview:
This application serves as an AI Technical Support Agent. It utilizes a "Router-Generator" architecture
backed by Firebase Firestore (Database) and Firebase Storage (Assets).

Architecture Flow:
1. User Input -> Router (Gemini) -> Identifies 'Topic Slug'.
2. DB Check -> Firestore collection 'tutorials'.
   - HIT: Retrieve JSON -> Render UI.
   - MISS: Generator (Gemini) -> Parse PDF -> Extract Images -> 
           Nano Banana Processing -> Upload to Storage -> Save to Firestore -> Render UI.

Dependencies:
- streamlit, firebase_admin, google-generativeai, pymupdf (fitz), Pillow

Configuration:
- Firebase Credentials must be set in st.secrets["firebase"].
- Gemini API Key must be set in st.secrets["GEMINI_API_KEY"].
"""

import streamlit as st
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
import io
import fitz  # PyMuPDF
import json
import time
import uuid
import requests
import os
from datetime import datetime

# --- 1. CONFIGURATION & SINGLETON SETUP ---

st.set_page_config(page_title="Universal Guide AI", layout="wide")

# Initialize Gemini
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    st.error("CRITICAL: GEMINI_API_KEY missing in secrets.")

# Initialize Firebase (Singleton to prevent re-initialization errors)
if not firebase_admin._apps:
    try:
        # Load credentials from Streamlit secrets
        cred_dict = dict(st.secrets["firebase"])
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred, {
            'storageBucket': st.secrets.get("FIREBASE_BUCKET_NAME") 
        })
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

db = firestore.client()
bucket = storage.bucket()

# --- 2. PROMPT ENGINEERING ---

# The "Nano Banana" Prompt for Visual consistency
NANO_BANANA_PROMPT = """Create a copy of the attached image based on the following prompt:

Role & Goal: You are UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image. You must prioritize data accuracy and structural integrity over artistic interpretation.

Replicate the exact window title

Exact Labels & Values (Row-by-Row): You MUST display the data exactly as written, without any changes or generalizations 

Icons - copy exactly where they appear in the source.

VISUAL INDICATOR: If a specific instruction is provided, add a clearly visible "pointing finger" emoji or graphic (👉) pointing EXACTLY at the UI element or data point mentioned in the instruction. The finger should be placed so it doesn't obscure the text but clearly directs the user's eye to the relevant area.

DO NOT use placeholder text or "lorem ipsum".
DO NOT generalize numbers or IP addresses.
DO NOT add artistic lighting, reflections, or textures.
DO NOT simplify the diagram; if there are 8 rows in the source, there must be exactly 8 rows in the output.

Final Quality Check: The final image must be a high-resolution scan of the attached image Every number, dot, icon and line must match the provided specification."""

ROUTER_SYSTEM_INSTRUCTION = """
You are an intelligent dispatcher for a technical support system.
Analyze the user's request and categorize it into a unique, English "topic_slug" (snake_case).
Input: User query (text or image description).
Output: JSON ONLY.

JSON Schema:
{
    "topic_slug": "unique_identifier_string", 
    "needs_clarification": boolean,
    "clarification_question": "string or null"
}
Example: "My screen is black" -> {"topic_slug": "black_screen_troubleshoot", "needs_clarification": false, "clarification_question": null}
"""

GENERATOR_SYSTEM_INSTRUCTION = """
You are an expert Technical Support Guide. 
Your task: Create a step-by-step tutorial based ONLY on the provided PDF context.

Output: JSON ONLY.
JSON Schema:
{
  "title": "Clear English Title",
  "topic_slug": "matches_input_slug",
  "intro": "Brief introduction...",
  "language": "en", 
  "steps": [
    {
      "step_number": 1,
      "instruction": "Actionable instruction...",
      "pdf_page_reference": 10, 
      "has_visual": true 
    }
  ]
}

Rules:
1. If a step corresponds to a diagram/screenshot in the PDF, set "has_visual": true and provide the precise "pdf_page_reference" (integer).
2. Keep instructions concise and professional.
"""

# --- 3. CORE LOGIC (BACKEND) ---

def extract_image_from_pdf(pdf_bytes, page_number):
    """Extracts a specific page from the PDF as an image."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # Adjust for 0-based index (PDF p.1 is index 0)
        # We assume the LLM provides the printed page number (1-based)
        page_index = page_number - 1
        if 0 <= page_index < len(doc):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=150) # High quality extraction
            img_data = pix.tobytes("png")
            return Image.open(io.BytesIO(img_data))
    except Exception as e:
        print(f"Extraction Error on Page {page_number}: {e}")
    return None

def mock_nano_banana_api(source_image, prompt):
    """
    SIMULATION: In a real environment, this sends the image to an Image-Gen Model (Imagen 3 / DALL-E 3).
    For now, it returns the source image to ensure the pipeline works.
    """
    time.sleep(1) # Simulate processing
    return source_image # Returns PIL Object

def upload_image_to_storage(image_obj, filename):
    """Uploads PIL image to Firebase Storage and returns the public URL."""
    try:
        blob = bucket.blob(f"generated_assets/{filename}")
        img_byte_arr = io.BytesIO()
        image_obj.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        blob.upload_from_string(img_byte_arr, content_type='image/png')
        blob.make_public() 
        return blob.public_url
    except Exception as e:
        st.error(f"Storage Upload Error: {e}")
        return None

def get_tutorial_from_db(slug):
    """Cache Hit: Retrieve from Firestore."""
    try:
        doc = db.collection("tutorials").document(slug).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        st.error(f"DB Read Error: {e}")
    return None

def save_tutorial_to_db(slug, data):
    """Save new tutorial to Firestore."""
    try:
        db.collection("tutorials").document(slug).set(data)
    except Exception as e:
        st.error(f"DB Write Error: {e}")

# --- 4. UI COMPONENTS ---

# --- 4. UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.header("🗄️ History")
        
        # --- DB STATUS DEBUG ---
        try:
            tutorials_ref = db.collection("tutorials")
            count = len(list(tutorials_ref.stream()))
            st.success(f"🟢 Database Connected\nCached Guides: {count}")
        except Exception as e:
            st.error(f"🔴 DB Connection Error: {e}")
        # -----------------------

        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        # Fetch last 10 chats
        try:
            docs = db.collection("chats").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
            for doc in docs:
                data = doc.to_dict()
                # Derive title
                title = "Conversation"
                if data.get("messages"):
                     # Simple logic: first user message is title
                     for m in data["messages"]:
                         if m["role"] == "user":
                             title = m["content"][:20] + "..."
                             break
                
                if st.button(f"📄 {title}", key=doc.id):
                    st.session_state.session_id = doc.id
                    st.session_state.messages = data.get("messages", [])
                    st.rerun()
        except Exception:
            pass

def render_tutorial(data):
    """Renders the step-by-step guide."""
    with st.container(border=True):
        st.subheader(f"📚 {data.get('title')}")
        st.caption(data.get('intro'))
        
        for step in data.get("steps", []):
            st.markdown(f"**Step {step['step_number']}**")
            st.write(step['instruction'])
            
            if step.get("image_url"):
                st.image(step["image_url"], caption="Visual Guide", width=400)
            elif step.get("has_visual"):
                st.info(f"Refer to manual page {step['pdf_page_reference']}")
            st.divider()

# --- 5. MAIN APP LOOP ---

from app.search_manager import GraphSearcher

# --- 5. MAIN APP LOOP ---

def main():
    # Session Initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "machine_id" not in st.session_state:
        st.session_state.machine_id = None

    render_sidebar()

    st.title("Enterprise Manual AI 🚀")
    st.markdown("Automated Troubleshooting & Technical Support")

    # Selection of Machine (Manual)
    try:
        machines_ref = db.collection("machines").stream()
        machine_list = [{"id": m.id, "name": m.to_dict().get("name", m.id)} for m in machines_ref]
        
        if machine_list:
            machine_names = [m["name"] for m in machine_list]
            selected_name = st.selectbox("Select Manual", machine_names)
            selected_id = next(m["id"] for m in machine_list if m["name"] == selected_name)
            st.session_state.machine_id = selected_id
        else:
            st.warning("No manuals processed yet. Please upload a PDF to the `manualai02a` bucket to start.")
    except Exception as e:
        st.error(f"Error fetching manuals: {e}")

    # Chat Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                if "tutorial" in msg:
                    render_tutorial(msg["tutorial"])
                elif "data" in msg:
                    render_search_result(msg["data"])
                else:
                    st.write(msg["content"])
            else:
                st.write(msg["content"])

    # User Input
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                try:
                    # Prepare history for backend
                    history = []
                    for m in st.session_state.messages[:-1]: # Exclude current prompt
                        history.append({"role": m["role"], "content": m["content"]})
                    
                    # Call Unified Backend API
                    # The backend URL is usually the same host during local dev or a specific Firebase Function URL
                    # For local testing, we might need a dynamic URL or a fixed one.
                    # Assuming local development for now, or using a proxy if needed.
                    backend_url = "http://localhost:5001/manualai-481406/us-central1/api/api/analyze" 
                    # Note: The double 'api' in path is common with firebase_functions + flask
                    
                    # Fallback for local testing if not running firebase emulators
                    if os.environ.get("USE_LOCAL_FLASK"):
                        backend_url = "http://localhost:5000/api/analyze"

                    payload = {
                        'message': prompt,
                        'machine_id': st.session_state.machine_id,
                        'history': json.dumps(history)
                    }
                    
                    response = requests.post(backend_url, data=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        status.update(label="Response received!", state="complete")
                        
                        if 'solution' in data:
                            st.markdown(data['solution'])
                            st.session_state.messages.append({"role": "assistant", "content": data['solution']})
                        elif 'tutorial' in data:
                            render_tutorial(data['tutorial'])
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "Here is your step-by-step tutorial.",
                                "tutorial": data['tutorial']
                            })
                        else:
                            st.error("Unexpected response format from backend.")
                    else:
                        st.error(f"Backend Error: {response.text}")
                        status.update(label="Request failed.", state="error")
                except Exception as e:
                    st.error(f"Communication Error: {e}")
                    status.update(label="Error.", state="error")

        # Save to History
        try:
            db.collection("chats").document(st.session_state.session_id).set({
                "timestamp": datetime.now(),
                "messages": st.session_state.messages,
                "machine_id": st.session_state.machine_id
            })
        except Exception as e:
            print(f"Error saving chat: {e}")

def render_search_result(node):
    """Renders a node from the Knowledge Graph."""
    st.success(f"### 🎯 Step: {node.get('label')}")
    st.write(node.get('text'))
    
    if node.get('ar_anchor'):
        st.info(f"**Physical Reference (AR Anchor):** {node['ar_anchor']}")
    
    steps = node.get('next_possible_steps', [])
    if steps:
        st.markdown("#### ➡️ Next Logical Steps")
        for step in steps:
            with st.expander(f"Condition: {step.get('condition', 'Next')}"):
                st.write(f"Following this leads to: **{step.get('to')}**")

if __name__ == "__main__":
    main()

