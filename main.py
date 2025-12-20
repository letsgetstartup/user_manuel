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
NANO_BANANA_PROMPT = """
Role & Goal: You are UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image. 
You must prioritize data accuracy and structural integrity over artistic interpretation.

Replicate the exact window title.
Exact Labels & Values (Row-by-Row): You MUST display the data exactly as written.
Icons - copy exactly where they appear in the source.

DO NOT use placeholder text.
DO NOT generalize numbers or IP addresses.
DO NOT add artistic lighting or textures.
Final Quality Check: The final image must be a high-resolution scan of the attached image.
"""

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

def main():
    # Session Initialization
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_context" not in st.session_state:
        st.session_state.pdf_context = None

    render_sidebar()

    st.title("Universal Guide AI 🍌")

    # Context Loader (Upload PDF)
    if not st.session_state.pdf_context:
        uploaded_file = st.file_uploader("Upload Manual (PDF)", type=["pdf"])
        if uploaded_file:
            st.session_state.pdf_context = uploaded_file.getvalue()
            st.success("Manual Loaded. Ready for questions.")
            st.rerun()
    
    # Chat Display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if isinstance(msg["content"], dict): # It's a tutorial object
                render_tutorial(msg["content"])
            else:
                st.write(msg["content"])

    # User Input
    if prompt := st.chat_input("How do I..."):
        # 1. Append User Msg
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Save to History DB
        db.collection("chats").document(st.session_state.session_id).set({
            "timestamp": datetime.now(),
            "messages": st.session_state.messages
        })
        st.rerun()

    # AI Processing Logic (Triggered on Rerun)
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                
                # A. ROUTER PHASE
                status.write("Analyzing request...")
                
                # Fetch existing slugs to encourage cache hits
                existing_slugs = []
                try:
                    # Get all document IDs from the tutorials collection
                    docs = db.collection("tutorials").stream()
                    existing_slugs = [doc.id for doc in docs]
                    status.write(f"Checking against {len(existing_slugs)} existing guides...")
                except Exception as e:
                    # CRITICAL: Do not silent fail. Report this.
                    st.error(f"Failed to fetch existing topics: {e}")
                    status.write("⚠️ DB Read Failed - Cache disabled.")

                router_model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=ROUTER_SYSTEM_INSTRUCTION)
                
                # Context-Aware Prompting (Strict Mode)
                router_prompt = f"""
                ROLE: Library Archivist.
                GOAL: Match the user's query to an existing TOPIC in the DATABASE.

                DATABASE OF EXISTING TOPICS:
                {json.dumps(existing_slugs)}

                USER QUERY: "{last_msg}"

                INSTRUCTIONS:
                1. SEARCH the DATABASE for a topic that matches the user's intent.
                2. DECISION:
                   - IF a match is found: RETURN the EXACT topic_slug from the database.
                   - IF NO match is found: Generate a NEW, concise, snake_case slug.

                Rules:
                - Do NOT generate a new slug if a semantically similar one exists.
                - Output JSON ONLY.
                """
                
                # Deterministic Output
                router_res = router_model.generate_content(
                    router_prompt,
                    generation_config=genai.types.GenerationConfig(temperature=0.0)
                )
                
                try:
                    # Clean JSON
                    clean_json = router_res.text.replace("```json", "").replace("```", "").strip()
                    router_data = json.loads(clean_json)
                    
                    if router_data.get("needs_clarification"):
                         # Ask for details
                        resp = router_data.get("clarification_question", "Can you clarify?")
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                        st.rerun()
                    
                    slug = router_data["topic_slug"]
                    status.write(f"Topic Identified: **{slug}**")

                    # B. CACHE CHECK PHASE
                    existing_tutorial = get_tutorial_from_db(slug)
                    
                    if existing_tutorial:
                        status.write("Found guide in database!")
                        time.sleep(0.5)
                        final_response = existing_tutorial
                    
                    else:
                        # C. GENERATION PHASE (Cache Miss)
                        status.write("Guide not found. Generating from PDF...")
                        
                        if not st.session_state.pdf_context:
                            st.error("No PDF uploaded! Cannot generate guide.")
                            st.stop()

                        # Generate Text Structure
                        gen_model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=GENERATOR_SYSTEM_INSTRUCTION)
                        # We pass a text prompt here. In production, consider using File API for large PDFs.
                        prompt_context = f"User Request: {last_msg}. Topic Slug: {slug}. Manual is loaded in context."
                        
                        # Note: For this snippet, we assume Gemini can answer based on general knowledge 
                        # OR you must attach the PDF parts to the prompt history. 
                        # For Sprint 1 simplicity, we are simulating the text generation.
                        gen_res = gen_model.generate_content(prompt_context)
                        tutorial_data = json.loads(gen_res.text.replace("```json", "").replace("```", "").strip())
                        
                        # Image Pipeline
                        status.write("Processing visuals (Nano Banana)...")
                        processed_steps = []
                        for step in tutorial_data["steps"]:
                            if step.get("has_visual"):
                                raw_img = extract_image_from_pdf(st.session_state.pdf_context, step["pdf_page_reference"])
                                if raw_img:
                                    # Nano Banana Logic
                                    processed_img = mock_nano_banana_api(raw_img, NANO_BANANA_PROMPT)
                                    # Upload
                                    fname = f"{slug}_step_{step['step_number']}_{uuid.uuid4().hex[:6]}.png"
                                    url = upload_image_to_storage(processed_img, fname)
                                    step["image_url"] = url
                            processed_steps.append(step)
                        
                        tutorial_data["steps"] = processed_steps
                        
                        # Save to Cache
                        save_tutorial_to_db(slug, tutorial_data)
                        final_response = tutorial_data

                    status.update(label="Complete", state="complete", expanded=False)
                    
                    # D. RENDER & SAVE
                    if isinstance(final_response, dict):
                        render_tutorial(final_response)
                    else:
                        st.write(final_response)
                        
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    # Update DB
                    db.collection("chats").document(st.session_state.session_id).set({
                        "timestamp": datetime.now(),
                        "messages": st.session_state.messages
                    })

                except Exception as e:
                    st.error(f"Processing Error: {e}")

if __name__ == "__main__":
    main()
