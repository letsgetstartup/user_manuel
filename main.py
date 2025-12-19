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
import os
from datetime import datetime

# --- CONFIGURATION & SETUP ---

# --- CONFIGURATION & SETUP ---

# 1. Setup Page
st.set_page_config(page_title="Nano Banana Support", layout="wide")

# 2. Load API Key
if "GEMINI_API_KEY" in st.secrets:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
else:
    API_KEY = os.environ.get("GEMINI_API_KEY")
    if API_KEY:
        genai.configure(api_key=API_KEY)
    else:
        st.error("Missing GEMINI_API_KEY in st.secrets or environment variables")

# 3. Initialize Firebase (Singleton Pattern)
if not firebase_admin._apps:
    try:
        if "firebase" in st.secrets:
            cred_dict = dict(st.secrets["firebase"])
            if "project_id" not in cred_dict and "firebase" in cred_dict:
                 cred_dict = cred_dict["firebase"]
            cred = credentials.Certificate(cred_dict)
            bucket_name = st.secrets.get("FIREBASE_BUCKET_NAME")
            firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
        else:
            st.error("Missing firebase credentials in st.secrets")
    except Exception as e:
        st.error(f"Failed to initialize Firebase: {e}")

try:
    db = firestore.client()
    bucket = storage.bucket()
except Exception as e:
    st.error(f"Firebase clients failed to initialize: {e}")

# --- CONSTANTS & PROMPTS ---

NANO_BANANA_PROMPT = """
Create a copy of the attached image based on the following prompt:

Role & Goal: You are UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image. You must prioritize data accuracy and structural integrity over artistic interpretation.

Replicate the exact window title
Exact Labels & Values (Row-by-Row): You MUST display the data exactly as written, without any changes or generalizations 
Icons - copy exactly where they appear in the source.

DO NOT use placeholder text or "lorem ipsum".
DO NOT generalize numbers or IP addresses.
DO NOT add artistic lighting, reflections, or textures.
DO NOT simplify the diagram; if there are 8 rows in the source, there must be exactly 8 rows in the output.
Final Quality Check: The final image must be a high-resolution scan of the attached image Every number, dot, icon and line must match the provided specification.
"""

ROUTER_SYSTEM_INSTRUCTION = """
You are an intelligent technical support dispatcher. Your goal is to understand the user's issue and decide the next step.
ALWAYS interact in English.

You MUST decide between these actions:
1. **Clarify**: The user's request is vague or missing details. Ask a helpful, conversational question.
2. **Identify**: You fully understand the task. State clearly what you understand the issue to be and identify a `topic_slug`.
3. **Chat**: General conversation or questions not requiring a technical tutorial.

Output ONLY JSON.
JSON Format:
{
    "action": "clarify" | "identify" | "chat",
    "topic_slug": "unique_snake_case_identifier" | null,
    "understanding": "Clear sentence describing what you think the user wants to do",
    "message": "The message to send to the user. If action is 'identify', ask: 'I understand you want to [understanding]. Should I generate the step-by-step tutorial from the manual?'"
}
"""

GENERATOR_SYSTEM_INSTRUCTION = """
You are an expert Technical Support Guide. Create a structured JSON tutorial based on the PDF manuals.
Output ONLY JSON.
ALWAYS output text in English.

JSON Format:
{
  "title": "Human Readable Title",
  "topic_slug": "same_slug_as_input",
  "intro": "Brief explanation...",
  "steps": [
    {
      "step_number": 1,
      "instruction": "Detailed text...",
      "pdf_page_reference": 10,
      "has_visual": true
    }
  ]
}
"""

# --- HELPERS ---

def upload_to_gemini(data, display_name, mime_type):
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf" if "pdf" in mime_type else ".png") as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        g_file = genai.upload_file(path=tmp_path, display_name=display_name)
        if "pdf" in mime_type:
            while g_file.state.name == "PROCESSING":
                time.sleep(2)
                g_file = genai.get_file(g_file.name)
        return g_file
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def save_chat_to_firestore(session_id, messages):
    try:
        doc_ref = db.collection("chats").document(session_id)
        serializable_messages = []
        for msg in messages:
            serializable_messages.append({"role": msg["role"], "content": msg["content"]})
        doc_ref.set({"timestamp": datetime.now(), "messages": serializable_messages})
    except Exception as e:
        st.error(f"Failed to save chat: {e}")

def load_chat_history_list():
    chats = []
    try:
        docs = db.collection("chats").order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
        for doc in docs:
            data = doc.to_dict()
            title = "New Chat"
            for msg in data.get("messages", []):
                if msg["role"] == "user":
                    content = msg["content"]
                    if isinstance(content, str):
                        title = content[:25] + "..."
                    break
            chats.append({"id": doc.id, "title": title})
    except Exception as e:
        st.sidebar.error(f"DB Error: {e}")
    return chats

def load_specific_chat(session_id):
    try:
        doc = db.collection("chats").document(session_id).get()
        if doc.exists:
            return doc.to_dict().get("messages", [])
    except Exception as e:
        st.error(f"Error loading chat: {e}")
    return []

def get_tutorial_from_firestore(slug):
    try:
        doc = db.collection("tutorials").document(slug).get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        st.error(f"Error getting tutorial: {e}")
    return None

def save_tutorial_to_firestore(slug, data):
    try:
        db.collection("tutorials").document(slug).set(data)
    except Exception as e:
        st.error(f"Error saving tutorial: {e}")

def upload_image_to_storage(image_obj, filename):
    try:
        blob = bucket.blob(f"generated_assets/{filename}")
        img_byte_arr = io.BytesIO()
        image_obj.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        blob.upload_from_string(img_byte_arr, content_type='image/png')
        blob.make_public() 
        return blob.public_url
    except Exception as e:
        st.error(f"Upload error: {e}")
        return None

def mock_nano_banana_api(source_image, prompt):
    """
    Simulate the Nano Banana UI Engineer API.
    In reality, you'd send source_image + prompt to your DALL-E/Stable Diffusion endpoint.
    """
    time.sleep(1.5) 
    # Placeholder: convert to grayscale to simulate technical replication
    return source_image.convert("L")

def extract_image_from_pdf(pdf_bytes, page_number):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if page_number < 1 or page_number > len(doc):
             return None
        page = doc.load_page(page_number - 1)
        
        # New extraction logic: find embedded images
        image_list = page.get_images(full=True)
        
        largest_area = 0
        best_rect = None
        
        for img in image_list:
            xref = img[0]
            # get_image_rects returns a list of rects where this image appears
            rects = page.get_image_rects(xref)
            for rect in rects:
                # Filter out small icons/decorations (e.g. < 100x100)
                if rect.width < 100 or rect.height < 100:
                    continue
                
                area = rect.width * rect.height
                if area > largest_area:
                    largest_area = area
                    best_rect = rect
        
        if best_rect:
            # Crop to the largest image
            # We use the page pixmap clipped to the rect to preserve any text/overlays ON TOP of the image
            pix = page.get_pixmap(clip=best_rect, dpi=200)
        else:
            # Fallback: full page if no suitable images found
            pix = page.get_pixmap(dpi=200)

        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return None

# --- UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.header("🗄️ History")
        if st.button("➕ New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.awaiting_confirmation = None
            st.rerun()
        st.divider()
        chats = load_chat_history_list()
        for chat in chats:
            if st.button(f"📄 {chat['title']}", key=f"chat_{chat['id']}", use_container_width=True):
                st.session_state.session_id = chat['id']
                st.session_state.messages = load_specific_chat(chat['id'])
                st.session_state.awaiting_confirmation = None
                st.rerun()

def render_tutorial_card(tutorial_data):
    with st.container(border=True):
        st.subheader(f"📚 {tutorial_data.get('title', 'Guide')}")
        intro = tutorial_data.get('intro', '')
        if intro:
            st.info(intro)
        for step in tutorial_data.get("steps", []):
            st.markdown(f"#### Step {step['step_number']}")
            st.write(step['instruction'])
            if step.get("image_url"):
                st.image(step["image_url"], caption="Nano Banana Generated Replica", width=600)
            elif step.get("has_visual"):
                 st.caption("*(Visual extraction from page " + str(step.get('pdf_page_reference', '??')) + ")*")
        st.success("Tutorial Complete.")

# --- MAIN APP ---

def main():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_context" not in st.session_state:
        st.session_state.pdf_context = None
    if "pdf_g_file" not in st.session_state:
        st.session_state.pdf_g_file = None
    if "awaiting_confirmation" not in st.session_state:
        st.session_state.awaiting_confirmation = None

    render_sidebar()

    st.title("🍌 Nano Banana Technical Support")

    # Knowledge Base
    with st.expander("🛠️ Knowledge Base Setup", expanded=not st.session_state.pdf_context):
        if not st.session_state.pdf_context:
            uploaded = st.file_uploader("Upload Manual (PDF)", type=["pdf"])
            if uploaded:
                st.session_state.pdf_context = uploaded.getvalue()
                with st.spinner("Indexing manual for AI..."):
                    st.session_state.pdf_g_file = upload_to_gemini(st.session_state.pdf_context, uploaded.name, "application/pdf")
                st.success("Knowledge base ready.")
                st.rerun()
        else:
            if st.button("🗑️ Clear Manual"):
                st.session_state.pdf_context = None
                st.session_state.pdf_g_file = None
                st.rerun()

    # Issue Photo Upload
    user_image_data = None
    with st.sidebar:
        st.divider()
        st.header("📸 Current Issue")
        u_img = st.file_uploader("Show me the problem", type=["png", "jpg", "jpeg"])
        if u_img:
            user_image_data = u_img.getvalue()
            st.image(user_image_data)

    # Chat Display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                content = msg["content"]
                if isinstance(content, dict) and "steps" in content:
                    render_tutorial_card(content)
                else:
                    st.write(content)

    # Input User
    if prompt := st.chat_input("Describe your issue..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_chat_to_firestore(st.session_state.session_id, st.session_state.messages)
        st.rerun()

    # Assistant Process
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        last_msg = st.session_state.messages[-1]["content"]
        
        with st.chat_message("assistant"):
            with st.status("Analyzing Workflow...", expanded=True) as status:
                
                # Check for "Yes/Go ahead" for confirmation
                if st.session_state.awaiting_confirmation:
                    intent_model = genai.GenerativeModel("gemini-1.5-flash")
                    # Quick check if the user said 'yes' or something positive
                    check = intent_model.generate_content(f"Does the following message confirm they want to proceed (yes/no)? Message: '{last_msg}'")
                    
                    if "yes" in check.text.lower():
                        # GENERATION PHASE
                        topic_slug = st.session_state.awaiting_confirmation
                        st.session_state.awaiting_confirmation = None # Reset
                        
                        # 1. Check Cache first
                        cached = get_tutorial_from_firestore(topic_slug)
                        if cached:
                            status.write("Loading guide from database...")
                            final_resp = cached
                        else:
                            status.write("Generating Step-by-Step tutorial from manual...")
                            gen_model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=GENERATOR_SYSTEM_INSTRUCTION)
                            
                            c_parts = []
                            if st.session_state.pdf_g_file: c_parts.append(st.session_state.pdf_g_file)
                            
                            # Re-run or get task from history
                            gen_res = gen_model.generate_content(c_parts + [f"Task: Generate a guide for {topic_slug}. Reference the manual for icons and layout."], generation_config={"response_mime_type": "application/json"})
                            tutorial_json = json.loads(gen_res.text)
                            
                            # 2. Image Pipeline
                            status.write("Replicating technical visuals (Nano Banana UI Engine)...")
                            processed_steps = []
                            for step in tutorial_json.get("steps", []):
                                step_data = step.copy()
                                if step.get("has_visual") and st.session_state.pdf_context:
                                    pdf_page = step.get("pdf_page_reference")
                                    if pdf_page:
                                        status.write(f"Extracting image from page {pdf_page}...")
                                        raw_img = extract_image_from_pdf(st.session_state.pdf_context, pdf_page)
                                        if raw_img:
                                            # Call Nano Banana
                                            gen_img = mock_nano_banana_api(raw_img, NANO_BANANA_PROMPT)
                                            # Upload
                                            u_id = uuid.uuid4().hex[:8]
                                            fname = f"{topic_slug}_step_{step['step_number']}_{u_id}.png"
                                            url = upload_image_to_storage(gen_img, fname)
                                            step_data["image_url"] = url
                                processed_steps.append(step_data)
                            
                            tutorial_json["steps"] = processed_steps
                            save_tutorial_to_firestore(topic_slug, tutorial_json)
                            final_resp = tutorial_json

                        render_tutorial_card(final_resp)
                        st.session_state.messages.append({"role": "assistant", "content": final_resp})
                        save_chat_to_firestore(st.session_state.session_id, st.session_state.messages)
                        status.update(label="Guide Created", state="complete")
                        st.stop()
                    else:
                        # User said no or something else, resume identify
                        st.session_state.awaiting_confirmation = None

                # ROUTING PHASE
                status.write("Checking manual context...")
                c_parts = []
                if st.session_state.pdf_g_file: c_parts.append(st.session_state.pdf_g_file)
                if user_image_data:
                    u_img_f = upload_to_gemini(user_image_data, "user_issue.png", "image/png")
                    c_parts.append(u_img_f)
                
                router = genai.GenerativeModel("gemini-1.5-flash", system_instruction=ROUTER_SYSTEM_INSTRUCTION)
                router_res = router.generate_content(c_parts + [f"User input history: {st.session_state.messages}"], generation_config={"response_mime_type": "application/json"})
                
                try:
                    r_data = json.loads(router_res.text)
                    
                    if r_data["action"] == "identify":
                        st.session_state.awaiting_confirmation = r_data["topic_slug"]
                        
                    st.session_state.messages.append({"role": "assistant", "content": r_data["message"]})
                    save_chat_to_firestore(st.session_state.session_id, st.session_state.messages)
                    st.write(r_data["message"])
                    status.update(label="Response generated", state="complete")

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": "I'm having trouble understanding. Please describe the task again."})
                    save_chat_to_firestore(st.session_state.session_id, st.session_state.messages)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
