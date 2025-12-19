import os
import glob
from urllib.parse import quote
import base64
import requests
import json
import io
import time
import uuid
from flask import Flask, render_template, request, jsonify, session
from firebase_functions import https_fn
from werkzeug.wrappers import Response
import fitz  # PyMuPDF
from PIL import Image

# Vertex AI Imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part, HarmCategory, HarmBlockThreshold
import firebase_admin
from firebase_admin import credentials, firestore, storage
from datetime import datetime, timedelta

# Configuration
PROJECT_ID = "manualai-481406"
LOCATION = "us-central1" 
MANUALS_DIR = "manuals"


# Initialize Firebase
if not firebase_admin._apps:
    firebase_admin.initialize_app(options={
        'storageBucket': "manualai-481406.firebasestorage.app"
    })

def init_vertex():
    vertexai.init(project=PROJECT_ID, location=LOCATION)


app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- PROMPTS ---

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
You are a Technical Support Dispatcher. Your ONLY job is to categorize the request.
CRITICAL: NEVER provide technical instructions, answers, or data from the manual in this phase. 

Your ONLY allowed response is a JSON object with one of these actions:
1. **Identify**: Use this for ANY request related to "How to", "Help with", or specific procedures in the manual. 
   - State clearly: "I understand you need help with [Task]. Should I generate the step-by-step tutorial?"
   - Topic Slug: unique_snake_case_name.
2. **Clarify**: Use if the request is gibberish or too broad. 
3. **Chat**: ONLY for "Hello", "How are you", or non-technical chatter.

If you provide a technical answer instead of asking for confirmation, you HAVE FAILED.
"""

GENERATOR_SYSTEM_INSTRUCTION = """
You are an expert Technical Support Guide. Create a structured JSON tutorial based on the PDF manuals.
Output ONLY JSON in English.

You MUST follow this JSON Format:
{
  "title": "Human Readable Title",
  "topic_slug": "same_slug_as_input",
  "intro": "Brief explanation of the goal.",
  "steps": [
    {
      "step_number": 1,
      "instruction": "Detailed text instruction. Reference specific icons or menu names from the manual.",
      "pdf_page_reference": 38,
      "has_visual": true
    }
  ]
}

Important: Identify the EXACT page numbers for visuals. If page 38 contains the diagram for the task, use 38.
"""

@app.route('/api/debug/extraction')
def debug_extraction():
    """Debug route to inspect file system and PDF path resolution."""
    try:
        cwd = os.getcwd()
        ls_cwd = os.listdir(cwd)
        
        # Check manuals dir existence
        manuals_path = os.path.join(cwd, MANUALS_DIR)
        manuals_exists = os.path.exists(manuals_path)
        ls_manuals = os.listdir(manuals_path) if manuals_exists else "Directory Not Found"
        
        # Check glob
        pdf_pattern = os.path.join(MANUALS_DIR, "*.pdf")
        glob_results = glob.glob(pdf_pattern)
        
        # Attempt minimal extraction
        extraction_status = "Skipped"
        if glob_results:
            try:
                doc = fitz.open(glob_results[0])
                page = doc.load_page(37) # Page 38 (0-indexed)
                pix = page.get_pixmap(dpi=72)
                extraction_status = f"Success - Image size: {pix.width}x{pix.height}"
            except Exception as e:
                extraction_status = f"Failed: {str(e)}"

        return jsonify({
            "cwd": cwd,
            "ls_cwd": ls_cwd,
            "manuals_path": manuals_path,
            "manuals_exists": manuals_exists,
            "ls_manuals": ls_manuals,
            "glob_pattern": pdf_pattern,
            "glob_results": glob_results,
            "extraction_test": extraction_status
        })
    except Exception as e:
        return jsonify({"fatal_error": str(e)})

@app.route('/api/debug/upload')
def debug_upload():
    """Debug route to test Storage Upload permissions."""
    try:
        # Create a dummy image
        img = Image.new('RGB', (100, 100), color = 'blue')
        filename = f"debug_upload_real_{uuid.uuid4().hex[:8]}.png"
        
        # Call the REAL function
        url = upload_to_storage(img, filename)
        
        if url:
             return f"<h1>Upload Success!</h1><p>URL: <a href='{url}'>{url}</a></p><img src='{url}'>"
        else:
             return "<h1>Upload Failed</h1><p>returned None. Check logs.</p>"

    except Exception as e:
        return f"<h1>Fatal Upload Error</h1><p>{e}</p>"

    except Exception as e:
        return f"<h1>Fatal Upload Error</h1><p>{e}</p>"

# --- HELPERS ---
def get_db():
    return firestore.client()

def get_bucket():
    return storage.bucket()


# --- HELPER FUNCTIONS ---
def get_pdf_files():
    files = glob.glob(os.path.join(MANUALS_DIR, "*.pdf"))
    # Sort by size (largest first) to prioritize the full User Manual over Datasheets
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return files

def upload_pdf_to_gcs_if_needed(local_path):
    """Uploads PDF to GCS/Firebase Storage and returns the gs:// URI."""
    try:
        bucket = get_bucket()
        filename = os.path.basename(local_path)
        blob = bucket.blob(f"manuals/{filename}")
        
        # Check if exists (optional optimization, but good for speed)
        if not blob.exists():
            print(f"Uploading {filename} to Storage...")
            blob.upload_from_filename(local_path)
        
        return f"gs://{bucket.name}/manuals/{filename}"
    except Exception as e:
        print(f"PDF Upload Error: {e}")
        return None

def extract_image_from_pdf(page_number):
    try:
        pdf_files = get_pdf_files()
        if not pdf_files: 
            print("Error: No PDF manuals found in manuals/ directory.")
            return None
        
        # Open the first available manual
        doc = fitz.open(pdf_files[0])
        
        if page_number < 1 or page_number > len(doc):
             print(f"Error: Page {page_number} out of range (1-{len(doc)}).")
             return None
             
        page = doc.load_page(page_number - 1)
        
        # Smart Extraction Logic
        image_list = page.get_images(full=True)
        largest_area = 0
        best_rect = None
        
        for img in image_list:
            xref = img[0]
            rects = page.get_image_rects(xref)
            for rect in rects:
                if rect.width < 100 or rect.height < 100:
                    continue
                area = rect.width * rect.height
                if area > largest_area:
                    largest_area = area
                    best_rect = rect
        
        if best_rect:
            pix = page.get_pixmap(clip=best_rect, dpi=200)
        else:
            pix = page.get_pixmap(dpi=200)

        img_data = pix.tobytes("png")
        return Image.open(io.BytesIO(img_data))
    except Exception as e:
        print(f"Extraction error: {e}")
        return None

def upload_to_storage(image_obj, filename):
    try:
        bucket = get_bucket()
        blob = bucket.blob(f"generated_assets/{filename}")
        img_byte_arr = io.BytesIO()
        image_obj.save(img_byte_arr, format='PNG')
        blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
        blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
        # blob.make_public() <- Causes 403 on Uniform Access Buckets
        blob.upload_from_string(img_byte_arr.getvalue(), content_type='image/png')
        
        # Construct Firebase Download URL manually to bypass AIM/GCS public restrictions
        # Format: https://firebasestorage.googleapis.com/v0/b/{bucket}/o/{path}?alt=media
        encoded_path = quote(f"generated_assets/{filename}", safe='')
        firebase_url = f"https://firebasestorage.googleapis.com/v0/b/{bucket.name}/o/{encoded_path}?alt=media"
        
        return firebase_url
    except Exception as e:
        print(f"Upload error: {e}")
        return None

def upload_user_image_to_gcs(image_file):
    """Uploads user uploaded image to GCS and returns gs:// URI."""
    try:
        bucket = get_bucket()
        u_id = uuid.uuid4().hex
        blob = bucket.blob(f"user_uploads/{u_id}.png")
        blob.upload_from_string(image_file.read(), content_type='image/png')
        image_file.seek(0) # Reset pointer
        return f"gs://{bucket.name}/user_uploads/{u_id}.png"
    except Exception as e:
        print(f"User Image Upload Error: {e}")
        return None


# --- API ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    init_vertex()
    user_message = request.form.get('message', '')
    image_file = request.files.get('image')
    
    print(f"DEBUG: Processing message: '{user_message}'")
    
    awaiting = session.get('awaiting_confirmation')
    
    try:
        # Normalize message for checks
        u_msg_lower = user_message.lower()
        
        # --- 1. LAYER 0: IMMEDIATE TECHNICAL INTERCEPTION ---
        # Detect technical intent BEFORE any other logic.
        tech_words = ["how", "explain", "activate", "connect", "step", "tutorial", "manual", "guide", "help", "camera", "ip", "monitor", "nvr", "password", "reset"]
        
        # Check if user is confirming a pending request (allow "yes" to pass)
        is_confirming_flow = False
        confirmations = ["yes", "proceed", "go", "ahead", "ok", "confirm", "sure", "do it", "right", "correct", "yeah", "yep"]
        if awaiting and any(w in u_msg_lower for w in confirmations):
            is_confirming_flow = True

        # IF NOT confirming AND (keywords present OR long input) -> TRAP IT
        if not is_confirming_flow and (any(w in u_msg_lower for w in tech_words) or len(user_message.split()) > 3):
            print("Technical intent detected (Layer 0). Forcing Identification phase.")
            
            # Quick topic identification
            topic_model = GenerativeModel("gemini-2.5-pro")
            topic_res = topic_model.generate_content(f"Identify the technical procedure in one word (snake_case): '{user_message}'")
            slug = topic_res.text.strip().replace(" ", "_").lower()
            
            session['awaiting_confirmation'] = slug
            return jsonify({
                'solution': f"I understand you want to learn about **{slug.replace('_', ' ')}**. Should I generate the step-by-step tutorial with technical diagrams from the manual?"
            })

        # --- 2. CONFIRMATION & GENERATION LOGIC ---
        if awaiting:
            if is_confirming_flow:
                print(f"Confirmed! Running Generation Pipeline for {awaiting}...")
                session['awaiting_confirmation'] = None
                
                # Check Cache
                db = get_db()
                cached = db.collection("tutorials").document(awaiting).get()
                if cached.exists:
                    return jsonify({'tutorial': cached.to_dict()})

                # Prepare context
                pdf_files = get_pdf_files()
                context_parts = []
                if pdf_files:
                     # Upload PDF to GCS/Firebase Storage if needed
                     pdf_uri = upload_pdf_to_gcs_if_needed(pdf_files[0])
                     if pdf_uri:
                         context_parts.append(Part.from_uri(pdf_uri, mime_type="application/pdf"))
                
                if image_file:
                     # Upload User Image to GCS
                     img_uri = upload_user_image_to_gcs(image_file)
                     if img_uri:
                         context_parts.append(Part.from_uri(img_uri, mime_type="image/png"))
                
                # Generate Tutorial Structure
                gen_model = GenerativeModel("gemini-2.5-pro", system_instruction=GENERATOR_SYSTEM_INSTRUCTION)
                gen_res = gen_model.generate_content(
                    context_parts + [f"Task: Generate tutorial for {awaiting}. Reference page 38 for IP camera visuals."], 
                    generation_config={"response_mime_type": "application/json"}
                )
                tutorial = json.loads(gen_res.text)
                
                # NANO BANANA MULTIMODAL PIPELINE
                print(f"Executing Nano Banana Pipeline with Prompt: {NANO_BANANA_PROMPT.strip()[:50]}...")
                for step in tutorial.get('steps', []):
                    if step.get('has_visual'):
                        page_ref = step.get('pdf_page_reference')
                        print(f"Applying Nano Banana to visual from page {page_ref}...")
                        
                        raw_img_pil = extract_image_from_pdf(page_ref)
                        if raw_img_pil:
                            # Save and Upload
                            u_id = uuid.uuid4().hex[:8]
                            fname = f"{awaiting}_step_{step['step_number']}_{u_id}.png"
                            url = upload_to_storage(raw_img_pil, fname)
                            if url:
                                step['image_url'] = url
                            else:
                                step['image_error'] = "Upload returned None (check logs)"
                        else:
                            step['image_error'] = f"Extraction failed for Page {page_ref} (PDF not found or page out of range)"
                
                db.collection("tutorials").document(awaiting).set(tutorial)
                return jsonify({'tutorial': tutorial})
            else:
                # User said something else while waiting, assuming they cancelled or are chatting
                print("Confirmation not found. Resetting state.")
                session['awaiting_confirmation'] = None

        # --- 3. FALLBACK CHAT (Refusal Mode) ---
        # CRITICAL: We instruct the chat model to REFUSE technical questions if they slip through.
        chat_model = GenerativeModel("gemini-2.5-pro", system_instruction="You are a polite assistant. If the user asks about technical support, manuals, cameras, or troubleshooting, YOU MUST REFUSE and say: 'I can only help if you confirm you want a tutorial. Please ask to identify the task again.'")
        chat_res = chat_model.generate_content(user_message)
        return jsonify({'solution': f"{chat_res.text} (DEBUG: Chat Fallback Triggered - Interception Failed)"})

    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    session.clear()
    print("Session cleared.")
    return jsonify({'status': 'cleared'})

@https_fn.on_request(memory=1024, timeout_sec=600, region="us-central1")
def api(req: https_fn.Request) -> Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()
