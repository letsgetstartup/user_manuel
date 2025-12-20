import os
import io
import logging
from flask import Flask, request, jsonify, session
from firebase_functions import https_fn
from werkzeug.wrappers import Response
from PIL import Image

# Vertex AI & Firebase Imports
# Vertex AI & Firebase Imports
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Tool, grounding
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud import storage as gcs
import fitz # PyMuPDF
import re
import json

# --- CONFIGURATION ---
PROJECT_ID = "manualai-481406"
VERTEX_REGION = "us-central1"
DATA_STORE_REGION = "eu"
DATA_STORE_ID = "manual02_1765869275504"

# Initialize Firebase
if not firebase_admin._apps:
    firebase_admin.initialize_app()

app = Flask(__name__)
app.secret_key = os.urandom(24)
db = firestore.client()
bucket = storage.bucket()

# Setup Logging
logging.basicConfig(level=logging.INFO)

def init_vertex():
    """Initializes Vertex AI."""
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)

# --- SYSTEM INSTRUCTIONS ---

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
Your task: Create a step-by-step tutorial based ONLY on the provided Context.

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
      "page_number": 10, 
      "has_visual": true,
      "image_url": ""   
    }
  ]
}

Rules:
1. If a step corresponds to a diagram/screenshot in the manual, set "has_visual": true and provide the precise "page_number" (integer).
2. Keep instructions concise and professional.
"""

# --- HELPER FUNCTIONS ---

def get_tutorial_from_db(slug):
    """Cache Hit: Retrieve from Firestore."""
    try:
        doc = db.collection("tutorials").document(slug).get()
        if doc.exists:
            logging.info(f"Cache HIT for: {slug}")
            return doc.to_dict()
    except Exception as e:
        logging.error(f"DB Read Error: {e}")
    return None

def save_tutorial_to_db(slug, data):
    """Save new tutorial to Firestore."""
    try:
        db.collection("tutorials").document(slug).set(data)
        logging.info(f"Saved tutorial to DB: {slug}")
    except Exception as e:
        logging.error(f"DB Write Error: {e}")

def extract_and_upload_images(tutorial_data, manual_uri):
    """Extracts images from PDF in GCS and updates the tutorial data with public URLs."""
    if not manual_uri or 'gs://' not in manual_uri:
        logging.warning("No valid GCS URI found for image extraction.")
        return tutorial_data

    try:
        # Parse GCS URI
        match = re.match(r'gs://([^/]+)/(.+)', manual_uri)
        if not match: return tutorial_data
        
        source_bucket_name, source_blob_name = match.groups()
        
        # Download PDF
        storage_client = gcs.Client()
        blob = storage_client.bucket(source_bucket_name).blob(source_blob_name)
        pdf_bytes = blob.download_as_bytes()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        updated_steps = []
        for step in tutorial_data.get("steps", []):
            page_num = step.get('page_number')
            
            # Logic: If has_visual is Requested OR we have a page number
            if page_num is not None and isinstance(page_num, int):
                try:
                    page_idx = page_num - 1
                    if 0 <= page_idx < len(doc):
                        page = doc[page_idx]
                        
                        # Heuristic: Get the largest image on the page
                        images = page.get_images()
                        best_img = None
                        max_area = 0
                        
                        for img in images:
                            xref, _, w, h, _, _, _, _, _ = img
                            if w > 100 and h > 100: # Filter small icons
                                area = w * h
                                if area > max_area:
                                    max_area = area
                                    best_img = img

                        if best_img:
                            xref = best_img[0]
                            base_image = doc.extract_image(xref)
                            img_data = base_image["image"]
                            ext = base_image["ext"]
                            
                            # Upload to Firebase Storage
                            filename = f"generated_visuals/{tutorial_data['topic_slug']}_step_{step['step_number']}_{page_num}.{ext}"
                            blob_out = bucket.blob(filename)
                            blob_out.upload_from_string(img_data, content_type=f"image/{ext}")
                            blob_out.make_public()
                            
                            step['image_url'] = blob_out.public_url
                            step['has_visual'] = True
                        else:
                             # Fallback: snapshot the whole page if no specific image found
                             # (Optional: implement if needed, for new skip)
                             step['has_visual'] = False
                             
                except Exception as inner_e:
                    logging.error(f"Image extraction failed for step {step['step_number']}: {inner_e}")
            
            updated_steps.append(step)
            
        tutorial_data['steps'] = updated_steps
        
    except Exception as e:
        logging.error(f"Global extraction error: {e}")

    return tutorial_data

# --- ROUTES ---

@app.route('/')
def index():
    return "Universal Guide AI API is Running."

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        init_vertex()
        user_message = request.form.get('message', '')
        image_file = request.files.get('image')
        
        logging.info(f"Request: {user_message} | Image: {bool(image_file)}")

        if not user_message and not image_file:
             return jsonify({'solution': "Please describe the issue."})

        # --- 1. ROUTER PHASE ---
        
        # [FIX] Fetch existing slugs to enforce cache hits
        existing_slugs = []
        try:
            docs = db.collection("tutorials").stream()
            existing_slugs = [doc.id for doc in docs]
            logging.info(f"Checking against {len(existing_slugs)} existing guides.")
        except Exception as e:
            logging.error(f"Failed to fetch slugs: {e}")

        router_model = GenerativeModel("gemini-2.0-flash-exp", system_instruction=ROUTER_SYSTEM_INSTRUCTION)
        
        # [FIX] Strict Context-Aware Prompt
        router_prompt = f"""
        ROLE: Library Archivist.
        GOAL: Match the user's query to an existing TOPIC in the DATABASE.

        DATABASE OF EXISTING TOPICS:
        {json.dumps(existing_slugs)}

        USER QUERY: "{user_message}"

        INSTRUCTIONS:
        1. SEARCH the DATABASE for a topic that matches the user's intent.
        2. DECISION:
            - IF a match is found: RETURN the EXACT topic_slug from the database.
            - IF NO match is found: Generate a NEW, concise, snake_case slug.

        Rules:
        - Do NOT generate a new slug if a semantically similar one exists.
        - Output JSON ONLY.
        """
        
        router_res = router_model.generate_content(
            [router_prompt], 
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.0
            }
        )
        
        router_data = json.loads(router_res.text)
        slug = router_data.get("topic_slug")
        
        if router_data.get("needs_clarification"):
            return jsonify({'solution': router_data.get("clarification_question")})

        if not slug:
             # Fallback
             slug = "general_inquiry"

        logging.info(f"Topic Identified: {slug}")

        # --- 2. CACHE CHECK (MEMORY) ---
        cached_tutorial = get_tutorial_from_db(slug)
        if cached_tutorial:
             logging.info("Returning Cached Tutorial")
             return jsonify({'tutorial': cached_tutorial})

        # --- 3. GENERATION PHASE (CACHE MISS) ---
        logging.info("Cache Miss. Generating...")
        
        # Grounding Setup
        grounding_tool = Tool.from_retrieval(
            grounding.Retrieval(
                source=grounding.VertexAISearch(
                    datastore=f"projects/{PROJECT_ID}/locations/{DATA_STORE_REGION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
                )
            )
        )

        gen_model = GenerativeModel(
            "gemini-2.5-pro", # Use 2.5 Pro for better reasoning
            system_instruction=GENERATOR_SYSTEM_INSTRUCTION,
            tools=[grounding_tool]
        )
        
        gen_prompt = f"""
        User Request: {user_message}
        Topic Slug: {slug}
        Generate a step-by-step tutorial.
        """
        
        gen_res = gen_model.generate_content(
            [gen_prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        
        try:
             # Logic to extract Manual URI from grounding metadata
            source_uri = ""
            if gen_res.candidates and gen_res.candidates[0].grounding_metadata:
                gm = gen_res.candidates[0].grounding_metadata
                if gm.grounding_chunks:
                    for chunk in gm.grounding_chunks:
                        if chunk.retrieved_context and chunk.retrieved_context.uri:
                            source_uri = chunk.retrieved_context.uri
                            break
            
            tutorial_data = json.loads(gen_res.text)
            
            # --- 4. IMAGE EXTRACTION ---
            if source_uri:
                tutorial_data = extract_and_upload_images(tutorial_data, source_uri)
            
            # --- 5. SAVE TO MEMORY ---
            save_tutorial_to_db(slug, tutorial_data)
            
            return jsonify({'tutorial': tutorial_data})

        except (json.JSONDecodeError, ValueError) as e:
            logging.error(f"Generation parsing error: {e}")
            # Fallback to plain text if JSON fails
            return jsonify({'solution': gen_res.text})

    except Exception as e:
        logging.error(f"Global Error: {e}")
        return jsonify({'error': str(e)}), 500

@https_fn.on_request(memory=1024, timeout_sec=60, region="us-central1")
def api(req: https_fn.Request) -> Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()

