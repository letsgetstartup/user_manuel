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

# --- CONFIGURATION ---
# Based on your screenshots:
PROJECT_ID = "manualai-481406" 
# CRITICAL: Vertex AI Model needs 'us-central1', but Data Store is in 'eu'
VERTEX_REGION = "us-central1"
DATA_STORE_REGION = "eu"
DATA_STORE_ID = "manual02_1765869275504" # The exact ID from your screenshot [cite: 8]

# Initialize Firebase
if not firebase_admin._apps:
    firebase_admin.initialize_app()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Setup Logging
logging.basicConfig(level=logging.INFO)

def init_vertex():
    """Initializes Vertex AI with the correct region."""
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)


# --- SYSTEM INSTRUCTIONS ---
# We removed the 'Refusal' instruction. Now we encourage using the tools.
TECH_SUPPORT_INSTRUCTION = """
You are an expert Technical Support Engineer for security camera systems (Hikvision, Dahua, etc.).
Your goal is to troubleshoot issues based ONLY on the provided user image and the official technical manuals.

PROTOCOL:
1. ANALYZE: Look at the user's uploaded image (if provided) to identify error codes, interface icons, or hardware models.
2. SEARCH: Use the 'GoogleSearchRetrieval' tool to find specific troubleshooting steps in the connected Data Store.
3. SYNTHESIZE: Combine the visual evidence with the manual's text.
4. CITE: When you provide a solution, mention which manual or section the information came from.

If the answer is not in the manuals, state that clearly. Do not make up information.
"""

# --- ROUTES ---

@app.route('/')
def index():
    return "Antigravity Copilot Backend is Running (Grounding Enabled)."

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    Receives text and/or image -> Queries Vertex AI with Grounding -> Returns Answer.
    """
    try:
        init_vertex()
        
        user_message = request.form.get('message', '')
        image_file = request.files.get('image')
        
        logging.info(f"Processing request: '{user_message}' | Image present: {bool(image_file)}")

        # --- CONFIGURE GROUNDING TOOL (Available for all paths) ---
        grounding_tool = Tool.from_retrieval(
            grounding.Retrieval(
                source=grounding.VertexAISearch(
                    datastore=f"projects/{PROJECT_ID}/locations/{DATA_STORE_REGION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
                )
            )
        )

        if not user_message and not image_file:
             return jsonify({'solution': "Please provide a description or an image of the error."})

        # Check for Confirmation Keywords
        CONFIRMATION_KEYWORDS = ["yes", "go ahead", "ok", "please", "sure", "generate"]
        is_confirmation = any(word in user_message.lower() for word in CONFIRMATION_KEYWORDS) if user_message else False
        
        # --- PATH 1: TUTORIAL GENERATION (JSON) ---
        if is_confirmation:
            context = session.get('last_analysis', '')
            if not context and not image_file:
                 return jsonify({'solution': "I don't have context on what to generate. Please describe the problem first."})
            
            logging.info("Generating JSON Tutorial...")
            
            # Use a specialized model for JSON generation
            tutorial_model = GenerativeModel(
                "gemini-2.5-pro",
                system_instruction="""
                You are a technical documentation expert.
                Generate a structured JSON tutorial based on the user's request and context.
                
                The JSON MUST follow this exact schema:
                {
                    "tutorial": {
                        "title": "Title of the Guide",
                        "intro": "Brief introduction.",
                        "steps": [
                            {
                                "step_number": 1,
                                "instruction": "Clear instruction.",
                                "page_number": 10,
                                "has_visual": true,
                                "image_url": "" 
                            }
                        ]
                    }
                }
                
                NOTE: You MUST identify the Page Number in the manual where this step is described and include it as "page_number" (integer). If you cannot find the page, omit the field.
                NOTE: Leave 'image_url' empty and 'has_visual' false initially; the system will fill them based on "page_number".
                """,
                tools=[grounding_tool]
            )
            
            prompt = f"Context: {context}\nUser Request: {user_message}\nGenerate a step-by-step tutorial in JSON format."
            
            response = tutorial_model.generate_content(
                [prompt],
                generation_config={"response_mime_type": "application/json"}
            )
            
            import json
            import json
            try:
                # Handle potentially multipart responses (e.g. text + citations)
                full_text = ""
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text:
                            full_text += part.text
                else:
                    full_text = response.text # Fallback logic

                # Clean up response text to ensure valid JSON
                text_response = full_text.strip()
                
                # Try to find JSON block using regex
                json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
                if json_match:
                    text_response = json_match.group(0)
                elif text_response.startswith("```json"):
                    text_response = text_response[7:-3]
                elif text_response.startswith("```"): # Generic block
                     text_response = text_response[3:-3]
                
                tutorial_data = json.loads(text_response)

                # --- PDF IMAGE EXTRACTION LOGIC ---
                manual_uri = session.get('manual_uri')
                if manual_uri and 'gs://' in manual_uri:
                    logging.info(f"Attempting to extract images from: {manual_uri}")
                    try:
                        # Parse correct bucket/blob from gs:// URI
                        # URI format: gs://bucket_name/path/to/file.pdf
                        match = re.match(r'gs://([^/]+)/(.+)', manual_uri)
                        if match:
                            source_bucket_name = match.group(1)
                            source_blob_name = match.group(2)

                            # Download PDF from Source GCS
                            storage_client = gcs.Client()
                            source_bucket = storage_client.bucket(source_bucket_name)
                            blob = source_bucket.blob(source_blob_name)
                            pdf_bytes = blob.download_as_bytes()
                            
                            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                            output_bucket = storage.bucket() # Default Firebase bucket
                            
                            for step in tutorial_data.get('tutorial', {}).get('steps', []):
                                page_num = step.get('page_number')
                                
                                # Only process if we have a page number
                                if page_num is not None and isinstance(page_num, int):
                                    try:
                                        # Convert 1-based page number to 0-based index
                                        page_idx = page_num - 1
                                        if 0 <= page_idx < len(doc):
                                            page = doc[page_idx]
                                            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for quality
                                            img_data = pix.tobytes("png")
                                            
                                            # Upload to Firebase Storage
                                            # Use a unique path
                                            filename = f"generated_visuals/{session.get('session_id', 'anon')}_{step['step_number']}_{page_num}.png"
                                            blob_out = output_bucket.blob(filename)
                                            blob_out.upload_from_string(img_data, content_type='image/png')
                                            blob_out.make_public()
                                            
                                            step['image_url'] = blob_out.public_url
                                            step['has_visual'] = True
                                            logging.info(f"Generated visual for Step {step['step_number']} from Page {page_num}")
                                    except Exception as inner_e:
                                        logging.error(f"Failed to extract page {page_num}: {inner_e}")
                    except Exception as e:
                        logging.error(f"Global PDF extraction failed: {e}")

                return jsonify(tutorial_data)
            except (json.JSONDecodeError, AttributeError): # AttributeError in case json_match is None but we try to continue
                logging.error("Failed to parse JSON response")
                return jsonify({'solution': "Error generating tutorial format. Here is the raw text:\n" + full_text})

        # --- PATH 2: ANALYSIS & CONFIRMATION (TEXT) ---
        else:
            # 1. Prepare Multimodal Content
            content_parts = []
            if user_message:
                content_parts.append(user_message)
            else:
                content_parts.append("Analyze this technical issue.")

            if image_file:
                img = Image.open(image_file)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_bytes = img_byte_arr.getvalue()
                part_image = Part.from_data(data=img_bytes, mime_type="image/png")
                content_parts.append(part_image)

            # 3. Initialize Model
            model = GenerativeModel(
                "gemini-2.5-pro",
                system_instruction=TECH_SUPPORT_INSTRUCTION + "\nIMPORTANT: Identify the problem, then END your response by asking: 'Shall I generate a step-by-step tutorial with diagrams?'",
                tools=[grounding_tool]
            )

            # 4. Generate Response
            response = model.generate_content(
                content_parts,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024
                }
            )

            answer_text = response.text
            
            # Extract source URI from Grounding Metadata
            source_uri = ""
            try:
                if response.candidates and response.candidates[0].grounding_metadata:
                    gm = response.candidates[0].grounding_metadata
                    if gm.grounding_chunks:
                        for chunk in gm.grounding_chunks:
                            if chunk.retrieved_context and chunk.retrieved_context.uri:
                                source_uri = chunk.retrieved_context.uri
                                break # Take the first valid citation
            except Exception as e:
                logging.error(f"Error extracting source URI: {e}")

            # Store context for the next turn
            session['last_analysis'] = f"User Image/Issue Analysis: {answer_text}"
            session['manual_uri'] = source_uri
            if 'session_id' not in session:
                session['session_id'] = os.urandom(8).hex()
            session.modified = True

            return jsonify({
                'solution': answer_text,
                'grounded': True
            })

    except Exception as e:
        logging.error(f"Analysis Error: {e}")
        return jsonify({'error': str(e), 'solution': "I encountered a system error while checking the manuals."}), 500

@app.route('/api/clear', methods=['POST'])
def clear_history():
    session.clear()
    return jsonify({'status': 'cleared'})

# Firebase Cloud Function Entry Point
@https_fn.on_request(memory=1024, timeout_sec=60, region="us-central1")
def api(req: https_fn.Request) -> Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()
