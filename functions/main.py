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
from vertexai.generative_models import GenerativeModel, Part, Tool, grounding, ToolConfig
from vertexai.vision_models import ImageGenerationModel
import firebase_admin
from firebase_admin import credentials, firestore, storage
from google.cloud import storage as gcs
import fitz # PyMuPDF
import re
import json
from datetime import timedelta

# --- CONFIGURATION ---
PROJECT_ID = "manualai-481406"
VERTEX_REGION = "us-central1"
DATA_STORE_REGION = "eu"
DATA_STORE_ID = "manual02_1765869275504"

# Initialize Firebase lazy-loaded

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Lazy Initialization
db = None
gcs_client = None 
upload_bucket = None
bucket = None

def init_services():
    """Initializes Firebase and GCS clients if not already initialized."""
    global db, gcs_client, upload_bucket, bucket
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(options={
            'storageBucket': f"{PROJECT_ID}.appspot.com"
        })

    if db is None:
        db = firestore.client()
    if gcs_client is None:
        gcs_client = gcs.Client()
        upload_bucket = gcs_client.bucket("manualai02a")
        bucket = storage.bucket()

# Setup Logging
logging.basicConfig(level=logging.INFO)

def init_vertex():
    """Initializes Vertex AI."""
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)

# --- NANO BANANA CONFIG ---
NANO_BANANA_PROMPT = """Create a copy of the attached image based on the following prompt:

Role & Goal: You are UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image. You must prioritize data accuracy and structural integrity over artistic interpretation.

Replicate the exact window title

Exact Labels & Values (Row-by-Row): You MUST display the data exactly as written, without any changes or generalizations 

Icons - copy exactly where they appear in the source.

DO NOT use placeholder text or "lorem ipsum".
DO NOT generalize numbers or IP addresses.
DO NOT add artistic lighting, reflections, or textures.
DO NOT simplify the diagram; if there are 8 rows in the source, there must be exactly 8 rows in the output.

Final Quality Check: The final image must be a high-resolution scan of the attached image Every number, dot, icon and line must match the provided specification."""

def extract_relevant_images(doc, page_index):
    """
    Finds specific image objects on a PDF page.
    Returns a list of PIL Images.
    """
    images = []
    try:
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            images.append(Image.open(io.BytesIO(image_bytes)))
            
        # If no images found as objects, fallback to a smart extraction of the page
        if not images:
            pix = page.get_pixmap(dpi=150)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append(img)
            
    except Exception as e:
        logging.error(f"Error in extract_relevant_images: {e}")
    return images

def call_nano_banana(source_image_pil):
    """
    The 'Nano Banana' API implementation using Imagen 3.
    """
    try:
        init_vertex()
        # Step 1: Create a high-quality visual description
        analysis_model = GenerativeModel("gemini-1.5-pro-002")
        
        img_byte_arr = io.BytesIO()
        source_image_pil.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        image_part = Part.from_data(img_bytes, mime_type="image/png")
        
        spec_prompt = f"{NANO_BANANA_PROMPT}\n\nTask: Based on the instructions above, provide a high-resolution, technical visual description of this image suitable for generating a pixel-perfect replica. Focus on text content, UI layout, and data accuracy."
        
        analysis_res = analysis_model.generate_content([image_part, spec_prompt])
        visual_spec = analysis_res.text
        
        # Step 2: Generate replica with Imagen 3
        imagen_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")
        final_prompt = f"Pixel-perfect UI replica diagram. {visual_spec}"
        
        response = imagen_model.generate_images(
            prompt=final_prompt[:1500],
            number_of_images=1
        )
        
        if response.images:
            gen_img_bytes = response.images[0]._image_bytes
            return Image.open(io.BytesIO(gen_img_bytes))
            
    except Exception as e:
        logging.error(f"Nano Banana API Error: {e}")
    return source_image_pil

def select_most_relevant_image(images, instruction):
    """
    Uses Gemini Vision to pick the best image index from a list for a given instruction.
    """
    if not images:
        return None
    if len(images) == 1:
        return images[0]

    try:
        init_vertex()
        model = GenerativeModel("gemini-1.5-flash") # Fast & good enough for selection
        
        prompt = f"""
        Task: Below are several images extracted from a technical manual.
        Which image best matches this instruction: "{instruction}"?
        
        Rules:
        1. If one image clearly depicts the UI or action described, pick it.
        2. Output ONLY the index of the image (starting from 0).
        3. If none match specifically, pick 0.
        """
        
        parts = [prompt]
        for idx, img in enumerate(images):
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            parts.append(f"Image {idx}:")
            parts.append(Part.from_data(img_byte_arr.getvalue(), mime_type="image/png"))
            
        res = model.generate_content(parts)
        text = res.text.strip()
        
        # Extract the first digit found
        match = re.search(r'\d+', text)
        if match:
            idx = int(match.group(0))
            if 0 <= idx < len(images):
                logging.info(f"Intelligent Selection: Picked Image {idx} for '{instruction[:30]}...'")
                return images[idx]
                
    except Exception as e:
        logging.error(f"Selection Error: {e}")
    
    return images[0] # Fallback

# --- SYSTEM INSTRUCTIONS ---

# --- SYSTEM INSTRUCTIONS ---

GENERATOR_SYSTEM_INSTRUCTION = """
You are an expert Technical Support Engineer.
Your Goal: Solve the user's specific problem using ONLY the provided Grounding Source (User Manuals). 

CRITICAL: 
1. YOU MUST USE THE GROUNDING TOOL (Google Search) for EVERY query. 
2. EVERY STEP MUST include a 'page_number' from the manual. BE PRECISE. If different steps are on different pages, show the exact page for each.
3. UNIQUE VISUALS: If the manual has specific screenshots for specific steps, ensure you link the correct page for that specific step. DO NOT use the same page/image for every step if the manual has more detail.
4. DO NOT answer from memory. If you cannot find info in the manuals, say so.

Input:
1. User provides a Message and/or a Photo of the error.
2. Grounding Tool provides specific context from manuals.

Process:
1. YOU MUST USE THE GROUNDING TOOL (Google Search) for every request. Do not answer from memory.
2. Examine the user's PHOTO (if provided) and MESSAGE to identify symbols, error codes, or UI screens.
3. Search the manuals for the specific issue identified.
4. Use the specific page numbers and diagrams found in the retrieval context.
5. Output the solution strictly in the JSON format below.

JSON Format:
{
  "title": "Specific Title of the Fix",
  "topic_slug": "generate_a_unique_specific_slug_based_on_content",
  "intro": "Direct answer to the user's question or analysis of their photo...",
  "steps": [
    {
      "step_number": 1,
      "instruction": "Detailed technical step...",
      "page_number": 12,
      "has_visual": true
    }
  ]
}

Rules:
- ACCURACY: Only output steps found in the retrieved context.
- VISUALS: If the manual shows a diagram for a step, set "has_visual": true and provide the precise "page_number".
- SINGLE OUTPUT: Provide ONLY ONE valid JSON object.
- FALLBACK: If the answer is not in the manual, return {"solution": "I could not find this information in the manual."}
"""

PLANNING_SYSTEM_INSTRUCTION = """
You are a helpful Technical Support Assistant.
Your Goal: check if the user's query can be answered using the provided Grounding Source (User Manuals).

Input:
1. User Query.
2. Grounding Tool context.

Process:
1. Search the manuals for the user's query.
2. If found:
   a. Provide a BRIEF SOLUTION to the issue (1-2 sentences).
   b. Ask: "Is this clear, or would you like me to generate the full step-by-step tutorial with technical diagrams from the manual?"
3. If NOT found: Apologize and say you couldn't find it.

Output: PURE TEXT (Do not use JSON).
"""

def analyze_image_for_query(image_file, user_message):
    """Step 1: Use Vision model to convert Photo -> Search Query."""
    try:
        logging.info("Step 1: Vision Analysis Started")
        vision_model = GenerativeModel("gemini-2.5-pro") # No tools for this step
        
        prompt = f"""
        Analyze this image. 
        If it shows an error screen, extract the exact error message and code.
        If it shows a device, identify the model or likely interface.
        
        User's Comment: "{user_message}"
        
        Task: Combine the visual evidence and the user's comment into a single, precise search query to find the solution in a manual.
        Output ONLY the search query. Do not add markdown or explanations.
        """
        
        image_file.seek(0)
        image_data = image_file.read()
        image_part = Part.from_data(data=image_data, mime_type="image/png")
        
        res = vision_model.generate_content([prompt, image_part])
        query = res.text.strip()
        logging.info(f"Step 1: Vision Result -> '{query}'")
        return query
        
    except Exception as e:
        logging.error(f"Vision Analysis Failed: {e}")
        return user_message # Fallback

def generate_canonical_slug(user_query):
    """
    Uses Gemini to generate a consistent, canonical slug for a user query.
    Example: "How to add an IP camera" -> "add_ip_camera"
    Example: "Adding a camera" -> "add_ip_camera"
    """
    try:
        logging.info(f"Generating Canonical Slug for: {user_query}")
        init_vertex()
        model = GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        Task: Convert the following user technical support query into a consistent, snake_case unique identifier (slug).
        
        Input Query: "{user_query}"
        
        Rules:
        1. Use snake_case (e.g., "reset_password", "connection_failure").
        2. Be GENERAL and CANONICAL. 
           - "How do I add a camera" -> "add_ip_camera"
           - "Adding new IP camera" -> "add_ip_camera"
           - "My NVR is beeping" -> "nvr_beeping_sound"
           - "Beeping noise from NVR" -> "nvr_beeping_sound"
        3. Output ONLY the slug. No json, no quotes, no markdown.
        """
        
        res = model.generate_content(prompt)
        slug = res.text.strip().lower().replace("```", "").replace("\n", "")
        logging.info(f"Canonical Slug Result: {slug}")
        return slug
    except Exception as e:
        logging.error(f"Slug Generation Error: {e}")
        # Fallback to a basic sanitization if AI fails
        return re.sub(r'[^a-zA-Z0-9_]', '_', user_query.lower())[:50]

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

def extract_text_from_gemini(response):
    """Safely extracts text from Gemini response, handling multiple parts and markdown."""
    full_text = ""
    try:
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                full_text += part.text
        else:
            full_text = response.text
    except Exception as e:
        logging.warning(f"Error extracting text from parts: {e}")
        # Last resort try/except for .text
        try:
           full_text = response.text
        except:
           pass

    # Clean Markdown
    clean_text = full_text.replace("```json", "").replace("```", "").strip()
    
    # [FIX] Regex Extraction for "Chatty" JSON (e.g. "Here is the JSON: {...}")
    try:
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        if match:
            return match.group(0)
    except:
        pass

    return clean_text

def download_manual_from_gcs(gcs_uri):
    """Downloads the manual PDF bytes from the GS URI provided by Vertex AI."""
    try:
        # URI format: gs://bucket-name/path/to/manual.pdf
        if not gcs_uri or not gcs_uri.startswith("gs://"):
            logging.error(f"Invalid GCS URI: {gcs_uri}")
            return None
            
        matches = re.match(r'gs://([^/]+)/(.+)', gcs_uri)
        if not matches:
            logging.warning(f"Could not parse GCS URI: {gcs_uri}")
            return None
            
        bucket_name = matches.group(1)
        blob_name = matches.group(2)
        
        logging.info(f"Downloading manual: {blob_name} from {bucket_name}")
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        logging.error(f"Failed to download PDF from GCS: {e}")
        return None

def extract_and_upload_images(tutorial_data, manual_uri):
    """
    1. Downloads PDF from GCS.
    2. Extracts images from specific pages.
    3. Uploads to Firebase for frontend display.
    """
    logging.info(f"Starting Extraction Pipeline for: {manual_uri}")
    extraction_log = [] # Debug log for frontend
    
    pdf_bytes = download_manual_from_gcs(manual_uri)
    
    if not pdf_bytes:
        logging.warning("Aborting extraction: PDF bytes not found.")
        extraction_log.append("ERROR: Failed to download PDF from GCS.")
        tutorial_data['intro'] = tutorial_data.get('intro', '') + f" [EXTRACTION LOG: {'; '.join(extraction_log)}]"
        return tutorial_data

    extraction_log.append(f"PDF downloaded: {len(pdf_bytes)} bytes")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        extraction_log.append(f"PDF opened: {len(doc)} pages")
        processed_steps = []

        for step in tutorial_data.get("steps", []):
            # Support both 'page_number' (Prompt default) and 'pdf_page_reference' (User code)
            page_val = step.get('page_number') or step.get('pdf_page_reference')
            
            if step.get("has_visual") and page_val:
                try:
                    # PDF pages are 0-indexed, LLM usually gives 1-indexed
                    page_num = int(page_val) - 1
                    extraction_log.append(f"Step {step['step_number']}: Nano Banana Analysis of page {page_num + 1}")
                    
                    if 0 <= page_num < len(doc):
                        # 1. Extract RELEVANT images (not full page)
                        page_images = extract_relevant_images(doc, page_num)
                        
                        if page_images:
                            # 2. Intelligent Selection (Vertex AI Vision)
                            # Instead of just picking the largest, pick the most relevant one for the instruction
                            source_img = select_most_relevant_image(page_images, step.get('instruction', ''))
                            
                            # 2. Call Nano Banana for pixel-perfect replica
                            final_img = call_nano_banana(source_img)
                            
                            # 3. Convert to bytes for upload
                            img_byte_arr = io.BytesIO()
                            final_img.save(img_byte_arr, format='PNG')
                            img_data = img_byte_arr.getvalue()
                            
                            extraction_log.append(f"Step {step['step_number']}: Nano Banana Replica Ready ({len(img_data)} bytes)")
    
                            # 4. Upload to the bucket we KNOW exists and works
                            blob_path = f"generated_visuals/{tutorial_data['topic_slug']}_step_{step['step_number']}_{page_num}.png"
                            blob = upload_bucket.blob(blob_path)
                            blob.upload_from_string(img_data, content_type='image/png')
                            
                            # 5. Serve via Proxy
                            step["image_url"] = f"/api/image/{blob_path}"
                            extraction_log.append(f"Step {step['step_number']}: PROXIED")
                            logging.info(f"Nano Banana image for Step {step['step_number']} -> {step['image_url']}")
                        else:
                            extraction_log.append(f"Step {step['step_number']}: NO RELEVANT IMAGES FOUND ON PAGE")
                    else:
                        extraction_log.append(f"Step {step['step_number']}: Page {page_num + 1} OUT OF RANGE")
                        logging.warning(f"Page {page_num} out of range for Step {step['step_number']}")
                        
                except Exception as e:
                    extraction_log.append(f"Step {step.get('step_number')}: EXCEPTION - {str(e)}")
                    logging.error(f"Image extraction failed for step {step.get('step_number')}: {e}")
            
            processed_steps.append(step)
        
        tutorial_data["steps"] = processed_steps
        tutorial_data['intro'] = tutorial_data.get('intro', '') + f" [EXTRACTION LOG: {'; '.join(extraction_log)}]"
        return tutorial_data

    except Exception as e:
        extraction_log.append(f"GLOBAL EXCEPTION: {str(e)}")
        logging.error(f"Global extraction error: {e}")
        tutorial_data['intro'] = tutorial_data.get('intro', '') + f" [EXTRACTION LOG: {'; '.join(extraction_log)}]"
        return tutorial_data

# --- ROUTES ---

@app.route('/')
def index():
    return "Universal Guide AI API is Running."

@app.route('/api/image/<path:image_path>')
def serve_image(image_path):
    """Proxies images from GCS to bypass authentication/signing issues in the browser."""
    try:
        init_services()
        # Security: Only allow paths starting with generated_visuals/
        if not image_path.startswith("generated_visuals/"):
            return "Unauthorized path", 403
            
        blob = upload_bucket.blob(image_path)
        if not blob.exists():
            logging.error(f"Image not found in GCS: {image_path}")
            return "Image not found", 404
        
        img_bytes = blob.download_as_bytes()
        return Response(img_bytes, mimetype='image/png')
    except Exception as e:
        logging.error(f"Proxy Error for {image_path}: {e}")
        return str(e), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        init_vertex()
        init_services() # Ensure DB/GCS are ready
        user_message = request.form.get('message', '')
        image_file = request.files.get('image')
        
        logging.info(f"Request: {user_message} | Image: {bool(image_file)}")

        if not user_message and not image_file:
             return jsonify({'solution': "Please describe the issue."})

        # --- TEST HOOK: FORCE IMAGE TEST ---
        if user_message and user_message.lower().strip() == "force test images":
            try:
                logging.info("TEST HOOK: Listing bucket for test...")
                storage_client = gcs.Client()
                blobs = list(storage_client.list_blobs("manualai02a", max_results=5))
                pdf_blob = next((b for b in blobs if b.name.endswith(".pdf")), None)
                
                if not pdf_blob:
                     return jsonify({'solution': "TEST FAIL: No PDFs found in manualai02a bucket."})
                
                test_uri = f"gs://manualai02a/{pdf_blob.name}"
                test_title = pdf_blob.name
                
                # Create Mock Tutorial
                mock_data = {
                    "title": f"TEST TUTORIAL: {test_title}",
                    "topic_slug": "test_image_extraction",
                    "intro": f"This is a FORCED TEST to verify image extraction from: {test_uri}.",
                    "steps": [
                        {
                            "step_number": 1,
                            "instruction": "Step 1 instruction",
                            "page_number": 1,
                            "has_visual": True
                        }
                    ]
                }
                
                final_data = extract_and_upload_images(mock_data, test_uri)
                return jsonify({'tutorial': final_data})
            except Exception as test_e:
                logging.error(f"TEST HOOK FAILED: {test_e}")
                return jsonify({'solution': f"TEST HOOK FAILED: {str(test_e)}"})
        # -----------------------------------

        # --- ROUTER LOGIC ---
        history_str = request.form.get('history', '[]')
        history = []
        try:
             history = json.loads(history_str)
        except:
             pass

        is_confirmation = False
        if not image_file and user_message.lower().strip() in ["yes", "y", "ok", "sure", "go ahead", "please", "generate", "confirm"]:
             is_confirmation = True
        
        final_query = user_message
        if is_confirmation and history:
             for msg in reversed(history):
                  if msg.get('role') == 'user':
                       content = msg.get('parts', [{}])[0].get('text', '').lower()
                       if content not in ["yes", "y", "ok", "sure", "go ahead", "please", "generate", "confirm"]:
                            final_query = msg.get('parts', [{}])[0].get('text', '')
                            break
        
        if image_file:
             final_query = analyze_image_for_query(image_file, user_message)

        # Grounding Tool Setup
        grounding_tool = Tool.from_retrieval(
            grounding.Retrieval(
                source=grounding.VertexAISearch(
                    datastore=f"projects/{PROJECT_ID}/locations/{DATA_STORE_REGION}/collections/default_collection/dataStores/{DATA_STORE_ID}"
                )
            )
        )

        # --- EXECUTION MODE ---
        if is_confirmation or image_file:
            logging.info(f"Mode: GENERATION for Query: {final_query}")
            
            # 1. Smart Cache Check
            canonical_slug = generate_canonical_slug(final_query)
            cached_tutorial = get_tutorial_from_db(canonical_slug)
            if cached_tutorial:
                logging.info(f"Cache HIT for {canonical_slug}")
                return jsonify({'tutorial': cached_tutorial})
            
            # 2. Generation
            gen_model = GenerativeModel(
                "gemini-2.5-pro",
                system_instruction=GENERATOR_SYSTEM_INSTRUCTION,
                tools=[grounding_tool]
            )
            
            # Tool Config for attribution (Safe wrapper)
            tool_config = None
            try:
                tool_config = ToolConfig(retrieval_config={"disable_attribution": False})
            except Exception as e:
                logging.warning(f"ToolConfig initialization failed (ignoring): {e}")
            
            gen_res = gen_model.generate_content(
                f"Search manuals and generate tutorial for: {final_query}",
                generation_config={"temperature": 0.1},
                tool_config=tool_config
            )
            
            response_text = extract_text_from_gemini(gen_res)
            
            try:
                # Handle multiple JSONs if necessary
                if "} {" in response_text:
                     response_text = response_text.split("} {")[0] + "}"
                
                tutorial_data = json.loads(response_text)
                tutorial_data['topic_slug'] = canonical_slug
                tutorial_data['intro'] = tutorial_data.get('intro', '') + f" (ID: {canonical_slug})"

                # 3. Metadata & Image Extraction
                source_uri = ""
                if gen_res.candidates and gen_res.candidates[0].grounding_metadata:
                    gm = gen_res.candidates[0].grounding_metadata
                    if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
                        for chunk in gm.grounding_chunks:
                            if chunk.retrieved_context and chunk.retrieved_context.uri:
                                source_uri = chunk.retrieved_context.uri
                                break
                
                if source_uri:
                    if source_uri.startswith("https://storage.googleapis.com/"):
                        source_uri = source_uri.replace("https://storage.googleapis.com/", "gs://")
                    tutorial_data = extract_and_upload_images(tutorial_data, source_uri)
                
                # 4. Save & Return
                save_tutorial_to_db(canonical_slug, tutorial_data)
                return jsonify({'tutorial': tutorial_data})

            except Exception as e:
                logging.error(f"Generation parsing failure: {e}")
                return jsonify({'solution': "I found the solution but had trouble formatting it. Please try asking again more specifically."})

        else:
            # PLANNING MODE (Text)
            logging.info(f"Mode: PLANNING for Query: {final_query}")
            plan_model = GenerativeModel(
                "gemini-2.5-pro",
                system_instruction=PLANNING_SYSTEM_INSTRUCTION,
                tools=[grounding_tool]
            )
            
            plan_res = plan_model.generate_content(
                f"Check manuals for: {final_query}",
                generation_config={"temperature": 0.3}
            )
            
            plan_text = extract_text_from_gemini(plan_res)
            return jsonify({'solution': plan_text})

    except Exception as e:
        logging.error(f"Global Analyze Error: {e}")
        return jsonify({'error': str(e)}), 500

@https_fn.on_request(memory=1024, timeout_sec=60, region="us-central1")
def api(req: https_fn.Request) -> Response:
    with app.request_context(req.environ):
        return app.full_dispatch_request()

