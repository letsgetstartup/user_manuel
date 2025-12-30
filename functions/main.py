import os
import json
import io
import base64
import glob
import requests
import uuid
import fitz
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify
from flask_cors import CORS
from firebase_functions import https_fn, storage_fn
from werkzeug.wrappers import Response
from google.cloud import storage
from google.cloud import firestore
from google import genai
from google.genai import types
from pypdf import PdfReader

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "manualai-481406")
LOCATION = "us-central1"
# Switched to gemini-2.0-flash for stability
MODEL_NAME = "gemini-2.0-flash" 
EMBEDDING_MODEL = "text-embedding-004"
MANUALS_DIR = "manuals"

# Vertex AI REST Endpoint (Legacy API)
API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
ENDPOINT_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.5-pro:generateContent"

from flask_cors import CORS

# --- CLIENTS ---
db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- FLASK APP ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes
MANUALS_CACHE = {} # machine_id -> text content

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def load_manuals():
    global MANUALS_CACHE
    if not os.path.exists(MANUALS_DIR):
        print(f"Warning: {MANUALS_DIR} not found.")
        return
    manual_files = glob.glob(os.path.join(MANUALS_DIR, "*.pdf"))
    for manual_path in manual_files:
        filename = os.path.basename(manual_path)
        machine_id = filename.replace(".pdf", "").replace(" ", "_").lower()
        text = extract_text_from_pdf(manual_path)
        if text:
            MANUALS_CACHE[machine_id] = text
            print(f"Loaded manual for: {machine_id}")

# Load once on startup
# Load once on startup - REMOVED for Lazy Loading
# load_manuals()


def extract_page_image(pdf_filename, page_num, machine_id=None):
    """Extracts a page from a local PDF as an image and uploads to Storage."""
    try:
        pdf_path = None
        
        # 1. Try machine_id lookup
        if machine_id:
            all_pdfs = [f for f in os.listdir(MANUALS_DIR) if f.lower().endswith('.pdf')]
            for f in all_pdfs:
                # Allow partial match (e.g., 'bes875' in 'bes875-instruction-manual')
                name_clean = f.replace(".pdf", "").replace(" ", "-").lower()
                if machine_id in name_clean:
                    pdf_path = os.path.join(MANUALS_DIR, f)
                    print(f"DEBUG: Found PDF via machine_id: {pdf_path}")
                    break
        
        # 2. Try exact filename
        if not pdf_path:
            test_path = os.path.join(MANUALS_DIR, pdf_filename)
            if os.path.exists(test_path):
                pdf_path = test_path
                print(f"DEBUG: Found PDF via exact name: {pdf_path}")

        # 3. Try fuzzy matching
        if not pdf_path:
            print(f"DEBUG: PDF not found at {pdf_filename}, fuzzy searching...")
            all_pdfs = [f for f in os.listdir(MANUALS_DIR) if f.lower().endswith('.pdf')]
            best_match = None
            # Simple word-based match
            search_words = pdf_filename.lower().replace("-", " ").replace(".", " ").split()
            for f in all_pdfs:
                f_words = f.lower().replace("-", " ").replace(".", " ").split()
                # If first 2 words match, it's likely the same
                if len(search_words) >= 2 and len(f_words) >= 2:
                    if search_words[0] == f_words[0] and search_words[1] == f_words[1]:
                        best_match = f
                        break
            
            if best_match:
                print(f"DEBUG: Found fuzzy match: {best_match}")
                pdf_path = os.path.join(MANUALS_DIR, best_match)
        
        if not pdf_path:
            print(f"ERROR: Could not resolve PDF for {pdf_filename}")
            return None
            
        doc = fitz.open(pdf_path)
        page_index = int(page_num) - 1
        if 0 <= page_index < len(doc):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Upload to Firebase Storage
            bucket_name = f"{PROJECT_ID}.firebasestorage.app"
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"generated_visuals/{os.path.basename(pdf_path)}_p{page_num}_{uuid.uuid4().hex[:6]}.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(img_bytes, content_type='image/png')
            blob.make_public()
            return blob.public_url
    except Exception as e:
        print(f"Image extraction error: {e}")
    return None

import uuid

@app.route('/api/analyze', methods=['POST'])
def analyze():
    user_message = request.form.get('message', '')
    machine_id = request.form.get('machine_id', '')
    
    # Lazy load manuals if cache is empty
    if not MANUALS_CACHE:
        print("DEBUG: Lazy loading manuals...")
        load_manuals()
    print(f"DEBUG: analyze called with message: {user_message}, machine: {machine_id}")
    history_json = request.form.get('history', '[]')
    history = json.loads(history_json)
    
    file = request.files.get('image')
    image_part = None
    if file:
        print("DEBUG: image received")
        img_bytes = file.read()
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    # Get specific context for this machine
    relevant_context = MANUALS_CACHE.get(machine_id, "")
    if not relevant_context and machine_id:
        print(f"WARNING: No manual found in cache for {machine_id}")

    # 1. Logic Check: Is this a confirmation for a tutorial?
    is_confirmation = any(word in user_message.lower() for word in ["yes", "go ahead", "ok", "generate", "sure"]) 
    
    last_assistant_msg = ""
    if history:
        for m in reversed(history):
            role = m.get("role", "").lower()
            if role in ["model", "assistant"]:
                if "content" in m:
                    last_assistant_msg = m["content"].lower()
                elif "parts" in m and m["parts"]:
                    last_assistant_msg = m["parts"][0].get("text", "").lower()
                break
    
    # Define Nano Banana Prompt
    NANO_BANANA_PROMPT = """You are a UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image.
    VISUAL INDICATOR: Add a clearly visible "pointing finger" emoji or graphic (👉) pointing EXACTLY at the UI element or part mentioned in the instruction.
    Instruction: {instruction}
    The finger should be placed so it doesn't obscure text but clearly directs the user's eye."""

    print(f"DEBUG: is_confirmation: {is_confirmation}, last_assistant_msg: {last_assistant_msg}")

    if is_confirmation and ("generate" in last_assistant_msg or "tutorial" in last_assistant_msg or "diagrams" in last_assistant_msg):
        print("DEBUG: Entering Tutorial Generation Phase")
        
        # Construct conversation context from history
        conversation_context = ""
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or (msg.get("parts", [{}])[0].get("text", "") if msg.get("parts") else "")
            conversation_context += f"{role.upper()}: {content}\n"
        
        # --- PHASE 2: GENERATE STRUCTURED TUTORIAL ---
        prompt = f"""
        Based on the CONVERSATION HISTORY below and the manual, create a detailed step-by-step tutorial TO SOLVE THE USER'S SPECIFIC ISSUE.
        
        CONVERSATION HISTORY:
        {conversation_context}
        
        USER'S PROBLEM:
        The user is likely asking for help with the issue discussed above. Do NOT generate a generic setup guide unless asked.
        Address the specific potential causes mentions in the assistant's previous analysis.
        
        VISUAL RULES:
        - ONLY set "has_visual": true if the step refers to a specific UI element, dial, button, or structural component shown in a DIAGRAM, ILLUSTRATION, or PICTURE.
        - If the page is primarily blocks of text, set "has_visual": false. We want diagrams, not text screenshots.
        - You MUST provide the exact "pdf_page_reference" (e.g. 7, 12, 24).
        - IMPORTANT: In the "instruction" text, always include a specific page reference in brackets at the end, e.g., "Rotate the Steam Dial clockwise [Page 10]".
        
        MANUAL CONTENT:
        {relevant_context[:500000]}
        
        JSON SCHEMA:
        {{
          "title": "Tutorial Title",
          "steps": [
            {{
              "step_number": 1,
              "instruction": "Short instruction [Page X]",
              "has_visual": true,
              "pdf_filename": "name.pdf",
              "pdf_page_reference": 10
            }}
          ]
        }}
        """
        try:
            print("DEBUG: Requesting structured JSON from Gemini...")
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            try:
                raw_data = json.loads(response.text)
                print(f"DEBUG: Gemini raw response: {response.text}")
                
                # Robust extraction of 'steps'
                tutorial_data = {}
                flat_steps = []

                def find_steps_recursive(data):
                    if isinstance(data, list):
                        # If it's a list of objects with instructions, we've found it
                        if data and isinstance(data[0], dict) and "instruction" in data[0]:
                            return data
                        for item in data:
                            res = find_steps_recursive(item)
                            if res: return res
                    elif isinstance(data, dict):
                        if "steps" in data and isinstance(data["steps"], list) and data["steps"]:
                            # Check if the nested steps have instructions
                            if "instruction" in data["steps"][0]:
                                return data["steps"]
                            else:
                                # Keep digging
                                return find_steps_recursive(data["steps"])
                        for v in data.values():
                            res = find_steps_recursive(v)
                            if res: return res
                    return None

                flat_steps = find_steps_recursive(raw_data)
                
                if not flat_steps:
                    # Fallback: if Gemini returned a list directly but no instruction key
                    if isinstance(raw_data, list):
                        flat_steps = raw_data
                    elif isinstance(raw_data, dict) and "steps" in raw_data:
                        flat_steps = raw_data["steps"]

                # Final sanitization: ensure each step has a number and is a dict
                sanitized_steps = []
                for i, s in enumerate(flat_steps or []):
                    if isinstance(s, dict):
                        s.setdefault("step_number", i + 1)
                        if "instruction" in s:
                            sanitized_steps.append(s)
                    elif isinstance(s, str):
                        sanitized_steps.append({
                            "step_number": i + 1,
                            "instruction": s,
                            "has_visual": False
                        })
                
                tutorial_data["steps"] = sanitized_steps
                tutorial_data.setdefault("title", "Tutorial")
                tutorial_data.setdefault("intro", "")
                
            except Exception as e:
                print(f"ERROR: JSON loads/parsing failed: {e}")
                return jsonify({'error': f"JSON Parse Error: {str(e)}"}), 500

            steps = tutorial_data.get("steps", [])
            print(f"DEBUG: Final sanitized tutorial contains {len(steps)} steps")
            
            # Extract images for steps that have visuals (limit to 10 to avoid timeout)
            visual_count = 0
            for step in steps:
                if step.get("has_visual") and step.get("pdf_filename") and step.get("pdf_page_reference"):
                    if visual_count >= 10:
                        # Fallback for steps beyond the limit: show manual reference only
                        print(f"DEBUG: Visual limit (10) reached for step {step['step_number']}")
                        continue
                        
                    print(f"DEBUG: Processing visual for step {step['step_number']} (Page {step['pdf_page_reference']})")
                    visual_count += 1
                    
                    # Extract original page
                    url = extract_page_image(step["pdf_filename"], step["pdf_page_reference"], machine_id=machine_id)
                    
                    if not url:
                        print(f"DEBUG: No image extracted for step {step['step_number']}, disabling visual.")
                        step["has_visual"] = False
                        continue

                    print(f"DEBUG: Extracted image URL: {url}")
                    # Apply Nano Banana (Visual detection and cropping)
                    try:
                        # Fetch original image bytes
                        img_response = requests.get(url, timeout=10)
                        if img_response.status_code == 200:
                            original_img_bytes = img_response.content
                                
                            # 1. Detect coordinates using Gemini Vision
                            print(f"DEBUG: Isolating diagram for: {step['instruction']}")
                            vision_prompt = f"""
                            Analyze this technical manual page and identify the SPECIFIC DIAGRAM, ILLUSTRATION, or CLARIFYING IMAGE related to this instruction: '{step['instruction']}'.
                            
                            RULES:
                            1. If the page is just a wall of text with no clear diagram for this specific instruction, return {{"skip": true, "reason": "text_heavy"}}.
                            2. If there IS a diagram (a drawing of the machine, a button, a cross-section), identify the bounding box of the WHOLE diagram area.
                            3. Return a JSON object: 
                               - If found: {{"box_2d": [ymin, xmin, ymax, xmax], "label": "component_name", "visual_utility_score": 0.0-1.0}}
                               - The visual_utility_score should be LOW (0.1-0.4) if the area is mostly text, and HIGH (0.7-1.0) if it is a clear graphic.
                               - If visual_utility_score < 0.5, we will discard this image.
                            Coordinates: 0-1000. JSON ONLY.
                            """
                            
                            vision_response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=[
                                    types.Part.from_bytes(data=original_img_bytes, mime_type="image/png"),
                                    vision_prompt
                                ],
                                config=types.GenerateContentConfig(
                                    response_mime_type="application/json",
                                    temperature=0
                                )
                            )
                            
                            raw_coords = json.loads(vision_response.text)
                            print(f"DEBUG: Detected raw coordinates: {raw_coords}")
                            
                            # Visual Utility Gate
                            if raw_coords.get("skip") or raw_coords.get("visual_utility_score", 1.0) < 0.5:
                                print(f"DEBUG: Skipping visual for step {step['step_number']} due to low utility: {raw_coords}")
                                step["has_visual"] = False
                                step.pop("image_url", None)
                                continue

                            # Handle multiple formats: [{...}], {...}, or [...]
                            if isinstance(raw_coords, list) and len(raw_coords) > 0 and isinstance(raw_coords[0], dict):
                                coords = raw_coords[0]
                            else:
                                coords = raw_coords

                            if isinstance(coords, dict):
                                box = coords.get('box_2d') or coords.get('coordinates') or [coords.get('ymin', 0), coords.get('xmin', 0), coords.get('ymax', 1000), coords.get('xmax', 1000)]
                            else:
                                box = coords if isinstance(coords, list) and len(coords) == 4 else [0, 0, 1000, 1000]

                            # 2. Process with Pillow
                            img = Image.open(io.BytesIO(original_img_bytes))
                            w, h = img.size
                            
                            # Convert normalized to pixel
                            ymin, xmin, ymax, xmax = box
                            left = xmin * w / 1000
                            top = ymin * h / 1000
                            right = xmax * w / 1000
                            bottom = ymax * h / 1000
                            
                            # Add 20% padding
                            pad_w = (right - left) * 0.2
                            pad_h = (bottom - top) * 0.2
                            left = max(0, left - pad_w)
                            top = max(0, top - pad_h)
                            right = min(w, right + pad_w)
                            bottom = min(h, bottom + pad_h)
                            
                            # Crop
                            cropped_img = img.crop((left, top, right, bottom))
                            cw, ch = cropped_img.size
                            
                            # 3. Add Pointing Finger (👉)
                            # We'll just paste a text emoji if font is available, or draw a simple arrow
                            draw = ImageDraw.Draw(cropped_img)
                            # Attempt to draw a bright yellow circle/arrow at the target (center of detected box relative to crop)
                            target_x = ( ( (xmin + xmax)/2 * w / 1000 ) - left )
                            target_y = ( ( (ymin + ymax)/2 * h / 1000 ) - top )
                            
                            # Drawing a "Nano Banana" yellow indicator
                            r = 15
                            draw.ellipse([target_x-r, target_y-r, target_x+r, target_y+r], outline="yellow", width=5)
                            draw.text((target_x + 20, target_y), "👉", fill="yellow")

                            # 4. Upload processed image
                            processed_byte_arr = io.BytesIO()
                            cropped_img.save(processed_byte_arr, format='PNG')
                            processed_bytes = processed_byte_arr.getvalue()
                            
                            bucket_name = f"{PROJECT_ID}.firebasestorage.app"
                            bucket = storage_client.bucket(bucket_name)
                            nano_path = f"nano_banana/{uuid.uuid4().hex[:8]}.png"
                            nano_blob = bucket.blob(nano_path)
                            nano_blob.upload_from_string(processed_bytes, content_type='image/png')
                            nano_blob.make_public()
                            
                            step["image_url"] = nano_blob.public_url
                            print(f"DEBUG: Nano Banana success! URL: {step['image_url']}")

                    except Exception as ne:
                        print(f"ERROR: Nano Banana processing failed: {ne}")
                        step["image_url"] = url # Fallback to full page if vision/crop fails
            
            return jsonify({'tutorial': tutorial_data})
        except Exception as e:
            print(f"ERROR: Tutorial Gen Failed: {e}")
            return jsonify({'error': f"Tutorial Gen Error: {str(e)}"}), 500
    else:
        print("DEBUG: Status: Summary Response Phase")
        # --- PHASE 1: SUMMARY RESPONSE + CONFIRMATION ---
        prompt = f"""
        You are a Technical Support assistant. 
        Analyze the user's request (text and/or image) using the Manual Content provided.
        - Be EXTREMELY precise. If troubleshooting, mention specific steps or parts (e.g., "check for a blocked filter basket").
        - Cite specific page numbers from the manual for every piece of advice (e.g., "[Page 29]").
        - Use the provided context to offer the most accurate solution.
        - End by asking if they want a step-by-step tutorial with diagrams.
        
        MANUAL CONTENT:
        {relevant_context[:500000]}
        """
        
        contents = [prompt]
        if user_message:
            contents.append(user_message)
        if image_part:
            contents.append(image_part)
            
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.4)
            )
            return jsonify({'solution': response.text})
        except Exception as e:
            return jsonify({'error': f"Analysis Error: {str(e)}"}), 500

from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

# ... (Existing code)

class GraphSearcher:
    def __init__(self):
        # reuse global db and client
        self.db = db 
        self.client = client
        self.embedding_model = EMBEDDING_MODEL

    def embed_query(self, text):
        """Turns user question into a vector."""
        result = self.client.models.embed_content(
            model=self.embedding_model,
            contents=text,
        )
        return result.embeddings[0].values

    def find_solution_node(self, machine_id, user_query):
        """Finds the most relevant starting node in the graph using Vector Search."""
        try:
            query_vector = self.embed_query(user_query)
            
            # Reference to the nodes collection for the specific machine
            nodes_ref = self.db.collection("machines").document(machine_id).collection("graph_nodes")
            
            # Perform Vector Search (Nearest Neighbor)
            vector_query = nodes_ref.find_nearest(
                vector_field="embedding_field",
                query_vector=Vector(query_vector),
                distance_measure=DistanceMeasure.COSINE,
                limit=1
            )
            
            results = vector_query.get()
            
            if not results:
                return None

            best_node = results[0]
            node_data = best_node.to_dict()
            
            # Fetch connected edges
            edges_ref = self.db.collection("machines").document(machine_id).collection("graph_edges")
            next_steps = edges_ref.where("from", "==", node_data['id']).stream()
            node_data['next_possible_steps'] = [edge.to_dict() for edge in next_steps]
            
            return node_data
        except Exception as e:
            print(f"Graph Search Error: {e}")
            return None

@app.route('/api/search_graph', methods=['POST'])
def search_graph():
    machine_id = request.json.get('machine_id')
    query = request.json.get('query')
    
    if not machine_id or not query:
        return jsonify({'error': 'Missing machine_id or query'}), 400
        
    searcher = GraphSearcher()
    result = searcher.find_solution_node(machine_id, query)
    
    if result:
        return jsonify({'result': result})
    else:
        return jsonify({'message': 'No matching solution found.'}), 404

@https_fn.on_request(memory=1024, timeout_sec=300, region="us-central1")
def api(req: https_fn.Request):
    with app.request_context(req.environ):
        return app.full_dispatch_request()

# --- ENTERPRISE INGESTION PIPELINE (KEEPING THIS FOR BACKGROUND PROCESSING) ---

GRAPH_EXTRACTION_PROMPT = """
You are an expert technical engineer for Otopia. 
Analyze the provided instruction manual PDF.
Your goal is to convert this manual into a Troubleshooting Knowledge Graph.

Output a JSON object with two arrays: "nodes" and "edges".

1. **NODES**: Represent a specific State, Symptom, Action, or Component.
   - `id`: Unique snake_case string (e.g., "water_tank_empty").
   - `label`: Short title of the node.
   - `text`: Detailed description of what to do or check.
   - `type`: One of ["symptom", "action", "decision", "component"].
   - `ar_anchor`: The physical part name for AR overlay (e.g., "steam_wand_tip", "power_button").

2. **EDGES**: Represent the flow.
   - `from`: Node ID.
   - `to`: Node ID.
   - `condition`: Logic for this transition (e.g., "If light is flashing", "If water flows").

CRITICAL: 
- Focus heavily on the "Troubleshooting" and "Maintenance" sections.
- Ensure the graph has a logical start and end for every problem.
"""

GRAPH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "nodes": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "id": {"type": "STRING"},
                    "label": {"type": "STRING"},
                    "text": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": ["symptom", "action", "decision", "component"]},
                    "ar_anchor": {"type": "STRING"}
                },
                "required": ["id", "label", "text", "type"]
            }
        },
        "edges": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "from": {"type": "STRING"},
                    "to": {"type": "STRING"},
                    "condition": {"type": "STRING"}
                },
                "required": ["from", "to"]
            }
        }
    },
    "required": ["nodes", "edges"]
}

def generate_embeddings(text_list):
    result = client.models.embed_content(model=EMBEDDING_MODEL, contents=text_list)
    return [e.values for e in result.embeddings]

@storage_fn.on_object_finalized(bucket="manualai02a", region="europe-west1")
def process_manual_ingestion(event, context=None):
    """Triggered by a change to a Cloud Storage object."""
    # Handle both CloudEvent (2nd gen) and (data, context) (1st gen/framework quirk)
    if context is not None:
        data = event
    else:
        data = event.data

    bucket_name = data["bucket"]
    file_name = data["name"]

    if not file_name.lower().endswith(".pdf"):
        return

    print(f"🚀 Processing Manual: {file_name}")
    machine_id = file_name.replace(".pdf", "").replace(" ", "_").lower()

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        pdf_bytes = blob.download_as_bytes()

        print("asking Gemini to build the graph structure...")
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                GRAPH_EXTRACTION_PROMPT,
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GRAPH_SCHEMA,
                temperature=0.0
            )
        )

        graph_data = json.loads(response.text)
        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        texts_to_embed = [f"{n['label']}: {n['text']}" for n in nodes]
        vectors = generate_embeddings(texts_to_embed) if texts_to_embed else []

        batch = db.batch()
        machine_ref = db.collection("machines").document(machine_id)
        batch.set(machine_ref, {
            "name": file_name,
            "processed_at": firestore.SERVER_TIMESTAMP,
            "status": "active"
        })

        for i, node in enumerate(nodes):
            node_ref = machine_ref.collection("graph_nodes").document(node['id'])
            node['embedding_field'] = vectors[i] if i < len(vectors) else []
            batch.set(node_ref, node)

        for edge in edges:
            edge_ref = machine_ref.collection("graph_edges").document()
            batch.set(edge_ref, edge)

        batch.commit()
        print(f"✅ Success! {machine_id} is live.")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        db.collection("ingestion_errors").document(machine_id).set({
            "machine_id": machine_id, "file_name": file_name, "error": str(e), "timestamp": firestore.SERVER_TIMESTAMP
        })
        raise e
