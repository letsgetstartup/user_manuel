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
        for i, page in enumerate(reader.pages):
            text += f"--- [PAGE_INDEX: {i}] ---\n"
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
        page_index = int(page_num)
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
        
        # Find the correct PDF filename for this machine
        current_pdf = None
        if machine_id:
            all_pdfs = [f for f in os.listdir(MANUALS_DIR) if f.lower().endswith('.pdf')]
            for f in all_pdfs:
                if machine_id in f.replace(".pdf", "").replace(" ", "_").lower():
                    current_pdf = f
                    break
        
        # --- PHASE 2: GENERATE STRUCTURED TUTORIAL ---
        prompt = f"""
        Based on the CONVERSATION HISTORY below and the manual, create a detailed step-by-step tutorial TO SOLVE THE USER'S SPECIFIC ISSUE.
        
        CONVERSATION HISTORY:
        {conversation_context}
        
        USER'S PROBLEM:
        The user is likely asking for help with the issue discussed above. Do NOT generate a generic setup guide unless asked.
        Address the specific potential causes mentions in the assistant's previous analysis.
        
        VISUAL RULES:
        - SET "has_visual": true if the step refers to a physical component, action, or setting that has a corresponding DIAGRAM or illustration on that page (even if there is also text).
        - ONLY set "has_visual": false if the page is literally ONLY blocks of text with no icons or diagrams.
        - You MUST provide the exact "pdf_page_reference" using the technical `PAGE_INDEX` found in the context markers (e.g. if the info is near `--- [PAGE_INDEX: 9] ---`, the reference is 9).
        - IMPORTANT: Use the filename "{current_pdf or 'manual.pdf'}" for the "pdf_filename" field.
        - IMPORTANT: In the "instruction" text, include the PAGE_INDEX in brackets at the end, e.g., "Rotate the Steam Dial clockwise [PAGE_INDEX: 10]".
        
        MANUAL CONTENT:
        {relevant_context[:500000]}
        
        JSON SCHEMA:
        {{
          "title": "Tutorial Title",
          "steps": [
            {{
              "step_number": 1,
              "instruction": "Short instruction [PAGE_INDEX: X]",
              "has_visual": true,
              "pdf_filename": "{current_pdf or 'manual.pdf'}",
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
            raw_data = json.loads(response.text)
            
            # Robust extraction of 'steps'
            tutorial_data = {}
            flat_steps = []

            def find_steps_recursive(data):
                if isinstance(data, list):
                    if data and isinstance(data[0], dict) and "instruction" in data[0]:
                        return data
                    for item in data:
                        res = find_steps_recursive(item)
                        if res: return res
                elif isinstance(data, dict):
                    if "steps" in data and isinstance(data["steps"], list) and data["steps"]:
                        if "instruction" in data["steps"][0]:
                            return data["steps"]
                        else:
                            return find_steps_recursive(data["steps"])
                    for v in data.values():
                        res = find_steps_recursive(v)
                        if res: return res
                return None

            flat_steps = find_steps_recursive(raw_data)
            
            if not flat_steps:
                if isinstance(raw_data, list):
                    flat_steps = raw_data
                elif isinstance(raw_data, dict) and "steps" in raw_data:
                    flat_steps = raw_data["steps"]

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

            # Process visuals for steps
            visual_count = 0
            for step in tutorial_data["steps"]:
                # Fallback: if filename is missing but has_visual is true, use current_pdf
                if step.get("has_visual") and not step.get("pdf_filename"):
                    step["pdf_filename"] = current_pdf
                
                if step.get("has_visual") and step.get("pdf_filename") and step.get("pdf_page_reference"):
                    if visual_count >= 10: break
                    visual_count += 1
                    url = extract_page_image(step["pdf_filename"], step["pdf_page_reference"], machine_id=machine_id)
                    if not url:
                        print(f"DEBUG: Extraction failed for step {step.get('step_number')}")
                        step["has_visual"] = False
                        continue
                    
                    try:
                        img_response = requests.get(url, timeout=10)
                        if img_response.status_code == 200:
                            original_img_bytes = img_response.content
                            vision_prompt = f"Analyze this manual page and find the diagram for: '{step['instruction']}'. Return JSON: {{\"box_2d\": [ymin, xmin, ymax, xmax], \"visual_utility_score\": 0.0-1.0}}."
                            vision_response = client.models.generate_content(
                                model="gemini-2.0-flash",
                                contents=[types.Part.from_bytes(data=original_img_bytes, mime_type="image/png"), vision_prompt],
                                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
                            )
                            coords = json.loads(vision_response.text)
                            # Handle Gemini returning a list of detections
                            if isinstance(coords, list) and len(coords) > 0:
                                coords = coords[0]
                            
                            print(f"DEBUG: Vision response for visual extraction: {coords}")
                            if isinstance(coords, dict) and coords.get("visual_utility_score", 1.0) >= 0.5:
                                box = coords.get('box_2d') or [0, 0, 1000, 1000]
                                img = Image.open(io.BytesIO(original_img_bytes))
                                w, h = img.size
                                left, top, right, bottom = box[1]*w/1000, box[0]*h/1000, box[3]*w/1000, box[2]*h/1000
                                # Add padding and crop
                                pad_w, pad_h = (right-left)*0.2, (bottom-top)*0.2
                                cropped_img = img.crop((max(0, left-pad_w), max(0, top-pad_h), min(w, right+pad_w), min(h, bottom+pad_h)))
                                
                                # Add enhanced Nano Banana yellow indicator
                                draw = ImageDraw.Draw(cropped_img)
                                target_x, target_y = ((box[1]+box[3])/2*w/1000) - max(0, left-pad_w), ((box[0]+box[2])/2*h/1000) - max(0, top-pad_h)
                                
                                # Draw a thick yellow ring
                                r1, r2 = 20, 25
                                draw.ellipse([target_x-r2, target_y-r2, target_x+r2, target_y+r2], fill="yellow", outline="black")
                                draw.ellipse([target_x-r1, target_y-r1, target_x+r1, target_y+r1], fill=None, outline="black")
                                
                                # Add the pointing finger emoji
                                try:
                                    draw.text((target_x + 30, target_y - 15), "👉", fill="yellow", stroke_width=2, stroke_fill="black")
                                except:
                                    # Fallback if emoji rendering fails
                                    draw.polygon([target_x+30, target_y, target_x+50, target_y-10, target_x+50, target_y+10], fill="yellow", outline="black")
                                
                                buf = io.BytesIO()
                                cropped_img.save(buf, format='PNG')
                                bucket_name = f"{PROJECT_ID}.firebasestorage.app"
                                bucket = storage_client.bucket(bucket_name)
                                nano_path = f"nano_banana/{uuid.uuid4().hex[:8]}.png"
                                blob = bucket.blob(nano_path)
                                blob.upload_from_string(buf.getvalue(), content_type='image/png')
                                blob.make_public()
                                step["image_url"] = blob.public_url
                            else:
                                step["image_url"] = url
                    except Exception:
                        step["image_url"] = url
            
            return jsonify({'tutorial': tutorial_data})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        # --- PHASE 1: SUMMARY RESPONSE ---
        graph_context = ""
        if machine_id:
            try:
                searcher = GraphSearcher()
                graph_node = searcher.find_solution_node(machine_id, user_message)
                if graph_node:
                    graph_context = f"GRAPH MATCH: {graph_node.get('label')} - {graph_node.get('text')}"
            except Exception as e:
                print(f"Graph Search Error: {e}")

        prompt = f"""
        Analyze the user's request using the Manual Content AND Graph info.
        1. Provide a clear point-based answer.
        2. Cite page references using the technical marker format found in the text, e.g., [PAGE_INDEX: 9].
        3. END by asking if they want a step-by-step tutorial with diagrams.
        
        GRAPH INFO: {graph_context}
        MANUAL: {relevant_context[:500000]}
        """
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=[prompt, user_message] + ([image_part] if image_part else []))
            return jsonify({'solution': response.text})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

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
