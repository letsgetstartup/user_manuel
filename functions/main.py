import os
import json
import io
import base64
import glob
import requests
from flask import Flask, render_template, request, jsonify
from firebase_functions import https_fn, storage_fn
from werkzeug.wrappers import Response
from google.cloud import storage, firestore
from google import genai
from google.genai import types
from pypdf import PdfReader

# --- CONFIGURATION ---
PROJECT_ID = os.environ.get("GCP_PROJECT", "manualai-481406")
LOCATION = "us-central1"
# Switched to gemini-2.0-flash as it is verified accessible in this project
MODEL_NAME = "gemini-2.5-pro" 
EMBEDDING_MODEL = "text-embedding-004"
MANUALS_DIR = "manuals"

# Vertex AI REST Endpoint (Legacy API)
API_KEY = "AQ.Ab8RN6LRdzDSFrtZ1BY3_3WMkV39SBGqiFo3mo2eRbJD4Ib9Tg"
ENDPOINT_URL = f"https://{LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/gemini-2.5-pro:generateContent"

# --- CLIENTS ---
db = firestore.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

# --- FLASK APP ---
app = Flask(__name__)
MANUALS_CONTEXT = ""

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
    global MANUALS_CONTEXT
    if not os.path.exists(MANUALS_DIR):
        print(f"Warning: {MANUALS_DIR} not found.")
        return
    manual_files = glob.glob(os.path.join(MANUALS_DIR, "*.pdf"))
    context = ""
    for manual_path in manual_files:
        text = extract_text_from_pdf(manual_path)
        if text:
            context += f"\n\n--- Content from {os.path.basename(manual_path)} ---\n{text}"
    MANUALS_CONTEXT = context
    print(f"Loaded {len(MANUALS_CONTEXT)} chars of context.")

# Load once on startup
load_manuals()

import fitz
from PIL import Image

def extract_page_image(pdf_filename, page_num):
    """Extracts a page from a local PDF as an image and uploads to Storage."""
    try:
        pdf_path = os.path.join(MANUALS_DIR, pdf_filename)
        if not os.path.exists(pdf_path):
            # Fallback to searching globally if directory structure differs
            print(f"PDF not found at {pdf_path}, searching...")
            return None
            
        doc = fitz.open(pdf_path)
        page_index = int(page_num) - 1
        if 0 <= page_index < len(doc):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            
            # Upload to Firebase Storage
            bucket_name = f"{PROJECT_ID}.firebasestorage.app"
            # Attempt to use specific bucket from environment or secrets if needed
            # For this hack, we'll try to find the default or provided one
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"generated_visuals/{pdf_filename}_p{page_num}_{uuid.uuid4().hex[:6]}.png"
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
    history_json = request.form.get('history', '[]')
    history = json.loads(history_json)
    
    file = request.files.get('image')
    image_part = None
    if file:
        img_bytes = file.read()
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

    # 1. Logic Check: Is this a confirmation for a tutorial?
    is_confirmation = any(word in user_message.lower() for word in ["yes", "go ahead", "ok", "generate"]) 
    # Also check if the last model message asked "Shall I generate..."
    last_model_msg = ""
    if history:
         for m in reversed(history):
             if m["role"] == "model":
                 last_model_msg = m["parts"][0].get("text", "").lower()
                 break
    
    if is_confirmation and ("generate" in last_model_msg or "tutorial" in last_model_msg):
        # --- PHASE 2: GENERATE STRUCTURED TUTORIAL ---
        prompt = f"""
        Based on the previous conversation and the manuals provided below, create a step-by-step tutorial.
        Identify the specific PDF filename and page numbers for visual diagrams.
        
        PDF CONTEXT:
        {MANUALS_CONTEXT[:500000]}
        
        Output MUST be JSON matching this schema:
        {{
          "title": "Clear Title",
          "intro": "Brief intro",
          "steps": [
            {{
              "step_number": 1,
              "instruction": "Do X",
              "has_visual": true,
              "pdf_filename": "name.pdf",
              "pdf_page_reference": 10
            }}
          ]
        }}
        """
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2
                )
            )
            tutorial_data = json.loads(response.text)
            
            # Extract images for steps that have visuals
            for step in tutorial_data.get("steps", []):
                if step.get("has_visual") and step.get("pdf_filename") and step.get("pdf_page_reference"):
                    url = extract_page_image(step["pdf_filename"], step["pdf_page_reference"])
                    if url:
                        step["image_url"] = url
            
            return jsonify({'tutorial': tutorial_data})
        except Exception as e:
            return jsonify({'error': f"Tutorial Gen Error: {str(e)}"}), 500

    else:
        # --- PHASE 1: SUMMARY RESPONSE + CONFIRMATION ---
        prompt = f"""
        You are a Technical Support assistant. 
        Analyze the user's request (text and/or image) using the Manuals Context provided.
        Provide a concise answer and end by asking if they want a step-by-step tutorial with diagrams.
        
        MANUALS CONTEXT:
        {MANUALS_CONTEXT[:500000]}
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

@https_fn.on_request(memory=1024, timeout_sec=300, region="us-central1")
def api(req: https_fn.Request) -> Response:
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
