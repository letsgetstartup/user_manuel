
import fitz # PyMuPDF
import os
import io
from PIL import Image
import base64

MANUALS_DIR = "functions/manuals"
PDF_NAME = "Baseline_User-Manual-of-Network-Video-Recorder_71-E-M_V3.4.90_20180124.pdf"
pdf_path = os.path.join(MANUALS_DIR, PDF_NAME)

def test_nano_banana_flow():
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print(f"Step 1: Extracting image from page 38...")
    doc = fitz.open(pdf_path)
    page = doc.load_page(37) # Page 38 is index 37
    
    # Render page as image (where the diagram is)
    # The user's screenshot showed the diagram in the middle.
    pix = page.get_pixmap(dpi=300)
    img_data = pix.tobytes("png")
    
    # Save the raw extraction for reference
    with open("raw_extraction_p38.png", "wb") as f:
        f.write(img_data)
    print("Saved raw_extraction_p38.png")

    print("\nStep 2: Preparing Nano Banana Prompt...")
    NANO_BANANA_PROMPT = """
    Create a copy of the attached image based on the following prompt:
    Role & Goal: You are UI Engineer. Your goal is to create a 1:1 pixel-perfect replica of the attached image. 
    You must prioritize data accuracy and structural integrity over artistic interpretation.

    Replicate the exact window title: "IP Camera Management Interface"
    Exact Labels & Values (Row-by-Row):
    - D1: Camera 01, Strong Pas, 10.16.1.250
    - D2: chan2, Strong Pas, 10.16.1.102
    - D3: Camera 01, Weak Pass.., 10.21.133.118
    ...and so on.

    Icons - copy exactly where they appear in the source.
    DO NOT use placeholder text. DO NOT generalize numbers.
    Final Quality Check: Every number, dot, icon and line must match.
    """
    print(NANO_BANANA_PROMPT)

if __name__ == "__main__":
    test_nano_banana_flow()
