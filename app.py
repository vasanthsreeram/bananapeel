import os
import tempfile
import shutil
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, Response, session
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
from PIL import Image
import cv2
import pytesseract
import numpy as np
import json
from google import genai
from google.genai import types
from weasyprint import HTML, CSS
from PyPDF2 import PdfMerger
import threading

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-in-production')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

GEMINI_MODEL = "gemini-3-pro-preview"

# Store progress for each job
job_progress = {}
job_results = {}
job_lock = threading.Lock()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_surrounding_color(img, x, y, w, h, sample_width=10):
    """Get the dominant color surrounding a text region by sampling the border pixels."""
    img_h, img_w = img.shape[:2]

    samples = []

    # Sample from left edge
    left_x = max(0, x - sample_width)
    if left_x < x:
        left_region = img[max(0, y):min(img_h, y + h), left_x:x]
        if left_region.size > 0:
            samples.append(left_region.reshape(-1, 3))

    # Sample from right edge
    right_x = min(img_w, x + w + sample_width)
    if right_x > x + w:
        right_region = img[max(0, y):min(img_h, y + h), x + w:right_x]
        if right_region.size > 0:
            samples.append(right_region.reshape(-1, 3))

    # Sample from top edge
    top_y = max(0, y - sample_width)
    if top_y < y:
        top_region = img[top_y:y, max(0, x):min(img_w, x + w)]
        if top_region.size > 0:
            samples.append(top_region.reshape(-1, 3))

    # Sample from bottom edge
    bottom_y = min(img_h, y + h + sample_width)
    if bottom_y > y + h:
        bottom_region = img[y + h:bottom_y, max(0, x):min(img_w, x + w)]
        if bottom_region.size > 0:
            samples.append(bottom_region.reshape(-1, 3))

    if samples:
        all_samples = np.vstack(samples)
        return np.median(all_samples, axis=0).astype(int)

    # Fallback to global background if no samples
    return img[10:50, 10:50].mean(axis=(0, 1)).astype(int)


def remove_text_from_slide(image_path: str, output_path: str, gemini_client, layout_data_out: dict):
    """Remove all text from a slide image using OCR detection and masking."""
    img = cv2.imread(image_path)

    layout_data = {
        "width": img.shape[1],
        "height": img.shape[0],
        "background_color": None,
        "graphics": []
    }

    # Get global background color for layout data
    bg_color = img[10:50, 10:50].mean(axis=(0, 1)).astype(int)
    layout_data["background_color"] = f"rgb({int(bg_color[2])}, {int(bg_color[1])}, {int(bg_color[0])})"

    # Use pytesseract to get bounding boxes for all text
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    detected_text = []
    n_boxes = len(data['text'])

    for i in range(n_boxes):
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        if conf > 60 and text and len(text) > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            detected_text.append(text)

            # Get the local surrounding color for this text region
            local_color = get_surrounding_color(img, x, y, w, h)

            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            # Fill with local surrounding color instead of global background
            img[y1:y2, x1:x2] = local_color

    # Detect graphics
    tolerance = 30
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    non_bg_mask = (diff.max(axis=2) > tolerance).astype(np.uint8) * 255

    contours, _ = cv2.findContours(non_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_dir = Path(output_path).parent / "graphics"
    output_dir.mkdir(exist_ok=True)

    graphic_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            roi = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            roi[:, :, 3] = mask
            cropped = roi[y:y+h, x:x+w]

            graphic_filename = f"graphic_{graphic_count + 1}.png"
            graphic_path = output_dir / graphic_filename
            cv2.imwrite(str(graphic_path), cropped)

            layout_data["graphics"].append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "filename": graphic_filename
            })

            graphic_count += 1

    cv2.imwrite(output_path, img)

    layout_data_out.update(layout_data)
    return detected_text


def get_generation_prompt(layout_data: dict, detected_text: list) -> str:
    """Get the simplified prompt for LLM-based HTML generation."""
    text_reference = " ".join(detected_text)

    return f"""I have an original slide image and extracted graphics data. Please generate a complete HTML file that recreates this slide EXACTLY.

LAYOUT DATA PROVIDED:
- Slide dimensions: {layout_data["width"]}px x {layout_data["height"]}px
- Background color: {layout_data["background_color"]}
- Graphics extracted: {len(layout_data["graphics"])} PNG files with transparent backgrounds

GRAPHICS LOCATIONS (use these EXACT coordinates):
{json.dumps(layout_data["graphics"], indent=2)}

TEXT DETECTED BY OCR (for reference only - DO NOT use OCR coordinates):
{text_reference}

TASK: Generate a complete, pixel-perfect HTML file that recreates this slide.

CRITICAL REQUIREMENTS:

1. GRAPHICS POSITIONING:
   - Use the EXACT coordinates provided in the graphics data above
   - Position each graphic using: src="graphics/[filename]", left: [x]px, top: [y]px, width: [width]px, height: [height]px
   - Graphics are already extracted with transparent backgrounds - just position them exactly as specified

2. TEXT CONTENT & POSITIONING:
   - IGNORE the OCR text coordinates - they are fragmented and unreliable
   - Look at the original slide image CAREFULLY and read ALL the text yourself
   - The OCR text list is only provided to confirm what text exists - use your vision to see the actual layout
   - Position text elements by analyzing the image directly
   - Create logical text blocks (titles, headers, bullet points, labels) based on what you see

3. TEXT STYLING:
   - The slide is {layout_data["width"]}px wide - font sizes must be LARGE and proportional:
     * Main title: approximately 100-120px (very large, bold)
     * Section headers: approximately 55-70px
     * Body text / bullet points: approximately 40-50px
     * Small labels: approximately 35-45px
   - Analyze EACH text element for its exact color by looking at the image
   - Match font weights (bold for headers/keywords, normal for body)
   - Use sans-serif fonts (Arial, Helvetica, sans-serif)

4. LAYOUT STRUCTURE:
   - Use absolute positioning for all elements
   - Background color: {layout_data["background_color"]}
   - Position text labels near their associated graphics accurately by looking at the image
   - Use z-index: 1 for graphics and z-index: 2 for text so text appears ON TOP of images
   - Recreate bullet points exactly (• character)
   - Preserve proper formatting, line breaks, and indentation

5. RESPONSIVE SCALING:
   - Wrap slide-container in a slide-wrapper div with dark background
   - Add JavaScript to scale the slide to fit viewport while maintaining aspect ratio
   - Center the slide on the page

Return ONLY the complete HTML file, nothing else. Do not wrap it in markdown code blocks."""


def generate_html_with_gemini(image_path: str, layout_data: dict, detected_text: list, output_path: Path, gemini_client):
    """Use Gemini to generate the HTML directly from the image and layout data."""
    img = Image.open(image_path)
    img.thumbnail([2048, 2048], Image.Resampling.LANCZOS)

    prompt = get_generation_prompt(layout_data, detected_text)

    config = types.GenerateContentConfig(
        response_mime_type="text/plain"
    )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[img, prompt],
        config=config
    )

    generated_html = response.text.strip()

    # Remove markdown code blocks if present
    if generated_html.startswith("```html"):
        generated_html = generated_html[7:]
    if generated_html.startswith("```"):
        generated_html = generated_html[3:]
    if generated_html.endswith("```"):
        generated_html = generated_html[:-3]
    generated_html = generated_html.strip()

    with open(output_path, 'w') as f:
        f.write(generated_html)

    return generated_html


def html_to_pdf(html_path: str, output_pdf_path: str, width: int, height: int):
    """Convert HTML file to PDF using WeasyPrint with custom page size."""
    page_css = CSS(string=f'''
        @page {{
            size: {width}px {height}px;
            margin: 0;
        }}
        body {{
            width: {width}px;
            height: {height}px;
        }}
    ''')

    HTML(filename=str(html_path)).write_pdf(
        str(output_pdf_path),
        stylesheets=[page_css]
    )

    return output_pdf_path


def update_progress(job_id: str, slide_num: int, stage: str, total_slides: int):
    """Update the progress for a specific job and slide."""
    with job_lock:
        if job_id not in job_progress:
            job_progress[job_id] = {
                'total': total_slides,
                'slides': {},
                'completed': 0
            }
        job_progress[job_id]['slides'][slide_num] = stage
        if stage == 'complete':
            job_progress[job_id]['completed'] = sum(
                1 for s in job_progress[job_id]['slides'].values() if s == 'complete'
            )


def process_single_slide(image_path: str, api_key: str, output_dir: Path, slide_num: int, job_id: str, total_slides: int) -> str:
    """Process a single slide with progress updates."""
    gemini_client = genai.Client(api_key=api_key)

    # Stage 1: Extracting text and graphics
    update_progress(job_id, slide_num, 'extracting', total_slides)

    graphics_dir = output_dir / "graphics"
    graphics_dir.mkdir(exist_ok=True)

    layout_data = {}
    no_text_path = output_dir / "no_text.png"
    detected_text = remove_text_from_slide(image_path, str(no_text_path), gemini_client, layout_data)

    # Stage 2: Generating HTML with AI
    update_progress(job_id, slide_num, 'generating', total_slides)

    html_path = output_dir / "slide.html"
    generate_html_with_gemini(image_path, layout_data, detected_text, html_path, gemini_client)

    # Stage 3: Converting to PDF
    update_progress(job_id, slide_num, 'converting', total_slides)

    pdf_path = output_dir / "slide.pdf"
    html_to_pdf(str(html_path), str(pdf_path), layout_data["width"], layout_data["height"])

    # Stage 4: Complete
    update_progress(job_id, slide_num, 'complete', total_slides)

    return str(pdf_path)


def process_slides_parallel(image_paths: list, api_key: str, output_dir: Path, job_id: str) -> str:
    """Process multiple slides in parallel and merge into a single PDF."""
    total_slides = len(image_paths)

    # Initialize progress for all slides
    with job_lock:
        job_progress[job_id] = {
            'total': total_slides,
            'slides': {i + 1: 'waiting' for i in range(total_slides)},
            'completed': 0
        }

    pdf_paths = [None] * total_slides

    def process_slide(args):
        idx, image_path = args
        slide_dir = output_dir / f"slide_{idx + 1}"
        slide_dir.mkdir(exist_ok=True)
        return idx, process_single_slide(str(image_path), api_key, slide_dir, idx + 1, job_id, total_slides)

    # Process all slides in parallel
    with ThreadPoolExecutor(max_workers=min(total_slides, 10)) as executor:
        futures = [executor.submit(process_slide, (i, path)) for i, path in enumerate(image_paths)]

        for future in as_completed(futures):
            idx, pdf_path = future.result()
            pdf_paths[idx] = pdf_path

    # Merge all PDFs into one
    if len(pdf_paths) == 1:
        final_path = pdf_paths[0]
    else:
        merger = PdfMerger()
        for pdf_path in pdf_paths:
            merger.append(pdf_path)

        merged_pdf_path = output_dir / "merged_slides.pdf"
        merger.write(str(merged_pdf_path))
        merger.close()
        final_path = str(merged_pdf_path)

    # Store result
    with job_lock:
        job_results[job_id] = final_path
        job_progress[job_id]['status'] = 'done'

    return final_path




@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and start processing."""
    if 'file' not in request.files:
        return {'error': 'No file uploaded'}, 400

    file = request.files['file']
    api_key = request.form.get('api_key', '').strip()

    if file.filename == '':
        return {'error': 'No file selected'}, 400

    if not api_key:
        return {'error': 'Please provide a Gemini API key'}, 400

    if not allowed_file(file.filename):
        return {'error': 'Invalid file type. Please upload a PDF, PNG, or JPG file.'}, 400

    try:
        # Create temporary directory for processing
        temp_dir = Path(tempfile.mkdtemp())

        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = temp_dir / filename
        file.save(str(file_path))

        # Convert PDF to images if needed
        if filename.lower().endswith('.pdf'):
            images = convert_from_path(str(file_path), dpi=200)
            if not images:
                return {'error': 'Failed to extract images from PDF'}, 400

            # Save all pages as images
            image_paths = []
            for i, img in enumerate(images):
                image_path = temp_dir / f"slide_{i + 1}.png"
                img.save(str(image_path))
                image_paths.append(image_path)
        else:
            image_paths = [file_path]

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Create output directory
        output_dir = temp_dir / "output"
        output_dir.mkdir(exist_ok=True)

        # Start processing in background thread
        def run_processing():
            try:
                process_slides_parallel(image_paths, api_key, output_dir, job_id)
            except Exception as e:
                with job_lock:
                    job_progress[job_id]['status'] = 'error'
                    job_progress[job_id]['error'] = str(e)

        thread = threading.Thread(target=run_processing)
        thread.start()

        return {'job_id': job_id, 'total_slides': len(image_paths)}

    except Exception as e:
        return {'error': str(e)}, 500


@app.route('/progress/<job_id>')
def progress(job_id):
    """Stream progress updates using Server-Sent Events."""
    def generate():
        import time
        while True:
            with job_lock:
                if job_id not in job_progress:
                    yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                progress_data = job_progress[job_id].copy()

            yield f"data: {json.dumps(progress_data)}\n\n"

            if progress_data.get('status') == 'done':
                break
            if progress_data.get('status') == 'error':
                break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream')


@app.route('/download/<job_id>')
def download(job_id):
    """Download the completed PDF."""
    with job_lock:
        if job_id not in job_results:
            return {'error': 'Job not found or not complete'}, 404
        pdf_path = job_results[job_id]

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name='converted_slides.pdf',
        mimetype='application/pdf'
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5020)
