import cv2
import pytesseract
import numpy as np
from PIL import Image
from pathlib import Path
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini client
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
GEMINI_MODEL = "gemini-3-pro-preview"


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


def remove_text_from_slide(image_path: str, output_path: str, use_gemini_refinement: bool = True):
    """Remove all text from a slide image using OCR detection and masking."""

    # Load image
    img = cv2.imread(image_path)

    # Layout data for HTML generation
    layout_data = {
        "width": img.shape[1],
        "height": img.shape[0],
        "background_color": None,
        "graphics": []
    }

    # Get global background color from top-left corner
    bg_color = img[10:50, 10:50].mean(axis=(0, 1)).astype(int)
    layout_data["background_color"] = f"rgb({int(bg_color[2])}, {int(bg_color[1])}, {int(bg_color[0])})"

    # Use pytesseract to get bounding boxes for all text
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb_img, output_type=pytesseract.Output.DICT)

    # Collect all detected text for reference (not coordinates)
    detected_text = []
    n_boxes = len(data['text'])
    masked_count = 0

    # Create a copy for text bounding box visualization
    text_boxes_img = img.copy()

    for i in range(n_boxes):
        # Only mask if confidence > 60 and text is not empty (filters out graphics)
        conf = int(data['conf'][i])
        text = data['text'][i].strip()
        if conf > 60 and text and len(text) > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

            # Collect text (no coordinates)
            detected_text.append(text)

            # Get the local surrounding color for this text region
            local_color = get_surrounding_color(img, x, y, w, h)

            # Add small padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(img.shape[1], x + w + padding)
            y2 = min(img.shape[0], y + h + padding)

            # Draw red bounding box on visualization image
            cv2.rectangle(text_boxes_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Fill text with local surrounding color instead of global background
            img[y1:y2, x1:x2] = local_color
            masked_count += 1

    # Save text bounding boxes visualization
    text_boxes_path = Path(output_path).parent / "text_boxes.png"
    cv2.imwrite(str(text_boxes_path), text_boxes_img)

    # Now detect graphics on the text-removed image
    # Create a mask where pixels differ from background color
    tolerance = 30
    diff = np.abs(img.astype(np.int16) - bg_color.astype(np.int16))
    non_bg_mask = (diff.max(axis=2) > tolerance).astype(np.uint8) * 255

    # Find contours of graphics (text already removed, so no need to exclude text regions)
    contours, _ = cv2.findContours(non_bg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract graphics with transparent background
    graphic_count = 0
    output_dir = Path(output_path).parent / "graphics"
    output_dir.mkdir(exist_ok=True)

    # Create a copy for debug output (with green borders)
    debug_img = img.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # filter tiny noise
            # Get bounding rect
            x, y, w, h = cv2.boundingRect(contour)

            # Create mask for this contour
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)  # filled

            # Extract region with alpha channel from text-removed image (clean, no green border)
            roi = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

            # Apply mask as alpha
            roi[:, :, 3] = mask

            # Crop to bounding rect
            cropped = roi[y:y+h, x:x+w]

            # Save
            graphic_filename = f"graphic_{graphic_count + 1}.png"
            graphic_path = output_dir / graphic_filename
            cv2.imwrite(str(graphic_path), cropped)

            # Store graphic position for HTML
            layout_data["graphics"].append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "filename": graphic_filename
            })

            # Draw green contour on debug image (after extraction)
            cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)

            graphic_count += 1

    # Use debug_img for the output file (shows green borders for reference)
    img = debug_img

    # Save result
    cv2.imwrite(output_path, img)

    # Save layout data as JSON
    json_path = Path(output_path).parent / "layout_data.json"
    with open(json_path, 'w') as f:
        json.dump(layout_data, f, indent=2)

    # Generate HTML with Gemini
    if use_gemini_refinement:
        html_path = Path(output_path).parent / "slide_llm_generated.html"
        generate_html_with_gemini(image_path, layout_data, detected_text, html_path)

    return layout_data


def get_generation_prompt(layout_data: dict, detected_text: list) -> str:
    """Get the simplified prompt for LLM-based HTML generation."""

    # Join detected text for reference
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


def generate_html_with_gemini(image_path: str, layout_data: dict, detected_text: list, output_path: Path):
    """Use Gemini to generate the HTML directly from the image and layout data."""

    # Load the original image
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

    # Save generated HTML
    with open(output_path, 'w') as f:
        f.write(generated_html)

    return generated_html


if __name__ == "__main__":
    input_path = "/Users/vasanth/nanobananaslides/presentation/slide3/full_slide.png"
    output_path = "/Users/vasanth/nanobananaslides/presentation/slide3/full_slide_no_text.png"

    # Set to False to skip Gemini refinement
    USE_GEMINI = True

    remove_text_from_slide(input_path, output_path, use_gemini_refinement=USE_GEMINI)
