import os
import json
import base64
import io
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from pdf2image import convert_from_path

load_dotenv()

# Initialize Gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Models
COORDINATOR_MODEL = "gemini-3-pro-preview"
IMAGE_EXTRACTION_MODEL = "gemini-2.5-flash-image"

# Output directory
OUTPUT_DIR = Path("presentation")


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """Convert PDF pages to PIL Images."""
    images = convert_from_path(pdf_path, dpi=dpi)
    return images


def analyze_slide_with_coordinator(image: Image.Image) -> dict:
    """
    Use the coordinator model to analyze the slide and return:
    - Bounding boxes for all images/graphics
    - Classification (crop vs smart extraction)
    - HTML template for the slide
    """
    # Resize image if too large for better processing
    img_copy = image.copy()
    img_copy.thumbnail([2048, 2048], Image.Resampling.LANCZOS)

    prompt = """Analyze this presentation slide and provide:

1. Detect ALL distinct graphic elements/illustrations (not text blocks). For each graphic, provide:
   - "label": descriptive name for the image
   - "box_2d": bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000
   - "extraction_type": either "crop" or "smart"

IMPORTANT for extraction_type - BE CONSERVATIVE with "smart":
- Use "crop" for most graphics - this is the default choice
- Use "smart" when text is INSIDE or OVERLAPPING the graphic (e.g., text inside a circular arrow, labels embedded within a diagram)
- Text labels NEXT TO or AROUND a graphic (but not inside it) do NOT require "smart" - use "crop"
- Simple standalone icons, arrows, or illustrations without internal text = "crop"
- Graphics that CONTAIN text within their boundaries (like a loop with explanation text inside) = "smart"

2. Generate the HTML to recreate this slide layout. The HTML should:
   - Match the slide's visual layout exactly
   - Use placeholder image tags like <img src="image1.png"> for each detected graphic (in order)
   - Include all text content with proper styling (fonts, colors, positioning)
   - Use inline CSS for styling
   - Use a cream/beige background color matching the slide (#F5F0E6 approximately)

Return a JSON object with this exact structure:
{
    "images": [
        {
            "label": "description of image",
            "box_2d": [ymin, xmin, ymax, xmax],
            "extraction_type": "crop" or "smart"
        }
    ],
    "html": "<!DOCTYPE html>..."
}

Important:
- Bounding boxes should tightly fit each distinct graphic element
- Do not include text-only regions as images
- The HTML should position images approximately where they appear in the slide"""

    config = types.GenerateContentConfig(
        response_mime_type="application/json"
    )

    response = client.models.generate_content(
        model=COORDINATOR_MODEL,
        contents=[img_copy, prompt],
        config=config
    )

    # Parse JSON response
    result = json.loads(response.text)
    return result


def crop_image(image: Image.Image, box_2d: list[int]) -> Image.Image:
    """Crop image using normalized bounding box coordinates."""
    width, height = image.size

    # Convert normalized coordinates (0-1000) to absolute pixels
    ymin, xmin, ymax, xmax = box_2d
    abs_x1 = int(xmin / 1000 * width)
    abs_y1 = int(ymin / 1000 * height)
    abs_x2 = int(xmax / 1000 * width)
    abs_y2 = int(ymax / 1000 * height)

    # Crop the image
    cropped = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
    return cropped


def smart_extract_image(cropped_image: Image.Image) -> Image.Image:
    """
    Use Gemini image model to extract just the graphic element,
    isolating it from any overlapping text.
    """
    prompt = """Extract ONLY the graphic/illustration from this image, removing any text that overlaps it.
    Return just the isolated graphic element with a transparent or white background where text was removed.
    Preserve the original quality and colors of the graphic."""

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"]
    )

    response = client.models.generate_content(
        model=IMAGE_EXTRACTION_MODEL,
        contents=[cropped_image, prompt],
        config=config
    )

    # Extract image from response using the SDK's helper method
    for part in response.parts:
        if part.text is not None:
            print(f"      Model response: {part.text[:100]}..." if len(part.text) > 100 else f"      Model response: {part.text}")
        elif part.inline_data is not None:
            # Use the SDK's as_image() helper
            return part.as_image()

    # If no image returned, return original
    print("      Warning: No image returned from smart extraction, using cropped image")
    return cropped_image


def process_slide(image: Image.Image, slide_number: int) -> dict:
    """Process a single slide: analyze, extract images, generate HTML."""

    # Create output directory for this slide
    slide_dir = OUTPUT_DIR / f"slide{slide_number}"
    slide_dir.mkdir(parents=True, exist_ok=True)

    # Save the full slide image for reference
    image.save(slide_dir / "full_slide.png")
    print(f"Saved full slide to {slide_dir}/full_slide.png")

    # Analyze slide with coordinator model
    print(f"Analyzing slide {slide_number} with {COORDINATOR_MODEL}...")
    analysis = analyze_slide_with_coordinator(image)

    print(f"Found {len(analysis.get('images', []))} images to extract")

    # Extract each image
    extracted_images = []
    for idx, img_info in enumerate(analysis.get("images", []), start=1):
        print(f"  Extracting image {idx}: {img_info['label']}")
        print(f"    Box: {img_info['box_2d']}, Type: {img_info['extraction_type']}")

        # Crop the image region
        cropped = crop_image(image, img_info["box_2d"])

        # Apply smart extraction if needed
        if img_info["extraction_type"] == "smart":
            print(f"    Using smart extraction for overlapping text...")
            final_image = smart_extract_image(cropped)
        else:
            print(f"    Using simple crop extraction...")
            final_image = cropped

        # Save the extracted image
        image_filename = f"image{idx}.png"
        final_image.save(slide_dir / image_filename)
        print(f"    Saved to {slide_dir}/{image_filename}")

        extracted_images.append({
            "filename": image_filename,
            "label": img_info["label"],
            "extraction_type": img_info["extraction_type"]
        })

    # Save HTML file
    html_content = analysis.get("html", "")
    with open(slide_dir / "page.html", "w") as f:
        f.write(html_content)
    print(f"Saved HTML to {slide_dir}/page.html")

    # Save analysis metadata
    with open(slide_dir / "metadata.json", "w") as f:
        json.dump({
            "slide_number": slide_number,
            "images": extracted_images,
            "analysis": analysis
        }, f, indent=2)
    print(f"Saved metadata to {slide_dir}/metadata.json")

    return {
        "slide_number": slide_number,
        "output_dir": str(slide_dir),
        "images_extracted": len(extracted_images),
        "html_generated": True
    }


if __name__ == "__main__":
    # Process slide 3 of the PDF
    pdf_path = "/Users/vasanth/nanobananaslides/Biomaterials_and_Host_Defense.pdf"
    slide_number = 3

    print(f"Converting PDF to images...")
    images = pdf_to_images(pdf_path)
    print(f"PDF has {len(images)} pages")

    # Process slide 2 (index 1)
    slide_image = images[slide_number - 1]
    print(f"Processing slide {slide_number} (size: {slide_image.size})")

    OUTPUT_DIR.mkdir(exist_ok=True)
    result = process_slide(slide_image, slide_number)

    print("\n" + "="*50)
    print("DONE!")
    print(f"Output directory: {result['output_dir']}")
    print(f"Images extracted: {result['images_extracted']}")
    print(f"HTML generated: {result['html_generated']}")
