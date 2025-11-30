# Banana Peel

Converts nano banana slides (PDFs or images) into proper editable PDF slides that can be imported into Canva, Google Slides, or PowerPoint.

## Features

- Upload PDF or image files (PNG, JPG, JPEG)
- Automatic text extraction and layout preservation
- Graphics extraction with transparent backgrounds
- AI-powered HTML generation using Google Gemini
- Multi-slide processing with real-time progress tracking
- Export to editable PDF format

## Prerequisites

### System Dependencies

You need to install the following system-level dependencies:

#### macOS
```bash
brew install poppler tesseract
```

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils tesseract-ocr
```

#### Windows
1. **Poppler**: Download from https://github.com/oschwartz10612/poppler-windows/releases/ and add to PATH
2. **Tesseract**: Download from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd bananapeel
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Get a Gemini API key:
   - Visit https://aistudio.google.com/apikey
   - Create a free API key
   - You'll enter this key in the web interface when uploading files

## Usage

1. Start the Flask development server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5020
```

3. Upload your slide:
   - Click the upload area or drag and drop your file
   - Enter your Gemini API key
   - Click "Convert to PDF"

4. Wait for processing:
   - Watch real-time progress for each slide
   - Each slide goes through: Extracting → AI Processing → Converting → Complete

5. Download your converted PDF:
   - Click the download button when processing is complete

## Importing to Other Tools

### Canva
1. Upload the converted PDF to Canva
2. Each element (text, graphics) will be editable individually

### Google Slides
1. File → Import slides
2. Select your converted PDF
3. Choose which slides to import

### PowerPoint
1. Insert → Pictures
2. Select your PDF pages
3. Edit as needed

## Configuration

### Change the AI Model

Edit `app.py` line 28 to use a different Gemini model:

```python
GEMINI_MODEL = "gemini-2.0-flash-exp"  # or "gemini-1.5-flash", etc.
```

### Change Server Port

Edit `app.py` line 492:

```python
app.run(debug=True, host='0.0.0.0', port=5020)  # Change port here
```

### File Size Limit

Edit `app.py` line 23 to change the maximum upload size:

```python
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB default
```

## Troubleshooting

### "Unable to get page count. Is poppler installed?"
- Make sure poppler is installed (see Prerequisites)
- Verify it's in your PATH: `pdfinfo --version`

### "pytesseract: TesseractNotFoundError"
- Make sure Tesseract OCR is installed (see Prerequisites)
- Verify it's in your PATH: `tesseract --version`

### "Invalid API key" or Gemini errors
- Verify your API key is correct from https://aistudio.google.com/apikey
- Check you have API quota remaining
- Ensure the model name in `GEMINI_MODEL` is accessible with your key

### Server won't start - Port already in use
```bash
# Find process using port 5020
lsof -i :5020
# Kill it
kill -9 <PID>
```

## Development

The application uses:
- **Flask**: Web framework
- **pdf2image**: PDF to image conversion (requires poppler)
- **pytesseract**: OCR text extraction (requires tesseract)
- **opencv-python**: Image processing and graphics extraction
- **google-genai**: AI-powered HTML generation
- **weasyprint**: HTML to PDF conversion

## License

MIT

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.
