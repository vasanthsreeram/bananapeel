import sys
from pathlib import Path
from weasyprint import HTML, CSS

def html_to_pdf(html_path: str, output_pdf_path: str = None, width: int = 3823, height: int = 2134):
    """Convert HTML file to PDF using WeasyPrint with custom page size."""

    html_path = Path(html_path)

    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)

    # Default output path: same directory as HTML, same name with .pdf extension
    if output_pdf_path is None:
        output_pdf_path = html_path.with_suffix('.pdf')
    else:
        output_pdf_path = Path(output_pdf_path)

    print(f"Converting HTML to PDF...")
    print(f"Input:  {html_path}")
    print(f"Output: {output_pdf_path}")
    print(f"Page size: {width}px x {height}px")

    # Create CSS to set the exact page size matching the slide
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

    # Convert HTML to PDF with custom page size
    HTML(filename=str(html_path)).write_pdf(
        str(output_pdf_path),
        stylesheets=[page_css]
    )

    print(f"✓ PDF generated successfully: {output_pdf_path}")
    return output_pdf_path


if __name__ == "__main__":
    # Default HTML path
    default_html = "/Users/vasanth/nanobananaslides/presentation/slide3/slide_llm_generated.html"

    # Use command line argument if provided, otherwise use default
    html_input = sys.argv[1] if len(sys.argv) > 1 else default_html
    output_pdf = sys.argv[2] if len(sys.argv) > 2 else None

    html_to_pdf(html_input, output_pdf)
