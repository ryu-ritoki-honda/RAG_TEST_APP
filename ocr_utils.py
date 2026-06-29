"""
OCR (Optical Character Recognition) support for RAG.
Extracts text from scanned PDFs and image files.
"""

import io
import tempfile
from pathlib import Path
from typing import List, Tuple
import easyocr
from PIL import Image
import pdf2image
from chunking import chunk_text_by_chars


# Initialize OCR reader (cached for performance)
_ocr_reader = None


def get_ocr_reader(languages: List[str] = ['en']):
    """
    Get or initialize the OCR reader (cached).
    
    Args:
        languages: List of language codes for OCR
        
    Returns:
        Initialized OCR reader
    """
    global _ocr_reader
    if _ocr_reader is None:
        print("Initializing OCR reader... (this may take a moment)")
        _ocr_reader = easyocr.Reader(languages, gpu=False)
    return _ocr_reader


def ocr_image(image_bytes: bytes, page_num: int = None) -> str: # type: ignore
    """
    Extract text from an image using OCR.
    
    Args:
        image_bytes: Raw image bytes
        page_num: Optional page number for context
        
    Returns:
        Extracted text from the image
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get OCR reader
        reader = get_ocr_reader()
        
        # Save image temporarily for easyocr
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Perform OCR
            results = reader.readtext(tmp_path)
            
            # Extract and organize text
            text_lines = []
            for detection in results:
                text = detection[1] # type: ignore
                if text.strip():
                    text_lines.append(text.strip())
            
            extracted_text = "\n".join(text_lines)
            
            # Add page context if provided
            if page_num:
                extracted_text = f"[OCR - Page {page_num}]\n{extracted_text}"
            else:
                extracted_text = f"[OCR]\n{extracted_text}"
            
            return extracted_text
            
        finally:
            try:
                Path(tmp_path).unlink()
            except:
                pass
                
    except Exception as e:
        print(f"Error performing OCR on image: {e}")
        return f"[OCR Error] Could not extract text: {e}"


def ocr_pdf(pdf_path: str) -> List[str]:
    """
    Extract text from a scanned PDF using OCR.
    Converts each PDF page to an image and performs OCR.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of text chunks from the PDF
    """
    text_chunks = []
    
    try:
        print(f"Starting OCR extraction for {Path(pdf_path).name}...")
        
        # Convert PDF pages to images
        images = pdf2image.convert_from_path(pdf_path)
        
        print(f"Found {len(images)} pages in PDF")
        
        # OCR each page
        for page_num, image in enumerate(images, start=1):
            print(f"Processing page {page_num}/{len(images)}...")
            
            # Convert PIL image to bytes
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes = img_bytes.getvalue()
            
            # Extract text from image
            text = ocr_image(img_bytes, page_num=page_num)
            
            # Chunk the extracted text
            chunks = chunk_text_by_chars(
                text,
                chunk_size=1000,
                overlap=200
            )
            
            text_chunks.extend(chunks)
        
        print(f"OCR extraction complete. Extracted {len(text_chunks)} chunks.")
        return text_chunks
        
    except Exception as e:
        print(f"Error during OCR PDF processing: {e}")
        return []


def ocr_image_file(image_path: str) -> List[str]:
    """
    Extract text from an image file using OCR.
    
    Args:
        image_path: Path to image file (JPG, PNG, etc.)
        
    Returns:
        List of text chunks
    """
    text_chunks = []
    
    try:
        # Read image
        image = Image.open(image_path)
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue()
        
        # Extract text
        text = ocr_image(img_bytes)
        
        # Chunk the text
        chunks = chunk_text_by_chars(
            text,
            chunk_size=1000,
            overlap=200
        )
        
        return chunks
        
    except Exception as e:
        print(f"Error during OCR image processing: {e}")
        return []


def is_scanned_pdf(pdf_path: str) -> bool:
    """
    Detect if a PDF is scanned (image-based) vs text-based.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        True if PDF appears to be scanned/image-based
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        
        # Check if first page has minimal text
        if len(reader.pages) > 0:
            first_page_text = reader.pages[0].extract_text()
            
            # If very little text extracted, likely scanned
            if not first_page_text or len(first_page_text.strip()) < 100:
                return True
        
        return False
        
    except Exception as e:
        print(f"Error checking if PDF is scanned: {e}")
        return False


def extract_text_with_ocr_fallback(pdf_path: str) -> Tuple[List[str], bool]:
    """
    Extract text from PDF, falling back to OCR if needed.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (text_chunks, ocr_used)
    """
    try:
        from pypdf import PdfReader
        
        reader = PdfReader(pdf_path)
        text = ""
        
        # Try to extract text normally
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # If we got text, use it
        if text.strip() and len(text.strip()) > 100:
            chunks = chunk_text_by_chars(
                text,
                chunk_size=1000,
                overlap=200
            )
            return chunks, False
        
        # Otherwise, use OCR
        print(f"PDF appears to be scanned. Using OCR...")
        ocr_chunks = ocr_pdf(pdf_path)
        return ocr_chunks, True
        
    except Exception as e:
        print(f"Error in extract_text_with_ocr_fallback: {e}")
        # Try OCR as last resort
        try:
            return ocr_pdf(pdf_path), True
        except:
            return [], False
