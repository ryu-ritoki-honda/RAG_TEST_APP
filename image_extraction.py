"""
Extract and process images from PDFs for RAG.
Uses Azure OpenAI's vision capabilities to analyze images.
"""

import base64
import io
import tempfile
from pathlib import Path
from typing import List, Tuple
import pdfplumber
from aoai_client import get_aoai_client


def extract_images_from_pdf(pdf_path: str) -> List[Tuple[bytes, int]]:
    """
    Extract images from PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of tuples (image_bytes, page_number)
    """
    images = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract images from the page
                try:
                    # pdfplumber extracts images as objects in the page
                    if hasattr(page, 'images') and page.images:
                        for img_idx, img in enumerate(page.images):
                            try:
                                # Get the bounding box and crop
                                bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                                im = page.within_bbox(bbox).to_image(resolution=150)
                                
                                # Convert PIL image to bytes
                                img_bytes = io.BytesIO()
                                im.save(img_bytes, format='JPEG')
                                img_bytes = img_bytes.getvalue()
                                
                                images.append((img_bytes, page_num + 1))
                            except Exception as e:
                                print(f"Warning: Could not process image {img_idx} on page {page_num + 1}: {e}")
                except Exception as e:
                    print(f"Warning: Could not extract images from page {page_num + 1}: {e}")
                    
    except Exception as e:
        print(f"Error extracting images from PDF: {e}")
    
    return images


def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    Encode image bytes to base64 string.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Base64 encoded string
    """
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def analyze_image_with_vision(image_bytes: bytes, page_num: int) -> str:
    """
    Use Azure OpenAI's vision model to analyze an image and generate description.
    
    Args:
        image_bytes: Raw image bytes
        page_num: Page number for context
        
    Returns:
        Text description of the image
    """
    try:
        client = get_aoai_client()
        
        # Encode image to base64
        image_base64 = encode_image_to_base64(image_bytes)
        
        # Determine image type (assume JPEG for now)
        image_media_type = "image/jpeg"
        
        # Call Azure OpenAI with vision
        response = client.chat.completions.create(
            model="gpt-4-vision",  # Use appropriate vision model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this image and provide a detailed description of its contents, including any text, charts, diagrams, or other visual elements. Be concise but thorough."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_media_type};base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        
        description = response.choices[0].message.content
        
        # Add page number context to description
        description_with_context = f"[Image from page {page_num}] {description}"
        
        return description_with_context
        
    except Exception as e:
        print(f"Error analyzing image with vision: {e}")
        return f"[Image from page {page_num}] Image could not be analyzed."


def extract_and_describe_images(pdf_path: str) -> List[str]:
    """
    Extract images from PDF and generate descriptions using Azure OpenAI vision.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of image descriptions as strings
    """
    descriptions = []
    
    # Extract raw images
    images = extract_images_from_pdf(pdf_path)
    
    if not images:
        print(f"No images found in {pdf_path}")
        return descriptions
    
    print(f"Found {len(images)} images in {pdf_path}")
    
    # Analyze each image
    for image_bytes, page_num in images:
        description = analyze_image_with_vision(image_bytes, page_num)
        descriptions.append(description)
        print(f"Analyzed image from page {page_num}")
    
    return descriptions


def extract_text_and_images_from_pdf(pdf_path: str) -> Tuple[List[str], List[str]]:
    """
    Extract both text and images from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Tuple of (text_chunks, image_descriptions)
    """
    from pypdf import PdfReader
    from chunking import chunk_text_by_chars
    
    text_chunks = []
    
    # Extract text using pypdf
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        text_chunks = chunk_text_by_chars(
            text,
            chunk_size=1000,
            overlap=200
        )
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    # Extract and describe images
    image_descriptions = extract_and_describe_images(pdf_path)
    
    return text_chunks, image_descriptions
