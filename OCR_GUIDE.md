# OCR (Optical Character Recognition) Support - Feature Guide

## Overview
The RAG system now includes comprehensive OCR (Optical Character Recognition) support for extracting text from scanned documents and images. This allows you to ingest:

- **Scanned PDFs** (image-based documents)
- **Image files** (JPG, PNG, BMP, TIFF, etc.)
- **Mixed PDFs** (combination of text and scanned pages)

## What the OCR System Does

### Text Extraction Pipeline
```
Input Document
    ↓
Detect if scanned/image-based
    ↓
Convert to Images (if PDF)
    ↓
Extract Text via EasyOCR
    ↓
Organize & Chunk Text
    ↓
Embed & Index in RAG
```

## Features

### 1. **Automatic Detection**
- Automatically detects if a PDF is scanned vs. text-based
- Falls back to OCR only when needed (saves processing time)
- Handles mixed PDFs with both text and scanned pages

### 2. **Multi-Language Support**
- Default: English
- Supports 80+ languages via EasyOCR
- Can be configured for multiple languages

### 3. **Smart Processing**
- Extracts text with position information
- Organizes text logically (top-to-bottom, left-to-right)
- Adds page context to each extracted chunk
- Chunks text appropriately for embedding

### 4. **Multiple File Format Support**
Supported file types for OCR:
- **PDFs**: `*.pdf` (scanned or mixed)
- **Images**: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.tiff`

## Usage

### Uploading Scanned Documents

#### Via Repository (Sidebar)
1. Click "Add files to repository"
2. Select PDF or image files
3. System automatically processes them with OCR if needed
4. Documents are indexed and searchable

#### Via Main Upload Area
1. Use "Upload documents" section
2. Select any supported file type
3. Files are processed and embedded automatically

### Search & Retrieval

When you ask a question:
- The system searches through all extracted text
- Results include OCR-extracted content alongside regular text
- Retrieved chunks are tagged with source (e.g., "[OCR - Page 3]")

### Example Use Cases

1. **Scanned Invoices & Receipts**
   - Extract line items, amounts, dates
   - Include in business document RAG

2. **Legal Documents**
   - Scanned contracts and agreements
   - Extract terms, conditions, signatures

3. **Historical Archives**
   - Digitized old documents
   - Make them searchable and accessible

4. **Handwritten Notes**
   - Scanned notebooks and notes
   - Extract structured information

5. **Forms & Questionnaires**
   - Filled-out paper forms
   - Extract responses and answers

## Technical Details

### OCR Engine: EasyOCR
- **Library**: `easyocr`
- **Advantages**:
  - No external dependencies (no Tesseract installation needed)
  - Multi-language support out of the box
  - Deep learning based (better accuracy)
  - CPU-friendly

### PDF Processing: pdf2image
- **Library**: `pdf2image`
- **Function**: Converts PDF pages to images for OCR processing

### Image Handling: Pillow
- **Library**: `Pillow`
- **Function**: Image format conversion and manipulation

## Performance Considerations

### Processing Time
- **First OCR run**: May take time (OCR models are downloaded on first use)
- **Subsequent runs**: Faster as models are cached
- **Large documents**: Multi-page documents take longer proportional to page count
- **Image quality**: Higher resolution images take longer but give better results

### Resource Usage
- **Memory**: Significant for large PDFs (all pages loaded)
- **Disk**: OCR models (~100-200MB per language)
- **GPU**: Optional (uses CPU by default)

### Optimization Tips
1. **Pre-process images**: Improve contrast/brightness before uploading
2. **Batch size**: Upload documents in manageable batches
3. **Language selection**: Specify only needed languages to reduce model size
4. **Image resolution**: 150 DPI is optimal balance

## Configuration

### Changing OCR Languages

Edit `ocr_utils.py`, function `get_ocr_reader()`:

```python
def get_ocr_reader(languages: List[str] = ['en']):  # Change here
    """Get or initialize the OCR reader (cached)."""
    ...
```

Available languages: 'en', 'ja', 'zh-chs', 'fr', 'de', 'es', 'pt', 'ar', etc.

### Adjusting Image Resolution

In `ocr_utils.py`, modify the resolution parameter in `ocr_pdf()`:

```python
images = pdf2image.convert_from_path(pdf_path, dpi=150)  # Adjust DPI here
```

Higher DPI = better quality but slower processing.

## Troubleshooting

### OCR Models Not Downloading
- **Issue**: First run takes very long or fails
- **Solution**: Check internet connection, ensure sufficient disk space (~200MB)

### Poor Text Extraction Quality
- **Issue**: Some text is missing or incorrectly recognized
- **Solution**:
  - Improve image quality/contrast before uploading
  - Check that correct language is selected
  - Try higher DPI setting

### Memory Issues with Large PDFs
- **Issue**: System runs out of memory
- **Solution**:
  - Split large PDFs into smaller files
  - Process one document at a time
  - Increase system available memory/RAM

### Slow Processing
- **Issue**: OCR is very slow
- **Solution**:
  - Consider pre-processing to rotate/crop images
  - Ensure no other heavy processes running
  - Check system resources (CPU, Memory)

## Installation

The required packages are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key OCR packages:
- `easyocr` - OCR engine
- `pdf2image` - PDF to image conversion
- `Pillow` - Image processing

## API Reference

### Core Functions

#### `ocr_image(image_bytes: bytes, page_num: int = None) -> str`
Extract text from image bytes.

```python
from ocr_utils import ocr_image
text = ocr_image(image_data, page_num=1)
```

#### `ocr_pdf(pdf_path: str) -> List[str]`
Extract text from entire PDF using OCR.

```python
from ocr_utils import ocr_pdf
chunks = ocr_pdf("document.pdf")
```

#### `ocr_image_file(image_path: str) -> List[str]`
Extract text from image file.

```python
from ocr_utils import ocr_image_file
chunks = ocr_image_file("scan.jpg")
```

#### `extract_text_with_ocr_fallback(pdf_path: str) -> Tuple[List[str], bool]`
Extract text from PDF with OCR fallback for scanned pages.

```python
from ocr_utils import extract_text_with_ocr_fallback
chunks, used_ocr = extract_text_with_ocr_fallback("mixed.pdf")
```

#### `is_scanned_pdf(pdf_path: str) -> bool`
Check if PDF is scanned (image-based).

```python
from ocr_utils import is_scanned_pdf
if is_scanned_pdf("doc.pdf"):
    print("This PDF needs OCR")
```

## Limitations

- **Accuracy**: OCR is ~95-98% accurate for clean documents, lower for poor quality
- **Handwriting**: Limited support for handwritten text
- **Complex layouts**: May struggle with multi-column layouts or tables
- **Language**: Best with single language documents
- **Speed**: Slower than text extraction (but faster than manual data entry!)

## Future Improvements

Potential enhancements:
- GPU acceleration support for faster processing
- Table structure recognition
- Layout analysis and preservation
- Multi-language document support
- Handwriting recognition improvements
- Confidence scoring per extracted line

---

For more information about EasyOCR, visit: https://github.com/JaidedAI/EasyOCR
