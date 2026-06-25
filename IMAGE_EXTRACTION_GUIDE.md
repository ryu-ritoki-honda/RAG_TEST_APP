# Image Extraction from PDFs - Feature Guide

## Overview
The RAG system now supports extracting information from images embedded in PDF files. When you upload a PDF, the system will:

1. **Extract Text**: As before, all text content is extracted and chunked for embedding
2. **Extract Images**: All images from the PDF are automatically extracted
3. **Analyze Images**: Each image is analyzed using Azure OpenAI's vision capabilities (GPT-4 Vision)
4. **Generate Descriptions**: Detailed descriptions of image contents are created
5. **Embed Everything**: Both text chunks and image descriptions are embedded and indexed

## How It Works

### Image Processing Pipeline
```
PDF Upload
    ↓
Extract Images (pdfplumber)
    ↓
Analyze with Azure OpenAI Vision
    ↓
Generate Text Descriptions
    ↓
Embed & Index (same as regular text)
```

### Image Analysis
Each extracted image is sent to Azure OpenAI's GPT-4 Vision model, which analyzes:
- Text content in the image
- Charts and diagrams
- Visual elements and patterns
- Spatial relationships
- Any other relevant visual information

The model generates a detailed description that captures the essential information from the image. This description is then embedded and indexed alongside regular text documents.

## Usage

### Uploading PDFs with Images

1. **Repository Upload (Sidebar)**:
   - Use "Add files to repository" in the sidebar
   - Select PDF files that contain images
   - The system will process both text and images automatically

2. **Main Document Upload**:
   - Use "Upload documents" in the main area
   - Select PDF files with images
   - Images will be extracted and analyzed

### Search & Retrieval

When you ask a question:
- The RAG system searches through both text chunks AND image descriptions
- Results may include information from images in PDFs
- Retrieved chunks show whether content came from text or images

## Requirements

Install the required packages:
```bash
pip install -r requirements.txt
```

Key dependencies for image extraction:
- `pdfplumber`: PDF parsing and image extraction
- `openai`: Azure OpenAI client for vision analysis

## Configuration

The image extraction uses your existing Azure OpenAI setup:
- Requires valid `AZURE_OPENAI_API_KEY` environment variable
- Uses GPT-4 Vision model (ensure your Azure OpenAI deployment includes this)
- API version: 2025-03-01-preview

## Performance Notes

- **First Run**: Image extraction may take longer on first load as images are analyzed
- **Caching**: Image descriptions are cached during your session
- **Cost**: Each image analyzed uses GPT-4 Vision tokens (consult Azure OpenAI pricing)

## Troubleshooting

### Images not extracting
- Verify the PDF actually contains embedded images
- Check Azure OpenAI API key is set correctly
- Review console output for specific errors

### Slow processing
- Large PDFs with many images will take longer to process
- Consider splitting large PDFs or extracting only key pages

### Vision model errors
- Ensure your Azure OpenAI deployment includes a vision-capable model
- Check API version compatibility (currently 2025-03-01-preview)

## Example Use Cases

1. **Technical Documentation**: Extract diagrams, flowcharts, screenshots
2. **Reports with Tables**: Analyze tables and charts presented as images
3. **Spreadsheet Exports**: Process scanned spreadsheets or receipts
4. **Presentations**: Extract content from PDF slide decks with visual elements
5. **Business Documents**: Process invoices, contracts, and forms with visual signatures

---

For more information, see `image_extraction.py` for implementation details.
