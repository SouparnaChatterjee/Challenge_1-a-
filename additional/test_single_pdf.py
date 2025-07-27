#!/usr/bin/env python3
"""Test script for processing individual PDFs"""

import json
import sys
import time
from process_pdfs import PDFProcessor

def test_pdf(pdf_path):
    """Test a single PDF file"""
    print(f"Testing PDF: {pdf_path}")
    print("-" * 50)
    
    # Initialize processor
    processor = PDFProcessor(model_path="heading_model_large8.txt")
    
    # Process PDF
    start_time = time.time()
    result = processor.process_pdf(pdf_path)
    end_time = time.time()
    
    # Display results
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print("\nJSON Output:")
    print(json.dumps(result, indent=2))
    
    # Save to file
    output_file = f"{pdf_path.rsplit('.', 1)[0]}_output.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_single_pdf.py <pdf_path>")
        sys.exit(1)
    
    test_pdf(sys.argv[1])
