#!/usr/bin/env python3
"""Process all specific PDFs for the challenge"""

import os
import shutil
from pathlib import Path
from process_pdfs import main as process_main

def setup_and_process():
    """Setup PDFs and process them"""
    # List of specific PDFs
    pdf_list = [
        "file01.pdf",
        "file02.pdf",
        "file03.pdf",
        "file04.pdf",
        "file05.pdf",
        "TOPJUMP-PARTY-INVITATION-20161003-V01.pdf",
        "STEMPathwaysFlyer.pdf",
        "E0H1CM114.pdf",
        "E0CCG5S312.pdf",
        "E0CCG5S239.pdf"
    ]
    
    # Ensure directories exist
    Path("sample_dataset/pdfs").mkdir(parents=True, exist_ok=True)
    Path("sample_dataset/outputs").mkdir(parents=True, exist_ok=True)
    
    # Copy PDFs from Downloads if they exist
    downloads_dir = Path(r"C:\Users\Lenovo\Downloads")
    pdfs_dir = Path("sample_dataset/pdfs")
    
    print("Setting up PDFs...")
    for pdf in pdf_list:
        source = downloads_dir / pdf
        dest = pdfs_dir / pdf
        
        if source.exists():
            shutil.copy2(source, dest)
            print(f"  ✓ Copied {pdf}")
        else:
            print(f"  ✗ Not found: {pdf}")
    
    print("\nProcessing PDFs...")
    process_main()

if __name__ == "__main__":
    setup_and_process()