# process_pdfs.py

import os
import json
import sys
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
from collections import defaultdict
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class PDFProcessor:
    def __init__(self, model_path='heading_model_large8.txt', scaler_path='heading_scaler_large8.pkl'):
        """Initialize with optional pre-trained model"""
        self.model = None
        self.scaler = None
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = lgb.Booster(model_file=model_path)
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Loaded ML model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
                print("Will use rule-based extraction only")
    
    def extract_text_with_bbox(self, pdf_path):
        """Extract text with bounding box info from PDF"""
        doc = fitz.open(pdf_path)
        text_blocks = []
        
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                text_blocks.append({
                                    'text': text,
                                    'bbox': span.get("bbox", [0, 0, 0, 0]),
                                    'size': span.get("size", 12),
                                    'flags': span.get("flags", 0),
                                    'font': span.get("font", ""),
                                    'page': page_num
                                })
        
        doc.close()
        return text_blocks
    
    def extract_features(self, text_block, all_sizes):
        """Extract features for ML classification"""
        text = text_block['text']
        size = text_block['size']
        
        # Calculate size statistics
        size_ratio = size / np.mean(all_sizes) if all_sizes else 1
        size_rank = len([s for s in all_sizes if s < size]) / len(all_sizes) if all_sizes else 0.5
        
        features = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'is_all_caps': int(text.isupper()),
            'size_ratio': size_ratio,
            'size_rank': size_rank,
            'is_bold': int(text_block['flags'] & 2**4 > 0),
            'is_numbered_list': int(any(text.startswith(f"{i}.") or text.startswith(f"{i})") for i in range(1, 10)))
        }
        
        return features
    
    def classify_with_ml(self, text_blocks):
        """Classify text blocks using ML model"""
        if not self.model or not text_blocks:
            return None
        
        # Extract all font sizes
        all_sizes = [block['size'] for block in text_blocks]
        
        # Extract features
        features_list = []
        for block in text_blocks:
            features = self.extract_features(block, all_sizes)
            features_list.append(list(features.values()))
        
        # Scale features
        X = np.array(features_list)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        classes = ['Body', 'Heading', 'Title']
        
        results = []
        for i, block in enumerate(text_blocks):
            pred_class = classes[np.argmax(predictions[i])]
            results.append({
                'text': block['text'],
                'type': pred_class,
                'confidence': float(np.max(predictions[i])),
                'size': block['size'],
                'page': block['page']
            })
        
        return results
    
    def extract_with_rules(self, text_blocks):
        """Rule-based extraction fallback"""
        if not text_blocks:
            return {"title": "", "headings": []}
        
        # Find potential title (largest text in first 20% of document)
        first_blocks = text_blocks[:max(1, len(text_blocks)//5)]
        title_candidate = max(first_blocks, key=lambda x: x['size'])
        
        # Find headings (larger than average text)
        avg_size = np.mean([block['size'] for block in text_blocks])
        headings = []
        
        for block in text_blocks:
            if block['size'] > avg_size * 1.2 and block != title_candidate:
                # Skip if too long (probably body text)
                if len(block['text']) < 100:
                    headings.append({
                        'text': block['text'],
                        'level': 1 if block['size'] > avg_size * 1.5 else 2,
                        'page': block['page']
                    })
        
        return {
            "title": title_candidate['text'],
            "headings": headings[:10]  # Limit to first 10 headings
        }
    
    def process(self, pdf_path):
        """Main processing method"""
        try:
            # Extract text blocks
            text_blocks = self.extract_text_with_bbox(pdf_path)
            
            if not text_blocks:
                return {"title": "", "headings": []}
            
            # Try ML classification first
            if self.model:
                ml_results = self.classify_with_ml(text_blocks)
                
                if ml_results:
                    # Extract title and headings from ML results
                    titles = [r for r in ml_results if r['type'] == 'Title' and r['confidence'] > 0.5]
                    headings = [r for r in ml_results if r['type'] == 'Heading' and r['confidence'] > 0.4]
                    
                    # Sort by confidence
                    titles.sort(key=lambda x: x['confidence'], reverse=True)
                    headings.sort(key=lambda x: (x['page'], -x['size']))
                    
                    return {
                        "title": titles[0]['text'] if titles else "",
                        "headings": [{"text": h['text'], "level": 1, "page": h['page']} 
                                   for h in headings[:10]]
                    }
            
            # Fallback to rule-based extraction
            return self.extract_with_rules(text_blocks)
            
        except Exception as e:
            print(f"  ✗ Error in processing: {str(e)}")
            return {"title": "", "headings": []}

def main():
    # Setup paths - NOW LOOKING IN THE CORRECT DIRECTORY
    pdf_dir = Path("sample_dataset/pdfs")
    output_dir = Path("output")
    
    # Check if PDF directory exists
    if not pdf_dir.exists():
        print(f"Error: PDF directory not found at {pdf_dir}")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Find all PDF files in the directory
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    for pdf in pdf_files[:5]:  # Show first 5
        print(f"  - {pdf.name}")
    if len(pdf_files) > 5:
        print(f"  ... and {len(pdf_files) - 5} more")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    print(f"\nOutput directory: {output_dir.absolute()}")
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Process each PDF
    print("\nProcessing PDFs...")
    success_count = 0
    
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
        
        try:
            # Process the PDF
            result = processor.process(str(pdf_file))
            
            # Format output
            output = {
                "title": result["title"],
                "headings": result["headings"]
            }
            
            # Save to JSON
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            
            print(f"  ✓ Saved output to {output_file}")
            
            # Show preview
            if output['title']:
                print(f"  Title: {output['title'][:60]}{'...' if len(output['title']) > 60 else ''}")
            print(f"  Found {len(output['headings'])} headings")
            
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {pdf_file.name}: {str(e)}")
            # Save empty result on error
            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, 'w') as f:
                json.dump({"title": "", "headings": []}, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Successfully processed: {success_count}/{len(pdf_files)} files")
    print(f"Output saved to: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
