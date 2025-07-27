#!/usr/bin/env python3
"""
Training script for the PDF heading classification model
Generates synthetic data and trains the LightGBM model
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cluster import KMeans
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import warnings
warnings.filterwarnings('ignore')

# Feature columns
FEATURE_COLS = [
    'char_count', 'word_count', 'is_all_caps',
    'size_ratio', 'size_rank', 'is_bold', 'is_numbered_list'
]

def extract_features_from_pdf(pdf_path):
    """Extract features from PDF for ML processing"""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return pd.DataFrame()

    all_blocks_data = []
    for page in doc:
        if page.rect.height == 0 or page.rect.width == 0:
            continue

        blocks = sorted(page.get_text("blocks"), key=lambda b: b[1])
        if not blocks:
            continue

        # Merge adjacent text blocks
        merged_blocks = []
        for b in blocks:
            if b[6] != 0:  # Skip non-text blocks
                continue
            text = b[4].strip()
            if not text:
                continue
            
            if merged_blocks and abs(b[1] - merged_blocks[-1]['bbox'][3]) < 10:
                prev_block = merged_blocks[-1]
                prev_block['text'] += f" {text}"
                prev_block['bbox'] = (
                    min(prev_block['bbox'][0], b[0]), 
                    min(prev_block['bbox'][1], b[1]),
                    max(prev_block['bbox'][2], b[2]), 
                    max(prev_block['bbox'][3], b[3])
                )
            else:
                merged_blocks.append({
                    'text': text, 
                    'bbox': (b[0], b[1], b[2], b[3]), 
                    'page_num': page.number
                })

        if not merged_blocks:
            continue
            
        page_df = pd.DataFrame(merged_blocks)
        page_df['y0'] = page_df['bbox'].apply(lambda x: x[1])
        page_df['y1'] = page_df['bbox'].apply(lambda x: x[3])
        page_df['block_height'] = page_df['y1'] - page_df['y0']

        # Feature Engineering
        page_df['char_count'] = page_df['text'].str.len()
        page_df['word_count'] = page_df['text'].str.split().str.len()
        page_df['is_all_caps'] = page_df['text'].str.isupper().astype(int)
        page_df['is_top_of_page'] = (page_df['y0'] / page.rect.height < 0.25).astype(int)
        page_df['is_centered'] = page_df.apply(
            lambda row: 1 if 0.4 < ((row['bbox'][0] + row['bbox'][2]) / 2 / page.rect.width) < 0.6 else 0, 
            axis=1
        )

        # Relative size features
        base_height = page_df['block_height'].median()
        page_df['height_ratio'] = page_df['block_height'] / (base_height + 1e-6)
        page_df['height_rank'] = page_df['block_height'].rank(method='dense', ascending=False)

        # Whitespace after
        page_df = page_df.sort_values(by='y0').reset_index(drop=True)
        page_df['next_y0'] = page_df['y0'].shift(-1).fillna(page.rect.height)
        page_df['whitespace_after'] = page_df['next_y0'] - page_df['y1']
        page_df.loc[page_df['whitespace_after'] < 0, 'whitespace_after'] = 0

        all_blocks_data.append(page_df)

    doc.close()
    
    if not all_blocks_data:
        return pd.DataFrame()
        
    final_df = pd.concat(all_blocks_data, ignore_index=True)
    return final_df.drop(columns=['next_y0'], errors='ignore').fillna(0)

def generate_doc_style_data(filename_base):
    """Generates document-style training data"""
    pdf_path = f"{filename_base}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter
    
    ts_h, h1_h, h2_h, bs_h = 24, 18, 14, 11
    c.setFont("Helvetica-Bold", ts_h)
    c.drawString(inch, h - inch, "A Report Title")
    c.setFont("Helvetica-Bold", h1_h)
    c.drawString(inch, h - 2*inch, "1. H1 Heading")
    c.setFont("Helvetica", bs_h)
    c.drawString(inch*1.1, h - 2.5*inch, "Some body text.")
    c.setFont("Helvetica-Bold", h2_h)
    c.drawString(inch*1.1, h - 3*inch, "1.1. An H2 Heading")
    c.save()

    df = extract_features_from_pdf(pdf_path)
    df['label'] = 'Body'

    # Label based on block_height
    df.loc[df['block_height'].between(ts_h - 5, ts_h + 5), 'label'] = 'Title'
    df.loc[df['block_height'].between(h1_h - 4, h1_h + 4), 'label'] = 'Heading'
    df.loc[df['block_height'].between(h2_h - 3, h2_h + 3), 'label'] = 'Heading'

    os.remove(pdf_path)
    return df

def generate_poster_style_data(filename_base):
    """Generates poster-style training data"""
    pdf_path = f"{filename_base}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)
    w, h = letter
    
    h1_h, bs_h = 36, 12
    c.setFont("Helvetica-Bold", h1_h)
    c.drawCentredString(w/2, h*0.7, "A BIG CENTERED HEADING")
    c.setFont("Helvetica", bs_h)
    c.drawCentredString(w/2, h*0.2, "Some small text at the bottom.")
    c.save()

    df = extract_features_from_pdf(pdf_path)
    df['label'] = 'Body'
    df.loc[df['block_height'].between(h1_h - 5, h1_h + 5), 'label'] = 'Heading'

    os.remove(pdf_path)
    return df

def main():
    print("=== Training PDF Heading Classification Model ===")
    
    # Generate training data
    all_data = []
    
    print("\n1. Generating synthetic training data...")
    for i in range(800):
        if i % 100 == 0:
            print(f"   Generated {i}/800 document-style samples")
        all_data.append(generate_doc_style_data(f"doc_{i}"))
    
    for i in range(200):
        if i % 50 == 0:
            print(f"   Generated {i}/200 poster-style samples")
        all_data.append(generate_poster_style_data(f"poster_{i}"))
    
    print(f"\n✅ Generated {len(all_data)} total training samples")
    
    # Combine and prepare data
    df = pd.concat(all_data, ignore_index=True)
    
    # Ensure all feature columns exist
    for col in FEATURE_COLS:
        if col not in df.columns:
            if col == 'size_ratio':
                df[col] = df.get('height_ratio', 0)
            elif col == 'size_rank':
                df[col] = df.get('height_rank', 0)
            elif col == 'is_bold':
                df[col] = 0
            elif col == 'is_numbered_list':
                df[col] = df['text'].str.match(r'^\d+\.').astype(int)
            else:
                df[col] = 0
    
    # Clean data
    df = df.dropna(subset=FEATURE_COLS + ['label'])
    df = df[df['char_count'] > 2]
    
    # Save training data
    df.to_csv('training_data_large8.csv', index=False)
    print(f"\n2. Saved training data: {len(df)} rows")
    print("   Label distribution:")
    print(df['label'].value_counts())
    
    # Train model
    print("\n3. Training LightGBM model...")
    X = df[FEATURE_COLS]
    label_map = {'Body': 0, 'Title': 1, 'Heading': 2}
    y_encoded = df['label'].map(label_map)
    
    # Check for missing labels
    if y_encoded.isnull().any():
        print("   Warning: Found invalid labels, dropping them")
        valid_indices = y_encoded.notnull()
        X = X[valid_indices]
        y_encoded = y_encoded[valid_indices]
    
    # Train model
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight='balanced',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        verbose=-1
    )
    
    model.fit(X, y_encoded)
    
    # Save model
    model.booster_.save_model('heading_model_large8.txt')
    print("\n✅ Model trained and saved to 'heading_model_large8.txt'")
    
    # Show feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\n4. Feature Importance:")
    print(feature_importance_df)

if __name__ == "__main__":
    main()