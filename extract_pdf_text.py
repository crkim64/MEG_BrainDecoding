from pdfminer.high_level import extract_text
import re

pdf_path = '/home/kcr/DecodingMEG AgentAI/Papers/ImageDecodingMEG.pdf'
try:
    text = extract_text(pdf_path)
    # Extract sections related to Architecture
    # Look for "Brain Module", "Architecture", "Subject Layer", "Spatial Attention"
    keywords = ["Brain Module", "architecture", "spatial attention", "subject layer", "dilated", "convolution", "residual"]
    
    print(f"--- Full Text Length: {len(text)} chars ---")
    
    # Simple keyword search context
    for kw in keywords:
        print(f"\n--- Context for '{kw}' ---")
        matches = [m.start() for m in re.finditer(kw, text, re.IGNORECASE)]
        for m in matches[:5]: # Show first 5 occurrences
            start = max(0, m - 200)
            end = min(len(text), m + 200)
            print(f"...{text[start:end].replace(chr(10), ' ')}...")

except Exception as e:
    print(f"Error reading PDF: {e}")
