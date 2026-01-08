import json
import os

try:
    with open('g:/AIUB 10th/CVPR/Final/Assignment_3_Face_Recognzation_Transfer_Learning.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    
    print("searching for 'save'...")
    for cell in code_cells:
        content = "".join(cell)
        if "save" in content:
            print("--- FOUND SAVE ---")
            print(content[:500]) # Print first 500 chars
            print("...")
            
    print("\nsearching for 'IMG_SIZE' or 'image_size'...")
    for cell in code_cells:
        content = "".join(cell)
        if "IMG_SIZE" in content or "image_size" in content or "resize" in content:
            print("--- FOUND IMG_SIZE/RESIZE ---")
            print(content[:500])
            print("...")

    print("\nsearching for 'class_names'...")
    for cell in code_cells:
        content = "".join(cell)
        if "class_names" in content:
            print("--- FOUND CLASS_NAMES ---")
            print(content[:500])
            print("...")

except Exception as e:
    print(e)
