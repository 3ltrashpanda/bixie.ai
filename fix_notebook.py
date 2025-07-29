#!/usr/bin/env python3
"""
Script to fix the train_benchmark.ipynb notebook
"""

import json
import re

def fix_notebook():
    # Read the original notebook
    with open('notebooks/train_benchmark.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Fix 1: Correct the data loading logic in Cell 4
    cell_4_source = notebook['cells'][4]['source']
    
    # Replace the incorrect data loading
    old_pattern = r'with open\("../data/bixie_V\.json"\) as f:\s*\n\s*vuln = json\.load\(f\)\s*\n\s*for v in vuln:\s*\n\s*v\["label"\] = 1\s*\n\s*with open\("../data/bixie_V\.json"\) as f:\s*\n\s*clean = json\.load\(f\)\s*\n\s*for c in clean:\s*\n\s*c\["label"\] = 0'
    
    new_data_loading = '''# Load vulnerable samples
with open("../data/bixie_V.json") as f:
    vuln = json.load(f)
    for v in vuln:
        v["label"] = 1

# Load clean samples (non-vulnerable)
with open("../data/bixie_noV.json") as f:
    clean = json.load(f)
    for c in clean:
        c["label"] = 0'''
    
    # Replace the data loading section
    cell_4_source = re.sub(old_pattern, new_data_loading, cell_4_source, flags=re.MULTILINE | re.DOTALL)
    
    # Add print statement to show data loading results
    cell_4_source += '\n\nprint(f"Loaded {len(vuln)} vulnerable samples and {len(clean)} clean samples")'
    
    notebook['cells'][4]['source'] = cell_4_source
    
    # Fix 2: Add torch import to Cell 6
    cell_6_source = notebook['cells'][6]['source']
    if 'import torch' not in cell_6_source:
        cell_6_source = 'import torch\n' + cell_6_source
    notebook['cells'][6]['source'] = cell_6_source
    
    # Fix 3: Add print statements to Cell 5 to show data statistics
    cell_5_source = notebook['cells'][5]['source']
    cell_5_source += '\n\nprint(f"Total samples: {len(texts)}")\nprint(f"Vulnerable samples: {sum(labels)}")\nprint(f"Clean samples: {len(labels) - sum(labels)}")'
    notebook['cells'][5]['source'] = cell_5_source
    
    # Fix 4: Add print statements to Cell 7
    cell_7_source = notebook['cells'][7]['source']
    cell_7_source += '\n\nprint(f"Training samples: {len(train_texts)}")\nprint(f"Validation samples: {len(val_texts)}")'
    notebook['cells'][7]['source'] = cell_7_source
    
    # Fix 5: Replace the extract_embeddings function with a text-based version
    cell_9_source = notebook['cells'][9]['source']
    new_extract_function = '''def extract_embeddings_from_texts(texts, labels, embedder):
    """Extract embeddings from text data using SAFEEmbedder"""
    X, y, failed = [], [], 0
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        try:
            # Convert text to bytes for the embedder
            text_bytes = text.encode('utf-8')
            emb = embedder.embed_binary_string(text_bytes)
            if emb is not None:
                X.append(emb)
                y.append(label)
            else:
                failed += 1
        except Exception as e:
            print(f"Failed to embed sample {i}: {e}")
            failed += 1
    
    return np.array(X), np.array(y), failed'''
    
    notebook['cells'][9]['source'] = new_extract_function
    
    # Fix 6: Update Cell 13 to use the new function and data
    cell_13_source = notebook['cells'][13]['source']
    new_embedding_call = '''embedder = SAFEEmbedder()
X, y, failed = extract_embeddings_from_texts(texts, labels, embedder)
print(f"Extracted {len(X)} embeddings ({failed} failed)")
print(f"Embedding shape: {X.shape}")'''
    
    notebook['cells'][13]['source'] = new_embedding_call
    
    # Fix 7: Add print statements to Cell 15
    cell_15_source = notebook['cells'][15]['source']
    cell_15_source += '\n\nprint(f"Training samples: {len(X_train)}")\nprint(f"Test samples: {len(X_test)}")'
    notebook['cells'][15]['source'] = cell_15_source
    
    # Fix 8: Update target names in Cell 17
    cell_17_source = notebook['cells'][17]['source']
    cell_17_source = cell_17_source.replace('target_names=["Good", "Vulnerable"]', 'target_names=["Clean", "Vulnerable"]')
    notebook['cells'][17]['source'] = cell_17_source
    
    # Fix 9: Update display labels in Cell 19
    cell_19_source = notebook['cells'][19]['source']
    cell_19_source = cell_19_source.replace('display_labels=["Good", "Vulnerable"]', 'display_labels=["Clean", "Vulnerable"]')
    notebook['cells'][19]['source'] = cell_19_source
    
    # Write the fixed notebook
    with open('notebooks/train_benchmark_fixed.ipynb', 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print("Fixed notebook saved as notebooks/train_benchmark_fixed.ipynb")

if __name__ == "__main__":
    fix_notebook() 