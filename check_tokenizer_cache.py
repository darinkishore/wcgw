#!/usr/bin/env python
"""Check tokenizer cache locations and try to load it"""

import os
from pathlib import Path
import tokenizers

print("=== Checking Tokenizer Cache ===\n")

# 1. Check HuggingFace cache locations
print("1. HuggingFace cache locations:")
hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
print(f"   HF_HOME: {hf_home}")
print(f"   Exists: {os.path.exists(hf_home)}")

# Check for Xenova tokenizer
hub_path = Path(hf_home) / "hub"
if hub_path.exists():
    print("\n2. Looking for Xenova/claude-tokenizer:")
    for item in hub_path.rglob("*"):
        if "Xenova" in str(item) and "claude" in str(item):
            print(f"   Found: {item}")
            if item.name == "tokenizer.json" and item.is_file():
                print(f"   -> Tokenizer file: {item}")
                print(f"   -> Size: {item.stat().st_size} bytes")

# 3. Try different loading methods
print("\n3. Testing different loading methods:")

# Method 1: Direct file path
tokenizer_files = list(Path(hf_home).rglob("**/tokenizer.json"))
if tokenizer_files:
    for tf in tokenizer_files:
        if "Xenova" in str(tf) and "claude" in str(tf):
            print(f"\n   Trying direct file load: {tf}")
            try:
                tokenizer = tokenizers.Tokenizer.from_file(str(tf))
                print("   ✓ SUCCESS: Direct file load works!")
                break
            except Exception as e:
                print(f"   ✗ Failed: {e}")

# Method 2: from_pretrained with local_files_only
print("\n   Trying from_pretrained with local_files_only=True:")
try:
    tokenizer = tokenizers.Tokenizer.from_pretrained("Xenova/claude-tokenizer", local_files_only=True)
    print("   ✓ SUCCESS: from_pretrained with local_files_only works!")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# Method 3: Set HF_HUB_OFFLINE
print("\n   Trying with HF_HUB_OFFLINE=1:")
os.environ["HF_HUB_OFFLINE"] = "1"
try:
    tokenizer = tokenizers.Tokenizer.from_pretrained("Xenova/claude-tokenizer")
    print("   ✓ SUCCESS: Works with HF_HUB_OFFLINE=1!")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# 4. Check environment
print("\n4. Environment variables:")
for var in ["HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_CACHE"]:
    print(f"   {var}: {os.environ.get(var, 'Not set')}")

print("\n5. Suggested fix:")
if tokenizer_files:
    tf = next((tf for tf in tokenizer_files if "Xenova" in str(tf) and "claude" in str(tf)), None)
    if tf:
        print(f"   Load directly from: {tf}")
        print(f"   Or set: export HF_HOME={hf_home}")