#!/usr/bin/env python
"""Script to pre-download the tokenizer and show where it's cached"""

import os
import tokenizers
from pathlib import Path

print("Downloading tokenizer...")

try:
    # Download the tokenizer
    tokenizer = tokenizers.Tokenizer.from_pretrained("Xenova/claude-tokenizer")
    print("✓ Tokenizer downloaded successfully!")
    
    # Find the cache location
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    print(f"\nTokenizer should be cached in: {hf_home}")
    
    # List the cache contents
    hub_path = Path(hf_home) / "hub"
    if hub_path.exists():
        print("\nCache contents:")
        for item in hub_path.rglob("*"):
            if item.is_file() and "Xenova" in str(item):
                print(f"  {item.relative_to(hf_home)}")
    
    print("\nTo use this cache on another machine:")
    print(f"1. Copy the entire directory: {hf_home}")
    print("2. Set HF_HOME environment variable to the copied location")
    print("3. Or set HF_HUB_OFFLINE=1 to use cached version only")
    
except Exception as e:
    print(f"✗ Failed to download: {e}")
    print("\nTry setting SSL cert bundle:")
    print("  export REQUESTS_CA_BUNDLE=$(python -m certifi)")
    print("  export SSL_CERT_FILE=$(python -m certifi)")