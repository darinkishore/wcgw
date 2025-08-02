#!/usr/bin/env python
"""Debug script to test tokenizer download and identify SSL issues"""

import os
import ssl
import certifi
import urllib.request
from huggingface_hub import hf_hub_download
import tokenizers

print("=== Tokenizer Debug Script ===\n")

# 1. Check SSL certificates
print("1. SSL Certificate Info:")
print(f"   Default CA bundle: {ssl.get_default_verify_paths()}")
print(f"   Certifi CA bundle: {certifi.where()}")
print(f"   OpenSSL version: {ssl.OPENSSL_VERSION}")

# 2. Test basic HTTPS connection
print("\n2. Testing HTTPS connection to HuggingFace:")
try:
    response = urllib.request.urlopen("https://huggingface.co")
    print(f"   ✓ Connection successful: {response.status}")
except Exception as e:
    print(f"   ✗ Connection failed: {e}")

# 3. Test HuggingFace Hub
print("\n3. Testing HuggingFace Hub download:")
try:
    # Try to download a small file
    test_file = hf_hub_download(repo_id="bert-base-uncased", filename="config.json")
    print(f"   ✓ HF Hub download successful: {test_file}")
except Exception as e:
    print(f"   ✗ HF Hub download failed: {e}")

# 4. Test tokenizer download
print("\n4. Testing tokenizer download:")
try:
    tokenizer = tokenizers.Tokenizer.from_pretrained("Xenova/claude-tokenizer")
    print(f"   ✓ Tokenizer download successful")
except Exception as e:
    print(f"   ✗ Tokenizer download failed: {e}")
    print(f"   Error type: {type(e).__name__}")

# 5. Environment variables that might help
print("\n5. Relevant environment variables:")
env_vars = ["REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "HF_HUB_OFFLINE", 
            "HF_HOME", "HUGGINGFACE_HUB_CACHE", "SSL_CERT_FILE", "SSL_CERT_DIR"]
for var in env_vars:
    value = os.environ.get(var, "Not set")
    print(f"   {var}: {value}")

# 6. Try with SSL verification disabled (NOT for production!)
print("\n6. Testing with SSL verification disabled (diagnostic only):")
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
try:
    tokenizer = tokenizers.Tokenizer.from_pretrained("Xenova/claude-tokenizer")
    print("   ✓ Works without SSL verification - this indicates a certificate issue")
except Exception as e:
    print(f"   ✗ Still fails: {e}")

print("\n=== Possible Solutions ===")
print("1. Update certificates: pip install -U certifi")
print("2. Set environment variable: export REQUESTS_CA_BUNDLE=$(python -m certifi)")
print("3. For Ubuntu/Debian: sudo apt-get update && sudo apt-get install ca-certificates")
print("4. Pre-download the tokenizer on another machine and copy the cache")