import sys
import os
import ssl
import socket
import traceback

print("--- DIAGNOSTIC REPORT ---")

# 1. Check Python & SSL Environment
print(f"\n1. ENVIRONMENT:")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
try:
    print(f"SSL Version (OpenSSL): {ssl.OPENSSL_VERSION}")
except:
    print("SSL Version: Could not retrieve")

# 2. Check Urllib3 Version (The source of your warning)
try:
    import urllib3
    print(f"Urllib3 Version: {urllib3.__version__}")
except ImportError:
    print("Urllib3: Not installed")

# 3. Check Certifi (The Certificate bundle
try:
    import certifi
    print(f"Certifi Path: {certifi.where()}")
    # FORCE FIX TEST: Uncommenting this simulates the fix
    os.environ['SSL_CERT_FILE'] = certifi.where()
    print("-> Applied Certifi override for this test.")
except ImportError:
    print("Certifi: Not installed (CRITICAL on macOS)")

# 4. Check API Key
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"\n2. API KEY: Found (Length: {len(api_key)})")
    print(f"   Key starts with: {api_key[:4]}...")
else:
    print("\n2. API KEY: [ERROR] Not found in .env or environment variables.")

# 5. Network Connectivity Test (DNS)
print(f"\n3. NETWORK DNS TEST:")
host = "generativelanguage.googleapis.com"
try:
    ip = socket.gethostbyname(host)
    print(f"   Success: {host} resolved to {ip}")
except Exception as e:
    print(f"   [ERROR] DNS Resolution failed: {e}")

# 6. Actual API Handshake Test
print(f"\n4. API HANDSHAKE TEST:")
if not api_key:
    print("   Skipping test (No API Key)")
else:
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        print("   Attempting to generate content with gemini-flash-latest...")
        response = client.models.generate_content(
            model='gemini-flash-latest',
            contents="Hello, can you hear me?"
        )
        print(f"   [SUCCESS] Response received: {response.text}")
    except Exception as e:
        print("   [FAILURE] API Call Failed.")
        print("   Traceback:")
        traceback.print_exc()

print("\n--- END REPORT ---")
