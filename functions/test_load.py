import sys
import os

# Create a mock Credential to avoid firebase_admin init error if no creds
import firebase_admin
from firebase_admin import credentials

print("Attempting to import main...")
try:
    import main
    print("SUCCESS: main imported.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
except Exception as e:
    print(f"GENERAL ERROR: {e}")
