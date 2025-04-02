import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    print(f"Error configuring API: {str(e)}")
    exit(1)

# Function to list available models
def list_available_models():
    try:
        # Get list of models
        models = genai.list_models()
        
        print("Available models:")
        print("-" * 50)
        
        # Print details for each model
        for model in models:
            print(f"Model Name: {model.name}")
            print(f"Description: {model.description}")
            print(f"Supported Methods: {', '.join(model.supported_generation_methods)}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error listing models: {str(e)}")

# Simple function to test if a specific model exists
def check_model_availability(model_name: str):
    try:
        models = genai.list_models()
        available_models = [m.name for m in models]
        
        if f"models/{model_name}" in available_models:
            print(f"Model '{model_name}' is available")
        else:
            print(f"Model '{model_name}' is not found in available models")
            
    except Exception as e:
        print(f"Error checking model availability: {str(e)}")

if __name__ == "__main__":
    # List all available models
    list_available_models()
    
    # Optionally check specific models
    print("\nChecking specific models:")
    check_model_availability("gemini-pro")
    check_model_availability("gemini-1.5-pro")