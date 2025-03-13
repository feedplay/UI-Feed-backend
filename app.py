import os
import re
import threading
import time
import functools
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Check if API key is available
if not GEMINI_API_KEY:
    raise ValueError("❌ ERROR: Missing Gemini API Key in .env file!")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create uploads folder if not exists

# Define UX principles and their analysis prompts with explicit output format instructions
UX_PROMPTS = {
    "visual": """
    Analyze this UI screenshot for visual design consistency issues. Consider color palette, typography, spacing, and alignment. Identify any inconsistencies and suggest improvements.
    
    Write in a friendly, conversational tone as if you're a helpful UX designer giving advice to a colleague. Avoid using markdown symbols, bullet points with asterisks, or excessive formatting.
    
    Focus on 3-5 key points with practical suggestions. Include specific action items.
    """,
    
    "ux-laws": """
    Evaluate this UI based on UX laws and principles such as Fitts's Law, Hick's Law, and Jakob's Law. Identify any violations and suggest improvements.
    
    Write in a friendly, conversational tone as if you're a helpful UX designer giving advice to a colleague. Avoid using markdown symbols, bullet points with asterisks, or excessive formatting.
    
    Focus on 3-5 key principles that apply to this design. For each principle, briefly explain what it is, how it applies to this UI, and what specific improvements could be made.
    """,
    
    "cognitive": """
    Assess the cognitive load in this UI. Identify areas that might be overwhelming or confusing for users, and suggest ways to reduce complexity.
    
    Write in a friendly, conversational tone as if you're a helpful UX designer giving advice to a colleague. Avoid using markdown symbols, bullet points with asterisks, or excessive formatting.
    
    Focus on 3-5 specific areas where cognitive load could be reduced. For each area, explain why it might be causing cognitive strain and provide a specific solution.
    """,
    
    "psychological": """
    Analyze the psychological effects of this UI design. How does it influence user behavior and perception? Consider aspects like color psychology, visual hierarchy, and emotional response.
    
    Write in a friendly, conversational tone as if you're a helpful UX designer giving advice to a colleague. Avoid using markdown symbols, bullet points with asterisks, or excessive formatting.
    
    Focus on 3-5 psychological aspects of the design. For each aspect, explain its current impact and suggest how it could be optimized for better user experience.
    """,
    
    "gestalt": """
    Evaluate how this UI applies Gestalt principles (proximity, similarity, continuity, closure, etc.). Identify any areas where these principles could be better applied.
    
    Write in a friendly, conversational tone as if you're a helpful UX designer giving advice to a colleague. Avoid using markdown symbols, bullet points with asterisks, or excessive formatting.
    
    Focus on 3-5 Gestalt principles that are most relevant to this design. For each principle, explain how it's currently being used (or not), and suggest specific improvements.
    """
}

# UI Detection prompt
UI_DETECTION_PROMPT = """
Analyze this image and determine if it contains a user interface (UI) element such as:
- Website or web application interface
- Mobile app screen
- Software dashboard
- Digital product interface
- UI wireframe or mockup
- Control panel or settings screen

Respond with just 'YES' if this is a UI-related image that could be analyzed for UX principles,
or 'NO' if this is not a UI-related image (e.g., photograph of a person, landscape, object, etc.).
"""

# Simple in-memory cache for UI detection results and analysis results
ui_detection_cache = {}
latest_analysis = []
latest_image_path = None
lock = threading.Lock()  # Global lock for thread safety

# Add caching decorator for expensive operations
def cached_function(expiry_seconds=300):
    """Cache decorator for expensive functions."""
    cache = {}
    cache_lock = threading.Lock()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            with cache_lock:
                # Check if we have a cached result that hasn't expired
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < expiry_seconds:
                        print(f"🔄 Cache hit for {func.__name__}")
                        return result
            
            # Run the function and cache the result
            result = func(*args, **kwargs)
            with cache_lock:
                cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator

# Apply caching to the UI detection function
@cached_function(expiry_seconds=600)  # Cache UI detection results for 10 minutes
def is_ui_image(image_path):
    """Determine if the uploaded image is UI-related."""
    # First check in-memory cache
    if image_path in ui_detection_cache:
        print(f"🔄 UI detection cache hit for {os.path.basename(image_path)}")
        return ui_detection_cache[image_path]
        
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Ask Gemini if this image contains UI elements
        response = model.generate_content([UI_DETECTION_PROMPT, image], stream=False)
        result = response.text.strip().upper()
        
        # Check if the response indicates this is a UI image
        is_ui = "YES" in result
        
        # Store in cache
        ui_detection_cache[image_path] = is_ui
        
        print(f"🔍 UI detection for {os.path.basename(image_path)}: {'✅ UI detected' if is_ui else '❌ Not UI'}")
        return is_ui
    except Exception as e:
        print(f"❌ Error during UI detection: {str(e)}")
        # In case of errors, default to not a UI
        return False

# Function to process AI responses
def process_gemini_response(text):
    """Clean up and humanize Gemini AI responses."""
    
    # Remove excessive markdown
    cleaned_text = text
    
    # Replace multiple newlines with double newlines for paragraph spacing
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Replace markdown headers with clean text
    cleaned_text = re.sub(r'^#\s+(.*?)$', r'\1:', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^##\s+(.*?)$', r'\1:', cleaned_text, flags=re.MULTILINE)
    
    # Clean up bullet points
    cleaned_text = re.sub(r'^\*\s+', '• ', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^-\s+', '• ', cleaned_text, flags=re.MULTILINE)
    
    # Replace bold markdown with actual text (maintain emphasis with HTML)
    cleaned_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', cleaned_text)
    cleaned_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', cleaned_text)
    
    # Add structure if not present
    if not re.search(r'(Issues|Improvements|Recommendations|Analysis):', cleaned_text, re.IGNORECASE):
        paragraphs = cleaned_text.split('\n\n')
        if len(paragraphs) >= 2:
            # Simple structure: First paragraph is analysis, rest are recommendations
            analysis = paragraphs[0]
            recommendations = '\n\n'.join(paragraphs[1:])
            cleaned_text = f"<strong>Analysis:</strong>\n{analysis}\n\n<strong>Recommendations:</strong>\n{recommendations}"
    
    return cleaned_text

def analyze_with_gemini(image_path):
    """Analyze the uploaded image using Gemini AI for all UX categories."""
    global latest_image_path, latest_analysis
    latest_image_path = image_path
    
    try:
        # First, check if this is a UI-related image
        if not is_ui_image(image_path):
            return [{
                "label": "Not UI Image",
                "confidence": "High",
                "response": "The uploaded image does not appear to contain user interface elements. Please upload a screenshot of a website, app, or other digital interface for UX analysis."
            }]
            
        image = Image.open(image_path).convert("RGB")
        results = []
        result_lock = threading.Lock()
        
        # Process categories in parallel
        def process_category(category, prompt):
            try:
                # Create specific prompt for better responses
                response = model.generate_content([prompt, image], stream=False)
                analysis_text = response.text if response else "No response from Gemini AI"
                
                # Process the response to make it more human-like
                processed_text = process_gemini_response(analysis_text)
                
                # Create a properly structured result
                category_title = category.replace('-', ' ').title()
                result = {
                    "label": f"{category_title} Design Analysis",
                    "confidence": "High",
                    "response": processed_text
                }
                
                with result_lock:
                    results.append(result)
                
                print(f"✅ Processed {category_title}")
            except Exception as e:
                print(f"❌ Error processing {category}: {str(e)}")
                with result_lock:
                    results.append({
                        "label": f"{category.replace('-', ' ').title()} Design Analysis",
                        "confidence": "Low",
                        "response": f"We encountered an issue analyzing this aspect of the design. Please try again or check a different category."
                    })
        
        # Create and start threads for each category
        threads = []
        for category, prompt in UX_PROMPTS.items():
            t = threading.Thread(target=process_category, args=(category, prompt))
            t.daemon = True
            threads.append(t)
            t.start()
        
        # Wait for all threads to complete
        for t in threads:
            t.join()
        
        # Store results for GET requests
        with lock:
            latest_analysis = results
        
        return results
    except Exception as e:
        error_msg = f"Failed to analyze image: {str(e)}"
        print(f"Error: {error_msg}")
        return [{
            "label": "Error",
            "confidence": "N/A",
            "response": error_msg
        }]

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    """Start processing the image in the background to save time later."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty file"}), 400

    # Save the uploaded file
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)
    
    # First check if the image is UI-related
    if not is_ui_image(image_path):
        return jsonify({"status": "warning", "message": "The uploaded image does not appear to be UI-related. Analysis may not be relevant."}), 200
    
    # Start background processing
    def background_processing():
        print(f"🔄 Starting background analysis for {file.filename}")
        analyze_with_gemini(image_path)
        print(f"✅ Background analysis complete")
    
    thread = threading.Thread(target=background_processing)
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "success", "message": "Preprocessing started"}), 200

@app.route("/analyze", methods=["POST"])
def analyze_image():
    """Handle image uploads and analyze across all UX principles."""
    if "image" not in request.files:
        return jsonify([{"label": "Error", "confidence": "N/A", "response": "No file uploaded"}]), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify([{"label": "Error", "confidence": "N/A", "response": "Empty file"}]), 400

    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    results = analyze_with_gemini(image_path)
    
    response = make_response(jsonify(results))
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/analyze", methods=["GET"])
def get_latest_analysis():
    """Return the most recent analysis results."""
    global latest_analysis, latest_image_path
    
    # Debug information
    print(f"📢 Latest Analysis Data: {latest_analysis}")
    print(f"📢 Latest Image Path: {latest_image_path}")
    
    # If we have an image path but no analysis yet, try to generate it
    if latest_image_path and (not latest_analysis or len(latest_analysis) == 0):
        latest_analysis = analyze_with_gemini(latest_image_path)
    
    response = make_response(jsonify(latest_analysis if latest_analysis else []))
    response.headers["Access-Control-Allow-Origin"] = "*"
    print(f"📢 Returning Response: {response.get_data(as_text=True)}")
    return response

@app.route("/")
def home():
    return """
    <html>
    <head><title>UX Analysis API</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #4A90E2; }
        .status { padding: 10px; background-color: #E3F2FD; border-radius: 4px; }
        code { background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }
    </style>
    </head>
    <body>
        <h1>UX Analysis API</h1>
        <div class="status">✅ Flask backend with Gemini Vision is running!</div>
        <p>This API analyzes UI screenshots and provides feedback on:</p>
        <ul>
            <li>Visual Design</li>
            <li>UX Laws</li>
            <li>Cognitive Load</li>
            <li>Psychological Effects</li>
            <li>Gestalt Principles</li>
        </ul>
        <p>Upload images to <code>/analyze</code> to get started.</p>
        <p><strong>Note:</strong> Only UI-related images (websites, apps, software interfaces) will be processed.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    print("🚀 Starting Flask server on http://0.0.0.0:5000...")
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"🔥 Error starting Flask: {e}")
