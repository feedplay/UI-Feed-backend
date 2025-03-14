import os
import re
import threading
import time
import functools
import concurrent.futures
import google.generativeai as genai
from flask import Flask, request, jsonify, make_response, session
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
import uuid

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
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24))  # Add a secret key for sessions
# Allow all origins for development, but you can restrict this in production
CORS(app, resources={r"/*": {"origins": "*"}})

# Create uploads folder - essential for Render deployment
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    
    Focus on 3-5 key principles that apply to this design , but do not go with gestalt principle. For each principle, briefly explain what it is, how it applies to this UI, and what specific improvements could be made.
    """,
    
    "cognitive": """
    Assess the cognitive load in this UI. Identify areas that might be overwhelming or confusing for users, and suggest ways to reduce cognitive burden.
    
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

# Simple in-memory cache for UI detection results
ui_detection_cache = {}
# Create a dictionary to store session-specific data
session_data = {}
lock = threading.Lock()  # Prevent concurrency issues

# Add caching decorator for expensive operations
def cached_function(expiry_seconds=300):
    """Cache decorator for expensive functions."""
    cache = {}
    lock = threading.Lock()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            with lock:
                # Check if we have a cached result that hasn't expired
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp < expiry_seconds:
                        print(f"🔄 Cache hit for {func.__name__}")
                        return result
            
            # Run the function and cache the result
            result = func(*args, **kwargs)
            with lock:
                cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator

def resize_image(image_path, max_size=(800, 800)):
    """Resize image to reduce memory usage before processing."""
    try:
        image = Image.open(image_path)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)  # Using Resampling.LANCZOS instead of ANTIALIAS
        image.save(image_path)
        print(f"📏 Resized image to max {max_size} dimensions")
    except Exception as e:
        print(f"❌ Error resizing image: {str(e)}")

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

def analyze_with_gemini(image_path, session_id):
    """Analyze the uploaded image using Gemini AI for all UX categories."""
    # Update session-specific data instead of global variables
    session_data[session_id] = {
        'image_path': image_path,
        'analysis': []
    }
    
    # Resize the image to reduce memory usage (important for Render's resources)
    resize_image(image_path)
    
    try:
        # First, check if this is a UI-related image
        if not is_ui_image(image_path):
            result = [{
                "label": "Not UI Image",
                "confidence": "High",
                "response": "The uploaded image does not appear to contain user interface elements. Please upload a screenshot of a website, app, or other digital interface for UX analysis."
            }]
            session_data[session_id]['analysis'] = result
            return result
            
        image = Image.open(image_path).convert("RGB")
        results = []
        
        # Use ThreadPoolExecutor for parallel processing with resource limits (important for Render)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_category = {}
            
            for category, prompt in UX_PROMPTS.items():
                def process_category(category, prompt):
                    try:
                        response = model.generate_content([prompt, image], stream=False)
                        analysis_text = response.text if response else "No response from Gemini AI"
                        
                        # Process the response to make it more human-like
                        processed_text = process_gemini_response(analysis_text)
                        
                        # Create a properly structured result
                        category_title = category.replace('-', ' ').title()
                        return {
                            "label": f"{category_title} Design Analysis",
                            "confidence": "High",
                            "response": processed_text
                        }
                    except Exception as e:
                        print(f"❌ Error processing {category}: {str(e)}")
                        return {
                            "label": f"{category.replace('-', ' ').title()} Design Analysis",
                            "confidence": "Low",
                            "response": f"We encountered an issue analyzing this aspect of the design. Please try again or check a different category."
                        }
                
                future = executor.submit(process_category, category, prompt)
                future_to_category[future] = category
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_category):
                category = future_to_category[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Processed {category}")
                except Exception as e:
                    print(f"🔥 Error with {category}: {str(e)}")
                    results.append({
                        "label": f"{category.replace('-', ' ').title()} Design Analysis",
                        "confidence": "Low",
                        "response": f"Error during analysis: {str(e)}"
                    })
        
        # Store results for this specific session
        with lock:
            session_data[session_id]['analysis'] = results
        
        return results
    except Exception as e:
        error_msg = f"Failed to analyze image: {str(e)}"
        print(f"Error: {error_msg}")
        result = [{
            "label": "Error",
            "confidence": "N/A",
            "response": error_msg
        }]
        session_data[session_id]['analysis'] = result
        return result

# Helper function to get or create a session ID
def get_session_id():
    """Get existing session ID from cookie or create a new one"""
    if 'session_id' not in request.cookies:
        return str(uuid.uuid4())
    return request.cookies.get('session_id')

@app.route("/preprocess", methods=["POST"])
def preprocess_image():
    """Start processing the image in the background to save time later."""
    # Get or create session ID
    session_id = get_session_id()
    
    if "image" not in request.files:
        response = make_response(jsonify({"status": "error", "message": "No file uploaded"}))
        response.set_cookie('session_id', session_id)
        return response, 400

    file = request.files["image"]
    if file.filename == "":
        response = make_response(jsonify({"status": "error", "message": "Empty file"}))
        response.set_cookie('session_id', session_id)
        return response, 400

    # Save the uploaded file
    filename = str(int(time.time())) + "_" + file.filename  # Add timestamp to prevent overwrites
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)
    
    # First check if the image is UI-related
    if not is_ui_image(image_path):
        response = make_response(jsonify({"status": "warning", "message": "The uploaded image does not appear to be UI-related. Analysis may not be relevant."}))
        response.set_cookie('session_id', session_id)
        return response, 200
    
    # Start background processing
    def background_processing():
        print(f"🔄 Starting background analysis for {filename}")
        results = analyze_with_gemini(image_path, session_id)
        print(f"✅ Background analysis complete with {len(results)} results")
    
    thread = threading.Thread(target=background_processing)
    thread.daemon = True
    thread.start()
    
    response = make_response(jsonify({"status": "success", "message": "Preprocessing started"}))
    response.set_cookie('session_id', session_id)
    return response, 200

@app.route("/analyze", methods=["POST"])
def analyze_image():
    """Handle image uploads and analyze across all UX principles."""
    # Get or create session ID
    session_id = get_session_id()
    
    if "image" not in request.files:
        response = make_response(jsonify([{"label": "Error", "confidence": "N/A", "response": "No file uploaded"}]))
        response.set_cookie('session_id', session_id)
        return response, 400

    file = request.files["image"]
    if file.filename == "":
        response = make_response(jsonify([{"label": "Error", "confidence": "N/A", "response": "Empty file"}]))
        response.set_cookie('session_id', session_id)
        return response, 400

    # Save with timestamp to prevent file overwrites
    filename = str(int(time.time())) + "_" + file.filename
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    results = analyze_with_gemini(image_path, session_id)
    
    response = make_response(jsonify(results))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.set_cookie('session_id', session_id)
    return response

@app.route("/analyze", methods=["GET"])
def get_latest_analysis():
    """Return the most recent analysis results for this session."""
    # Get session ID from cookie (or create new one)
    session_id = get_session_id()
    
    # Check if we have data for this session
    if session_id in session_data and session_data[session_id].get('image_path'):
        # If no analysis yet but we have an image path, try to generate it
        if not session_data[session_id].get('analysis'):
            image_path = session_data[session_id]['image_path']
            session_data[session_id]['analysis'] = analyze_with_gemini(image_path, session_id)
        
        analysis_results = session_data[session_id].get('analysis', [])
    else:
        # New session with no data yet
        analysis_results = []
    
    print(f"📢 Returning Analysis for session {session_id}: {len(analysis_results)} items")
    response = make_response(jsonify(analysis_results))
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.set_cookie('session_id', session_id)
    return response

# Simple cleanup function to prevent storage overflow on Render
def cleanup_old_files():
    """Delete files older than 1 hour to prevent storage overflow."""
    try:
        now = time.time()
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path) and os.stat(file_path).st_mtime < now - 3600:
                os.remove(file_path)
                print(f"🧹 Cleaned up old file: {filename}")
    except Exception as e:
        print(f"❌ Error during cleanup: {str(e)}")

# Cleanup old session data periodically as well
def cleanup_old_sessions():
    """Delete session data older than 1 hour."""
    try:
        with lock:
            now = time.time()
            sessions_to_remove = []
            for s_id, data in session_data.items():
                if 'timestamp' not in data:
                    data['timestamp'] = now  # Add timestamp for new sessions
                elif data['timestamp'] < now - 3600:
                    sessions_to_remove.append(s_id)
            
            for s_id in sessions_to_remove:
                del session_data[s_id]
                print(f"🧹 Cleaned up old session: {s_id}")
    except Exception as e:
        print(f"❌ Error during session cleanup: {str(e)}")

@app.route("/")
def home():
    # Run cleanup on homepage visits
    cleanup_old_files()
    cleanup_old_sessions()
    
    # Get or create session ID and set it in cookie
    session_id = get_session_id()
    
    # Initialize session data if needed
    if session_id not in session_data:
        with lock:
            session_data[session_id] = {'timestamp': time.time()}
    
    html_response = """
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
    
    response = make_response(html_response)
    response.set_cookie('session_id', session_id)
    return response

# Important for Render deployment
port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    print(f"🚀 Flask server is starting on port {port}...")
    try:
        # For Render deployment, use 0.0.0.0 as host and disable debug mode
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        print(f"🔥 Error starting Flask: {e}")
