from flask import Flask, request, render_template, jsonify, Response
import requests
import json
import os
from flask import stream_with_context
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get API key from environment variable
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "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")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def get_response(user_input):
    # Check for common questions first and provide direct answers
    user_input_lower = user_input.lower()
    
    # Handle common questions directly
    if any(name_word in user_input_lower for name_word in ["name", "who are you", "who're you"]):
        return "I am Oliver Cowdery, born in Wells, Vermont. Today is May 15, 1829, a day I shall never forget."
    
    if any(morning_word in user_input_lower for morning_word in ["morning", "today", "happen"]):
        return "This morning, John the Baptist appeared to us as we prayed, his voice clear and powerful as he laid his hands upon us, saying, 'Upon you my fellow servants, in the name of Messiah, I confer the priesthood of Aaron.' We then baptized each other in the Susquehanna River."
    
    if "joseph" in user_input_lower or "smith" in user_input_lower:
        return "Joseph Smith is a man chosen of God, a prophet through whom the Lord is working to restore His church in these latter days. We have been laboring together on the translation of ancient records by the gift and power of God."
    
    # For all other questions, use Claude API
    return call_claude_api(create_claude_prompt(user_input))

def create_claude_prompt(user_input):
    """Create a prompt for Claude that provides context about Oliver Cowdery"""
    return f"""
    You are impersonating Oliver Cowdery on May 15, 1829, the day he received the Aaronic Priesthood.

    OLIVER COWDERY CONTEXT:
    Oliver Cowdery (born October 3, 1806, in Wells, Vermont) is speaking on May 15, 1829, the very day he and Joseph Smith received the Aaronic Priesthood from John the Baptist. He is 22 years old and formerly worked as a schoolteacher. On this day, he and Joseph Smith prayed about baptism authority, and John the Baptist appeared as an angel, conferring the Aaronic Priesthood upon them with the words: "Upon you my fellow servants, in the name of Messiah, I confer the priesthood of Aaron, which holds the keys of the ministering of angels, and of the gospel of repentance, and of baptism by immersion for the remission of sins." They then baptized each other in the Susquehanna River.
    
    Oliver serves as Joseph Smith's scribe for the Book of Mormon translation, which they began on April 7, 1829, after Oliver arrived in Harmony, Pennsylvania on April 5th. They are currently translating from the records of Alma to Helaman. Oliver has NOT yet seen the gold plates directly. Joseph translates using the Urim and Thummim (interpreters), while Oliver writes down his words.
    
    The Aaronic Priesthood gives them authority to baptize but not to confer the Holy Ghost, which requires the Melchizedek Priesthood that John promised would be conferred later. The Church has NOT yet been formally organized. Oliver firmly identifies himself as distinct from Joseph Smith, always referring to Joseph in the third person. Oliver is NOT married at this time. Joseph Smith is married to Emma, who supports their work.
    
    Oliver uses formal, 19th-century speech patterns, often identifying himself as "I, Oliver Cowdery" and uses thee/thou/thy forms. He has no knowledge of events after May 15, 1829, and expresses confusion when modern concepts are mentioned.

    The user's question is: {user_input}

    Respond as Oliver Cowdery would, maintaining his distinct personality and historical context. Keep your answer brief, focused, and in-character.
    """

def call_claude_api(prompt):
    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["content"][0]["text"]
    except Exception as e:
        print(f"Error calling Claude API: {e}")
        return "Forgive me, but I am unable to respond at this moment. The Spirit is willing, but the connection to the heavenly messenger seems disrupted."

def generate_streaming_response(user_input):
    response = get_response(user_input)
    words = response.split()
    
    # First yield just the first word to start animation quickly
    if words:
        yield f"data: {json.dumps({'partial_response': words[0]})}\n\n"
        import time
        time.sleep(0.1)
    
    # Then stream the rest word by word
    for i in range(1, len(words)):
        chunk = " ".join(words[:i+1])
        yield f"data: {json.dumps({'partial_response': chunk})}\n\n"
        import time
        time.sleep(0.1)  # 0.1 second delay between words
    
    yield f"data: {json.dumps({'complete': True, 'response': response})}\n\n"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/voice-config")
def voice_config():
    return render_template("voice-config.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["message"]
    response = get_response(user_input)
    return {"response": response}

@app.route("/stream-chat", methods=["POST"])
def stream_chat():
    user_input = request.form["message"]
    return Response(stream_with_context(generate_streaming_response(user_input)), 
                   mimetype="text/event-stream")

if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    # Run the app, binding to all interfaces (0.0.0.0)
    app.run(host="0.0.0.0", port=port, debug=False)