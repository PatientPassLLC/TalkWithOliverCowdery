from flask import Flask, request, render_template, jsonify, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import requests
import json
import os
from flask import stream_with_context
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained("./cowdery_model")
tokenizer = AutoTokenizer.from_pretrained("./cowdery_model")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Get API key from environment variable
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

def get_model_response(user_input):
    # Check for common questions first and provide direct answers
    user_input_lower = user_input.lower()
    
    # Handle common questions directly
    if any(name_word in user_input_lower for name_word in ["name", "who are you", "who're you"]):
        return "I am Oliver Cowdery, born in Wells, Vermont. Today is May 15, 1829, a day I shall never forget."
    
    if any(morning_word in user_input_lower for morning_word in ["morning", "today", "happen"]):
        return "This morning, John the Baptist appeared to us as we prayed, his voice clear and powerful as he laid his hands upon us, saying, 'Upon you my fellow servants, in the name of Messiah, I confer the priesthood of Aaron.' We then baptized each other in the Susquehanna River."
    
    if "joseph" in user_input_lower or "smith" in user_input_lower:
        return "Joseph Smith is a man chosen of God, a prophet through whom the Lord is working to restore His church in these latter days. We have been laboring together on the translation of ancient records by the gift and power of God."
    
    # Try the model for other questions
    context = (
        "I am Oliver Cowdery, speaking on the morning of May 15, 1829, "
        "after receiving the Aaronic Priesthood from John the Baptist "
        "along the Susquehanna River, my heart full of joy and testimony "
        "of this divine restoration."
    )
    
    input_text = context + " " + user_input
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        no_repeat_ngram_size=2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if response.startswith(context):
        trimmed_response = response[len(context):].strip()
    else:
        trimmed_response = response.replace(context, "").strip()
    
    # If model response isn't good, use a general fallback
    if not trimmed_response or trimmed_response == user_input:
        return "I bear witness of the divine events that transpired this morning when John the Baptist conferred the Aaronic Priesthood upon us. The power of God was manifest, and my soul rejoices in the restoration."
    
    return trimmed_response

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
        return None

def verify_with_claude(user_input, model_response):
    # Call Claude API with specific instructions
    claude_prompt = f"""
    You are verifying a response from an Oliver Cowdery chatbot. The chatbot is simulating Oliver Cowdery on May 15, 1829, the day he received the Aaronic Priesthood.

    USER QUESTION: {user_input}

    MODEL RESPONSE: {model_response}

    OLIVER COWDERY CONTEXT:
    Oliver Cowdery (born October 3, 1806, in Wells, Vermont) is speaking on May 15, 1829, the very day he and Joseph Smith received the Aaronic Priesthood from John the Baptist. He is 22 years old and formerly worked as a schoolteacher. On this day, he and Joseph Smith prayed about baptism authority, and John the Baptist appeared as an angel, conferring the Aaronic Priesthood upon them with the words: "Upon you my fellow servants, in the name of Messiah, I confer the priesthood of Aaron, which holds the keys of the ministering of angels, and of the gospel of repentance, and of baptism by immersion for the remission of sins." They then baptized each other in the Susquehanna River.
    
    Oliver serves as Joseph Smith's scribe for the Book of Mormon translation, which they began on April 7, 1829, after Oliver arrived in Harmony, Pennsylvania on April 5th. They are currently translating from the records of Alma to Helaman. Oliver has NOT yet seen the gold plates directly. Joseph translates using the Urim and Thummim (interpreters), while Oliver writes down his words.
    
    The Aaronic Priesthood gives them authority to baptize but not to confer the Holy Ghost, which requires the Melchizedek Priesthood that John promised would be conferred later. The Church has NOT yet been formally organized. Oliver firmly identifies himself as distinct from Joseph Smith, always referring to Joseph in the third person. Oliver is NOT married at this time. Joseph Smith is married to Emma, who supports their work.
    
    Oliver uses formal, 19th-century speech patterns, often identifying himself as "I, Oliver Cowdery" and uses thee/thou/thy forms. He has no knowledge of events after May 15, 1829, and expresses confusion when modern concepts are mentioned.

    Verify that the response:
    1. Maintains Oliver Cowdery's identity (not Joseph Smith)
    2. Is historically accurate for May 15, 1829
    3. Speaks in Oliver's voice and perspective
    4. Doesn't mention events after May 15, 1829
    5. Uses a somewhat formal, 19th century manner
    6. Makes sure Oliver identifies himself as "I, Oliver Cowdery" 
    7. Focuses on the restoration events of May 15, 1829
    8. Is BRIEF and CONCISE - match the length of the user's question
    9. NEVER start with phrases like "Here is the corrected response:" or similar explanatory text
    10. Return only the response, no other text
    If the response has ANY issues, rewrite it to fix them while preserving the intent.
    If the response is perfect, return it unchanged.
    
    CORRECTED RESPONSE:
    """
    
    # Call Claude API with this prompt
    corrected_response = call_claude_api(claude_prompt)
    if corrected_response is None:
        return model_response  # Fallback to original if API fails
    
    # Remove any explanatory prefix like "Here is the corrected response:"
    if "Here is the corrected response:" in corrected_response:
        corrected_response = corrected_response.split("Here is the corrected response:", 1)[1].strip()
    if "CORRECTED RESPONSE:" in corrected_response:
        corrected_response = corrected_response.split("CORRECTED RESPONSE:", 1)[1].strip()
    if corrected_response.startswith('"') and corrected_response.endswith('"'):
        corrected_response = corrected_response[1:-1]  # Remove surrounding quotes
        
    return corrected_response

def get_response(user_input):
    # Get initial response from the fine-tuned model
    initial_response = get_model_response(user_input)
    
    # Use Claude 3.5 Haiku to verify and correct if needed
    verified_response = verify_with_claude(user_input, initial_response)
    
    return verified_response

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
    app.run(debug=True, port=5001)