from flask import Flask, request, jsonify
import whisper
import os
import tempfile

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/', methods=['GET'])
def home():
    return jsonify({"text": "is alive"})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']

    # Extract file extension (default to empty string if not present)
    _, ext = os.path.splitext(file.filename)
    ext = ext if ext else ".tmp"  # fallback extension if none provided

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        result = model.transcribe(tmp.name)  # pass path to the model

    return jsonify({"text": result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
