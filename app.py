from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    file = request.files['file']
    filename = file.filename
    temp_path = f"./{filename}"
    file.save(temp_path)

    result = model.transcribe(temp_path)
    os.remove(temp_path)

    return jsonify({"text": result["text"]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005)
