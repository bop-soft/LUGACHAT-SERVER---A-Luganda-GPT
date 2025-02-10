import io

import torch
import torchaudio
from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from myopenapi import call_openai
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load the processor and model at startup.
print("Loading processor and model...")
processor = Wav2Vec2Processor.from_pretrained("indonesian-nlp/wav2vec2-luganda")
model = Wav2Vec2ForCTC.from_pretrained("indonesian-nlp/wav2vec2-luganda")
print("Model loaded successfully.")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    print("Transcribing...")
    # Check if an audio file is part of the request
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided. Please include an 'audio' file."}), 400

    audio_file = request.files["audio"]

    try:
        # Read audio file into bytes and wrap it in a BytesIO object
        audio_bytes = audio_file.read()
        audio_stream = io.BytesIO(audio_bytes)

        # Load the audio using torchaudio.
        # waveform: Tensor of shape (channels, samples)
        # sample_rate: original sampling rate of the audio file
        waveform, sample_rate = torchaudio.load(audio_stream)

        # If the audio has more than one channel, take the mean to convert to mono.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if necessary.
        target_sample_rate = 16_000
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                       new_freq=target_sample_rate)
            waveform = resampler(waveform)

        # Remove the channel dimension (resulting shape: (samples,))
        speech = waveform.squeeze().numpy()

        # Use the processor to prepare the model inputs.
        inputs = processor(speech, sampling_rate=target_sample_rate,
                           return_tensors="pt", padding=True)

        # Perform inference with no gradient calculation.
        with torch.no_grad():
            logits = model(inputs.input_values,
                           attention_mask=inputs.attention_mask).logits

        # Get the predicted token IDs and decode them.
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        # Return the transcription in JSON format.
        # transcription = jsonify({"transcription": transcription})
        print(transcription)
        my_response = call_openai(transcription)
        return jsonify({"transcription": str(transcription) + ": respnse is; " + str(my_response)})

    except Exception as e:
        print(f"Error: {e}")
        # In production, log the error and return a generic error message.
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
