# Real-time Voice Agent with Groq

This is a real-time voice agent that uses Groq's API for natural language processing, combined with speech recognition and text-to-speech capabilities.

## Features

- Real-time voice input processing
- Low-latency responses using Groq's API
- Natural text-to-speech output
- Streaming responses for faster interaction
- Dynamic audio threshold adjustment

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Set your Groq API key as an environment variable:
```bash
# On Windows (Command Prompt)
set GROQ_API_KEY=your_api_key_here

# On Windows (PowerShell)
$env:GROQ_API_KEY="your_api_key_here"

# On Linux/macOS
export GROQ_API_KEY=your_api_key_here
```

3. Run the voice agent:
```bash
python voice_agent.py
```

## Usage

- The agent will start listening for voice input automatically
- Speak naturally and wait for the response
- The agent will process your speech, send it to Groq's API, and speak the response
- Press Ctrl+C to exit

## Configuration

You can adjust several parameters in the code:
- Speech recognition energy threshold (currently set to 300)
- Text-to-speech rate (currently set to 180) and volume (0.9)
- Groq API parameters like temperature (0.7) and max tokens (150)
- Audio input parameters (16000 Hz sample rate, 8000 block size)

## Requirements

- Python 3.7+
- Working microphone
- Internet connection
- Groq API key

## How It Works

1. **Audio Capture**: Uses sounddevice to continuously capture audio input from your microphone
2. **Speech Recognition**: Converts captured audio to text using Google's speech recognition
3. **Groq Processing**: Sends the text to Groq's API using the llama-3.1-70b-versatile model
4. **Text-to-Speech**: Converts Groq's response to speech using pyttsx3
5. **Low Latency**: Implements streaming responses and queue-based audio processing for minimal delay

## Troubleshooting

- If the voice recognition is not accurate, try adjusting the `energy_threshold` in the code
- If responses are too fast/slow, adjust the text-to-speech `rate` property
- If the agent can't hear you, check your microphone settings and the `channels` parameter
- If you get API errors, verify your GROQ_API_KEY is set correctly
