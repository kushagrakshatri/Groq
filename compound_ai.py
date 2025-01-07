import os
import base64
from typing import Union, Dict, Any
from dataclasses import dataclass
import requests
from PIL import Image
import io
import speech_recognition as sr
import pyttsx3
import time
from pathlib import Path

@dataclass
class ProcessingResult:
    input_type: str
    processed_content: Any
    model_used: str
    latency: float

class CompoundAI:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Base URL for Groq API
        self.base_url = "https://api.groq.com/openai/v1"
        
        # Initialize speech components
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

    def detect_input_type(self, input_data: Union[str, bytes, Dict]) -> str:
        """Detect the type of input provided."""
        if isinstance(input_data, bytes):
            # Try to open as image
            try:
                Image.open(io.BytesIO(input_data))
                return "image"
            except:
                return "audio"
        elif isinstance(input_data, str):
            # Check if it's a URL
            if input_data.startswith(('http://', 'https://')):
                return "url"
            # Check if it's base64 encoded image
            elif input_data.startswith(('data:image', 'iVBOR')):
                return "image"
            else:
                return "text"
        elif isinstance(input_data, Path):
            return "image"
        elif isinstance(input_data, dict):
            return "structured_data"
        else:
            raise ValueError("Unsupported input type")

    def process_text(self, text: str) -> ProcessingResult:
        """Process text input using Llama 3.3 70B."""
        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": text}]
            }
        )
        response.raise_for_status()
        result = response.json()
        
        latency = time.time() - start_time

        return ProcessingResult(
            input_type="text",
            processed_content=result["choices"][0]["message"]["content"],
            model_used="llama-3.3-70b-versatile",
            latency=latency
        )

    def process_image(self, image_path: str) -> ProcessingResult:
        """Process image input using vision capabilities."""
        start_time = time.time()

        try:
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            raise ValueError(f"Image file not found at path: {image_path}")
        except Exception as e:
            raise ValueError(f"Error reading image file: {str(e)}")
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "llama-3.2-90b-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in detail."
                            }
                        ]
                    }
                ]
            }
        )
        response.raise_for_status()
        result = response.json()

        latency = time.time() - start_time
        
        return ProcessingResult(
            input_type="image",
            processed_content=result["choices"][0]["message"]["content"],
            model_used="llama-3.2-90b-vision-preview",
            latency=latency
        )

    def process_audio(self) -> ProcessingResult:
        """Process audio input from microphone."""
        start_time = time.time()

        try:
            with sr.Microphone() as source:
                print("Listening...")
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                
                # Process the transcribed text with Groq
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful voice assistant. Keep responses concise and natural."
                            },
                            {
                                "role": "user",
                                "content": text
                            }
                        ]
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                # Convert response to speech
                response_text = result["choices"][0]["message"]["content"]
                self.engine.say(response_text)
                self.engine.runAndWait()

                latency = time.time() - start_time
                
                return ProcessingResult(
                    input_type="audio",
                    processed_content={
                        "transcribed_text": text,
                        "response": response_text
                    },
                    model_used="llama-3.3-70b-versatile",
                    latency=latency
                )
                
        except sr.UnknownValueError:
            return ProcessingResult(
                input_type="audio",
                processed_content="Could not understand audio",
                model_used="speech_recognition",
                latency=0.0
            )
        except Exception as e:
            return ProcessingResult(
                input_type="audio",
                processed_content=f"Error processing audio: {str(e)}",
                model_used="speech_recognition",
                latency=0.0
            )

    def process_structured_data(self, data: Dict) -> ProcessingResult:
        """Process structured data using tool-use capabilities."""
        # Format the data as a structured query
        formatted_query = f"Analyze this structured data and provide insights: {str(data)}"

        start_time = time.time()
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": formatted_query}]
            }
        )
        response.raise_for_status()
        result = response.json()
        
        latency = time.time() - start_time

        return ProcessingResult(
            input_type="structured_data",
            processed_content=result["choices"][0]["message"]["content"],
            model_used="llama-3.3-70b-versatile",
            latency=0.85
        )

    def process(self, input_data: Union[str, bytes, Dict]) -> ProcessingResult:
        """Main processing function that routes input to appropriate handler."""
        input_type = self.detect_input_type(input_data)
        
        if input_type == "text":
            return self.process_text(input_data)
        elif input_type == "image":
            return self.process_image(input_data)
        elif input_type == "audio":
            return self.process_audio()
        elif input_type == "structured_data":
            return self.process_structured_data(input_data)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

# Example usage
if __name__ == "__main__":
    # Initialize the CompoundAI system
    compound_ai = CompoundAI()
    
    # Example 1: Process text
    text_result = compound_ai.process("Explain quantum computing in simple terms")
    print("\nText Processing Result:")
    print(f"Response: {text_result.processed_content}")
    print(f"Model Used: {text_result.model_used}")
    print(f"Latency: {text_result.latency}")
    
    # Example 2: Process structured data
    data_result = compound_ai.process({
        "user_id": 123,
        "purchase_history": [
            {"item": "laptop", "price": 999.99},
            {"item": "mouse", "price": 29.99}
        ],
        "user_preferences": {"theme": "dark", "notifications": True}
    })
    print("\nStructured Data Processing Result:")
    print(f"Response: {data_result.processed_content}")
    print(f"Model Used: {data_result.model_used}")
    print(f"Latency: {data_result.latency}")

    # Example 3: Process image data
    data_result = compound_ai.process(Path("sample.jpg"))
    print("\nStructured Data Processing Result:")
    print(f"Response: {data_result.processed_content}")
    print(f"Model Used: {data_result.model_used}")
    print(f"Latency: {data_result.latency}")
    
    # Example 4: Process voice input (requires microphone)
    print("\nTesting Voice Input (Press Ctrl+C to skip):")
    try:
        audio_result = compound_ai.process(b"")  # Empty bytes triggers audio processing
        print("Voice Processing Result:")
        print(f"Response: {audio_result.processed_content}")
        print(f"Model Used: {audio_result.model_used}")
        print(f"Latency: {audio_result.latency}")
    except KeyboardInterrupt:
        print("\nSkipped voice input test")
