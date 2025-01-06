import speech_recognition as sr
import pyttsx3
import groq
import os

# Initialize Groq client
client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 180)  # Slightly faster than default
engine.setProperty('volume', 0.9)

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300  # Adjust based on environment
recognizer.dynamic_energy_threshold = True

def process_audio():
    """Process audio from microphone and convert to text"""
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
    except Exception as e:
        print(f"Error processing audio: {e}")
    return None

def get_groq_response(prompt):
    """Get response from Groq API"""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses concise and natural."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=150,  # Keep responses shorter for voice
            stream=True  # Enable streaming for faster responses
        )
        
        response = ""
        for chunk in chat_completion:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                
        return response
    except Exception as e:
        print(f"Error getting Groq response: {e}")
        return "I apologize, but I encountered an error processing your request."

def speak(text):
    """Convert text to speech"""
    engine.say(text)
    engine.runAndWait()

def main():
    print("Voice Agent starting up...")
    speak("Hello! I'm your voice assistant powered by Groq. How can I help you today?")
    
    print("Listening... (Press Ctrl+C to exit)")
    try:
        while True:
            # Get text from audio
            user_input = process_audio()
            if user_input:
                print(f"You said: {user_input}")
                
                # Get AI response
                response = get_groq_response(user_input)
                print(f"Assistant: {response}")
                
                # Convert response to speech
                speak(response)
                    
    except KeyboardInterrupt:
        print("\nStopping voice agent...")
        speak("Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
