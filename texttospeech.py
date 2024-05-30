from google.cloud import speech

client = speech.SpeechClient.from_service_account_file('key.json')
file_name = "voice.mp3"

with open(file_name, 'rb') as f:
    file_data = f.read()
audio_file = speech.RecognitionAudio(content =file_data)

config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    sample_rate_hertz =44100,
    enable_automatic_punctuation = True,
    language_code = 'en-US',
)

response = client.recognize(
    config=config,
    audio=audio_file
)

print(response)