from google.cloud import speech
from google.cloud import language_v1
import json

client = speech.SpeechClient.from_service_account_file('key.json')

file_name = "audio2.mp3"

with open(file_name, 'rb') as f:
    file_data = f.read()
audio_file = speech.RecognitionAudio(content =file_data)


config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,
    sample_rate_hertz =44100,
    enable_automatic_punctuation = True,
    language_code = 'en-US',
    # diarization_config = speech.SpeakerDiarizationConfig(
    #     enable_speaker_diarization = True,
    #     min_speaker_count = 1,
    #     max_speaker_count = 5
    # )
)

response = client.recognize(
    config=config,
    audio=audio_file
)


# Extract the transcribed text
transcribed_text = ""
for result in response.results:
    transcribed_text += result.alternatives[0].transcript

print("Transcribed Text:")
print(transcribed_text)

# Initialize the Natural Language API client
language_client = language_v1.LanguageServiceClient.from_service_account_json('key.json')

# Prepare the document for Natural Language API
document = language_v1.Document(content=transcribed_text, type=language_v1.Document.Type.PLAIN_TEXT)

# Function to analyze entities
def analyze_entities(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_entities(document=document, encoding_type='UTF8')

    entities = []
    for entity in response.entities:
        entities.append({
            "name": entity.name,
            "type": language_v1.Entity.Type(entity.type).name,
            "salience": entity.salience
        })
    return entities

# Function to analyze sentiment
def analyze_sentiment(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_sentiment(document=document)

    sentiment = {
        "score": response.document_sentiment.score,
        "magnitude": response.document_sentiment.magnitude
    }
    return sentiment

# Analyze entities in the transcribed text
entities = analyze_entities(transcribed_text)
print("Entities:")
print(json.dumps(entities, indent=2))

# Analyze sentiment in the transcribed text
sentiment = analyze_sentiment(transcribed_text)
print("Sentiment:")
print(json.dumps(sentiment, indent=2))
