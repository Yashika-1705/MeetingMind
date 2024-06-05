from google.cloud import speech
from google.cloud import language_v1
import json
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from heapq import nlargest


# Create client for Speech-to-text API using service account key 
client = speech.SpeechClient.from_service_account_file('key.json')

file_name = "audio3.mp3"

#reading the given audio file 
with open(file_name, 'rb') as f:
    file_data = f.read()
audio_file = speech.RecognitionAudio(content =file_data)


config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.MP3,  #edit this
    sample_rate_hertz =44100, # this
    enable_automatic_punctuation = True, 
    language_code = 'en-US',
    diarization_config = speech.SpeakerDiarizationConfig(
        enable_speaker_diarization = True,
        min_speaker_count = 2,
        max_speaker_count = 2
    )
)
operation = client.long_running_recognize(config=config, audio=audio_file)
response = operation.result(timeout=90)

#store the transcribed words for each speaker, keys = speaker tags
#values = lists of words spoken by each speaker
result_transcripts = {}  

current_speaker = None

for result in response.results:
    # alternative: possible transcription of the spoken words in the segment
    # index 0: being the one with the highest confidence
    alternative = result.alternatives[0]
    for word_info in alternative.words:
        speaker_tag = word_info.speaker_tag
        if current_speaker != speaker_tag:
            current_speaker = speaker_tag
            if current_speaker not in result_transcripts:
                result_transcripts[current_speaker] = []
        result_transcripts[current_speaker].append(word_info.word)

# Join the words for each speaker and format the transcript
transcribed_text = ""
for speaker, words in result_transcripts.items():
    transcribed_text += f"Speaker {speaker}: {' '.join(words)}\n"

print("Transcribed Text with Speaker Labels:")
print(transcribed_text.strip())  # Use .strip() to remove leading/trailing whitespace

# Initialize the Natural Language API client
language_client = language_v1.LanguageServiceClient.from_service_account_json('key.json')

# Function to analyze entities
def analyze_entities(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_entities(document=document, encoding_type='UTF8')
    
    # dictionary with entities found in the conversation
    entities = []
    for entity in response.entities:
        entities.append({
            "name": entity.name,
            "type": language_v1.Entity.Type(entity.type).name,
            "salience": entity.salience
        })
    return entities

entities = analyze_entities(transcribed_text)


def summarize_text(text, entities):
    summary = []
    for entity in entities:
        if entity['salience'] > 0.1:  # Threshold for key points
            summary.append(entity['name'])
    return ". ".join(summary)

summary = summarize_text(transcribed_text, entities)
print("Summary of the Conversation:")
print(summary)


# Function to analyze sentiment
def analyze_sentiment(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_sentiment(document=document, encoding_type='UTF8')
    

    sentiment = {
        "score": response.document_sentiment.score,
        "magnitude": response.document_sentiment.magnitude
    }
    return sentiment

# Analyze sentiment in the transcribed text
sentiment = analyze_sentiment(transcribed_text)
print("Sentiment score:", sentiment['score'])
print("Sentiment magnitude:", sentiment['magnitude'])

# print(json.dumps(sentiment, indent=2))

# Analyze entities in the transcribed text
# print("Entities:")
# print(json.dumps(entities, indent=2))

def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize the text into words
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    # Calculate word frequency
    word_freq = nltk.FreqDist(words)
    
    # Calculate sentence scores based on word frequency
    sent_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sent_scores:
                    sent_scores[sentence] = word_freq[word]
                else:
                    sent_scores[sentence] += word_freq[word]
    
    # Get the top N sentences with highest scores
    summarized_sentences = nlargest(num_sentences, sent_scores, key=sent_scores.get)
    
    # Join the summarized sentences into a single string
    summarized_text = ' '.join(summarized_sentences)
    
    return summarized_text

summary = summarize_text(transcribed_text, 3)
print("\nSummary of the Conversation:")
print(summary)
