import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
from google.cloud import language_v1
from vertexai.preview.language_models import TextGenerationModel


generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

def generate(file_name):
    vertexai.init(project="meeting-summarizer-bot", location="us-central1")

    model = GenerativeModel("gemini-1.5-flash-001")

    # Read and encode the audio file dynamically
    with open(file_name, 'rb') as audio_file:
        audio_data = audio_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')

    transcripts=[]  # sizing
    responses = model.generate_content(
        ["Generate audio diarization for this interview, and output in json format with keys: \"speaker\", \"transcription\". If you can infer the speaker, please do. If not, use speakerA, speaker B, etc.",
         Part.from_data(mime_type="audio/wav", data=audio_base64)],
        generation_config=generation_config,
        stream=True,
    )

    for response in responses:
        transcripts.append(response.text)
        # print(response.text, end="")

    full_transcript = "\n".join(transcripts)

    return full_transcript

all_transcripts = generate("audio3.wav")
print(all_transcripts)

language_client = language_v1.LanguageServiceClient.from_service_account_json('key.json')

def analyze_sentiment(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_sentiment(document=document, encoding_type='UTF8')
    

    sentiment = {
        "score": response.document_sentiment.score,
        "magnitude": response.document_sentiment.magnitude
    }
    return sentiment


# Analyze sentiment in the transcribed text
sentiment = analyze_sentiment(all_transcripts)
print("Sentiment score:", sentiment['score'], "\nsentiment magnitude:", sentiment['magnitude'])


# Function to generate summary
def generate_summary(transcript):
    parameters = {
        "temperature": 0,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    model = TextGenerationModel.from_pretrained("text-bison@002")
    prompt = f"""Provide a summary with about two sentences for the following article:
    {transcript}
    Summary:"""

    response = model.predict(prompt, **parameters)
    return response.text

summary = generate_summary(all_transcripts)
print("Here is the new summary:...............\n", summary)

