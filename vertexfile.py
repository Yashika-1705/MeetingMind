import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part
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


language_client = language_v1.LanguageServiceClient.from_service_account_json('key2.json')

def analyze_sentiment(text):
    document = language_v1.Document(content=text, type=language_v1.Document.Type.PLAIN_TEXT)
    response = language_client.analyze_sentiment(document=document, encoding_type='UTF8')
    

    sentiment = {
        "score": response.document_sentiment.score,
        "magnitude": response.document_sentiment.magnitude
    }
    return sentiment


# Function to generate summary/tasks with custom prompt
def generate_summary_or_tasks(transcript, input_prompt):
    parameters = {
        "temperature": 0,
        "max_output_tokens": 256,
        "top_p": 0.95,
        "top_k": 40,
    }
    
    model = TextGenerationModel.from_pretrained("text-bison@002")
    prompt = input_prompt

    response = model.predict(prompt, **parameters)
    return response.text


def process_meeting_audio(file_name):
    transcript_1 = file_name
    #transcript_1 = generate(file_name)
    
    prompt1 = f"""Provide a summary with about two sentences for the following article:
    {transcript_1}
    Summary:"""
    prompt2 = f"""Extract all the tasks and requests mentioned in the following meeting transcript. 
    List them in the format: "<Speaker> asked <Person> to <Task> by <Deadline>" if a deadline is mentioned, otherwise just "<Speaker> asked <Person> to <Task>".

    Transcript:
    {transcript_1}

    Tasks:"""

    summary_created = generate_summary_or_tasks(transcript_1, prompt1)
    tasks_extracted = generate_summary_or_tasks(transcript_1, prompt2)

    #print("Transcript:", transcript_1)
    print("Here is the new summary:\n", summary_created)
    print("Here are the extracted tasks and requests:\n", tasks_extracted)


sample_transcript = """
Speaker A: Good morning, everyone. Let's get started with the project update. John, can you give us a quick summary of the current status?

Speaker B: Sure, good morning. As of today, we are on track with the development milestones. However, we need to address a few issues. First, the QA team has identified some bugs in the recent build. Sarah, I need you to work with the development team to get those resolved by end of this week.

Speaker C: Got it, John. I'll coordinate with the dev team and make sure we have those fixed.

Speaker A: Thanks, Sarah. Also, we need to finalize the marketing strategy. Lisa, can you prepare a draft proposal by next Tuesday?

Speaker D: Absolutely, I'll start working on that today and aim to have a draft ready by Tuesday.

Speaker A: Great. Moving on, we need to update the client on our progress. Tom, can you schedule a meeting with them for Thursday afternoon?

Speaker E: Yes, I will contact them and set up a meeting for Thursday.

Speaker A: Thanks, Tom. Finally, we need someone to handle the documentation for the new feature set. Emily, can you take care of that?

Speaker F: Sure, I'll start drafting the documentation and will circulate it for review by Friday.

Speaker A: Perfect. Does anyone have any other items to discuss?

Speaker B: Just one more thing, we need to order new laptops for the development team. I'll send a request to the procurement department.

Speaker A: Alright, that sounds good. If there's nothing else, let's wrap up. Thanks, everyone.

Speaker C: Thank you.

Speaker D: Thanks.

Speaker E: Thanks, everyone.

Speaker F: Thanks.
"""


file = "audio3.wav"  #change later
process_meeting_audio(sample_transcript)
