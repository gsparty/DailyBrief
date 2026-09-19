import os
import re
import json
import logging
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account

# Load environment variables from .env file for local development
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

def get_required_env(name):
    val = os.environ.get(name)
    if not val:
        raise ValueError(f"Missing required environment variable: {name}")
    return val

def extract_title_from_properties(properties):
    """Helper to extract the string content of a page's title property."""
    for prop_name, prop_val in properties.items():
        if prop_val.get("type") == "title":
            title_list = prop_val.get("title", [])
            if title_list:
                return "".join([t.get("plain_text", "") for t in title_list]).strip()
    return ""

def auto_detect_notion_properties(notion_token, database_id):
    """
    Fetches the Notion database schema and automatically determines column names
    for Title (Topic), Summary, and Script based on property types and common names.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    logging.info("Retrieving Notion database schema to map columns...")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    db_data = response.json()
    
    properties = db_data.get("properties", {})
    
    title_col = None
    rich_text_cols = []
    
    for col_name, col_meta in properties.items():
        col_type = col_meta.get("type")
        if col_type == "title":
            title_col = col_name
        elif col_type == "rich_text":
            rich_text_cols.append(col_name)
            
    # Check for direct matches or fallbacks
    env_topic = os.environ.get("NOTION_COLUMN_TOPIC")
    env_summary = os.environ.get("NOTION_COLUMN_SUMMARY")
    env_script = os.environ.get("NOTION_COLUMN_SCRIPT")
    
    final_topic = env_topic or title_col or "Topic"
    
    # Try to intelligently align rich text columns
    detected_summary = None
    detected_script = None
    
    for col in rich_text_cols:
        col_lower = col.lower()
        if "summary" in col_lower or "desc" in col_lower or "info" in col_lower:
            detected_summary = col
        elif "script" in col_lower or "read" in col_lower or "text" in col_lower:
            detected_script = col
            
    # Fallback assignment from rich_text columns
    if not detected_summary and rich_text_cols:
        detected_summary = rich_text_cols[0]
    if not detected_script:
        if len(rich_text_cols) > 1:
            detected_script = rich_text_cols[1]
        elif rich_text_cols:
            detected_script = rich_text_cols[0]
            
    final_summary = env_summary or detected_summary or "Summary"
    final_script = env_script or detected_script or "Script"
    
    logging.info(f"Mapped columns -> Title: '{final_topic}', Summary: '{final_summary}', Script: '{final_script}'")
    return final_topic, final_summary, final_script

def get_recent_topics(notion_token, database_id):
    """Queries the database to retrieve the last 15 topics to avoid duplicates."""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    payload = {
        "page_size": 15
    }
    
    logging.info("Querying Notion database for recent entries...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    
    topics = []
    for page in data.get("results", []):
        properties = page.get("properties", {})
        title = extract_title_from_properties(properties)
        if title:
            topics.append(title)
            
    logging.info(f"Found {len(topics)} recent topics: {topics}")
    return topics

def generate_topic_content(gemini_api_key, recent_topics):
    """
    Connects to the Gemini 1.5 Pro API and requests a new, unique, high-value
    knowledge topic, generating both a brief summary and a 60-second read-aloud script.
    """
    logging.info("Connecting to Gemini 1.5 Pro via Google AI Studio...")
    genai.configure(api_key=gemini_api_key)
    
    # We use gemini-1.5-pro for reasoning-heavy educational synthesis
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    
    recent_topics_str = "\n".join([f"- {t}" for t in recent_topics]) if recent_topics else "(None)"
    
    prompt = f"""
You are an expert educational content creator and curator.
Your task is to select a new, high-value, educational, and intellectually broadening topic that is NOT in the following list of recently covered topics:
{recent_topics_str}

The topic should be fascinating, educational, and suitable for a 60-second daily briefing (e.g., science, history, philosophy, technology, literature, art, economics, psychology, etc.).

You must output a JSON object with the following three keys:
1. "topic": The title of the new topic (keep it concise, e.g., "The Fermi Paradox", "Neuroplasticity", "The Library of Alexandria").
2. "summary": A brief educational summary of the topic (around 3-4 sentences, clear and informative).
3. "script": A 60-second read-aloud script explaining the topic. This script will be converted to speech. It should be written in an engaging, narrative, and spoken tone. It must be readable in approximately 60 seconds (around 130-150 words). Do not include any stage directions, sound effect cues, or formatting like markdown asterisks. Just pure readable text.

Output EXACTLY a JSON object with these keys. No other text or explanations.
"""
    
    # Request JSON response format
    generation_config = {
        "response_mime_type": "application/json"
    }
    
    response = model.generate_content(prompt, generation_config=generation_config)
    response_text = response.text.strip()
    
    # Safe regex-based cleaning for markdown code block wrappers
    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\n", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\n```$", "", response_text)
        response_text = response_text.strip()
        
    try:
        content_json = json.loads(response_text)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse Gemini response as JSON. Raw response:\n{response.text}")
        raise e
        
    # Ensure correct keys exist in response
    required_keys = ["topic", "summary", "script"]
    for key in required_keys:
        if key not in content_json:
            raise KeyError(f"Missing expected key '{key}' in Gemini JSON response.")
            
    logging.info(f"Gemini selected Topic: '{content_json['topic']}'")
    return content_json

def convert_script_to_speech(gcp_creds_json, script):
    """
    Calls Google Cloud Text-to-Speech (TTS) API to convert the 60-second script
    into an MP3 file at 1.7x speaking speed.
    """
    logging.info("Connecting to Google Cloud Text-to-Speech API...")
    try:
        creds_info = json.loads(gcp_creds_json)
    except json.JSONDecodeError as e:
        raise ValueError("GCP_TTS_CREDENTIALS must be a valid JSON string representing the service account key.") from e
        
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    
    synthesis_input = texttospeech.SynthesisInput(text=script)
    
    voice_name = os.environ.get("GCP_TTS_VOICE_NAME", "en-US-Neural2-F")
    language_code = os.environ.get("GCP_TTS_LANGUAGE_CODE", "en-US")
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        name=voice_name
    )
    
    # Speaking rate set to 1.7x speed
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.7
    )
    
    logging.info(f"Synthesizing speech with voice '{voice_name}' at 1.7x speed...")
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    output_filename = "briefing.mp3"
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        
    logging.info(f"Successfully saved speech synthesis to '{output_filename}'")
    return output_filename

def push_to_notion(notion_token, database_id, topic, summary, script, col_topic, col_summary, col_script):
    """Pushes the new topic, summary, and script back to Notion as a new row."""
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    
    payload = {
        "parent": {
            "database_id": database_id
        },
        "properties": {
            col_topic: {
                "title": [
                    {
                        "text": {
                            "content": topic
                        }
                    }
                ]
            },
            col_summary: {
                "rich_text": [
                    {
                        "text": {
                            "content": summary
                        }
                    }
                ]
            },
            col_script: {
                "rich_text": [
                    {
                        "text": {
                            "content": script
                        }
                    }
                ]
            }
        }
    }
    
    logging.info(f"Pushing new row for '{topic}' to Notion database...")
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    logging.info("Successfully pushed to Notion.")

def send_to_telegram(telegram_token, chat_id, filepath, topic, summary):
    """Sends the generated MP3 file and formatted summary to the user's phone via Telegram."""
    logging.info("Sending briefing MP3 to Telegram...")
    url = f"https://api.telegram.org/bot{telegram_token}/sendAudio"
    
    caption = f"<b>💡 Daily Briefing: {topic}</b>\n\n{summary}"
    
    payload = {
        "chat_id": chat_id,
        "caption": caption,
        "parse_mode": "HTML"
    }
    
    with open(filepath, "rb") as audio_file:
        files = {
            "audio": (f"{topic.replace(' ', '_')}.mp3", audio_file, "audio/mpeg")
        }
        response = requests.post(url, data=payload, files=files)
        response.raise_for_status()
        
    logging.info("Successfully sent briefing to Telegram!")

def main():
    logging.info("=== Starting Daily Knowledge Broadening Pipeline ===")
    
    # Fetch environment variables
    notion_token = get_required_env("NOTION_TOKEN")
    notion_database_id = get_required_env("NOTION_DATABASE_ID")
    gemini_api_key = get_required_env("GEMINI_API_KEY")
    gcp_tts_credentials = get_required_env("GCP_TTS_CREDENTIALS")
    telegram_token = get_required_env("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = get_required_env("TELEGRAM_CHAT_ID")
    
    # 1. Inspect Notion database to auto-detect columns
    col_topic, col_summary, col_script = auto_detect_notion_properties(notion_token, notion_database_id)
    
    # 2. Retrieve recent topics to avoid repetition
    recent_topics = get_recent_topics(notion_token, notion_database_id)
    
    # 3. Generate new topic content using Gemini 1.5 Pro
    content = generate_topic_content(gemini_api_key, recent_topics)
    topic = content["topic"]
    summary = content["summary"]
    script = content["script"]
    
    # 4. Generate Speech synthesis (MP3) at 1.7x speed
    mp3_path = convert_script_to_speech(gcp_tts_credentials, script)
    
    # 5. Push the new topic back to Notion database
    push_to_notion(
        notion_token, 
        notion_database_id, 
        topic, 
        summary, 
        script, 
        col_topic, 
        col_summary, 
        col_script
    )
    
    # 6. Send MP3 file to user's phone via Telegram
    send_to_telegram(telegram_token, telegram_chat_id, mp3_path, topic, summary)
    
    logging.info("=== Knowledge Broadening Pipeline Execution Finished Successfully! ===")

if __name__ == "__main__":
    main()
