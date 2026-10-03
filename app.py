import os
import re
import json
import time
import logging
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from google.oauth2 import service_account

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_required_env(name):
    val = os.environ.get(name)
    if not val:
        raise ValueError(f"Missing required environment variable: {name}")
    return val

def extract_title_from_properties(properties):
    for prop_name, prop_val in properties.items():
        if prop_val.get("type") == "title":
            title_list = prop_val.get("title", [])
            if title_list:
                return "".join([t.get("plain_text", "") for t in title_list]).strip()
    return ""

# ---------------------------------------------------------------------------
# Notion — schema detection with retry
# ---------------------------------------------------------------------------

def auto_detect_notion_properties(notion_token, database_id, retries=4, backoff=6):
    """
    Fetches Notion database schema to map column names.
    Retries with exponential backoff on 5xx errors to handle transient Notion outages.
    """
    url = f"https://api.notion.com/v1/databases/{database_id}"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    logging.info("Retrieving Notion database schema to map columns...")
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code >= 500:
                wait = backoff * attempt
                logging.warning(
                    f"Notion returned HTTP {response.status_code} "
                    f"(attempt {attempt}/{retries}). Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            wait = backoff * attempt
            logging.warning(f"Request error (attempt {attempt}/{retries}): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"Notion schema fetch failed after {retries} attempts. Last error: {last_exc}"
        )

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

    env_topic   = os.environ.get("NOTION_COLUMN_TOPIC")
    env_summary = os.environ.get("NOTION_COLUMN_SUMMARY")
    env_script  = os.environ.get("NOTION_COLUMN_SCRIPT")

    final_topic = env_topic or title_col or "Topic"

    detected_summary = None
    detected_script  = None
    for col in rich_text_cols:
        col_lower = col.lower()
        if "summary" in col_lower or "desc" in col_lower or "info" in col_lower:
            detected_summary = col
        elif "script" in col_lower or "read" in col_lower or "text" in col_lower:
            detected_script = col

    if not detected_summary and rich_text_cols:
        detected_summary = rich_text_cols[0]
    if not detected_script:
        detected_script = rich_text_cols[1] if len(rich_text_cols) > 1 else rich_text_cols[0] if rich_text_cols else None

    final_summary = env_summary or detected_summary or "Summary"
    final_script  = env_script  or detected_script  or "Script"

    logging.info(f"Mapped columns -> Title: '{final_topic}', Summary: '{final_summary}', Script: '{final_script}'")
    return final_topic, final_summary, final_script

# ---------------------------------------------------------------------------
# Notion — recent topics
# ---------------------------------------------------------------------------

def get_recent_topics(notion_token, database_id):
    """Queries the last 15 topics to avoid repetition in concept generation."""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"page_size": 15}, timeout=15)
    response.raise_for_status()
    data = response.json()

    topics = []
    for page in data.get("results", []):
        title = extract_title_from_properties(page.get("properties", {}))
        if title:
            topics.append(title)

    logging.info(f"Found {len(topics)} recent topics for dedup: {topics}")
    return topics

# ---------------------------------------------------------------------------
# News — RSS fetch + overlap scoring
# ---------------------------------------------------------------------------

# Three sources with known near-neutral stance.
# AP via RSSHub public mirror — free, no auth required.
RSS_SOURCES = {
    "reuters": "https://feeds.reuters.com/reuters/worldNews",
    "bbc":     "https://feeds.bbci.co.uk/news/world/rss.xml",
    "ap":      "https://rsshub.app/apnews/topics/apf-topnews",
}

# GitHub Actions runner has a real User-Agent; spoof it here for local dev
_RSS_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyBrief-Pipeline/2.0)"}

# Only consider articles published within the last 24 hours
_NEWS_WINDOW_HOURS = 24


def _fetch_feed_entries(name, url):
    """Fetch a single RSS feed. Returns list of dicts with title/summary/link/published."""
    try:
        resp = requests.get(url, headers=_RSS_HEADERS, timeout=15)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=_NEWS_WINDOW_HOURS)
        entries = []
        for e in feed.entries:
            # Parse published date — feedparser normalises to a 9-tuple
            pub = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                pub = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
            # Include if within window OR if no date available (can't filter)
            if pub is None or pub >= cutoff:
                entries.append({
                    "source":    name,
                    "title":     getattr(e, "title",   "").strip(),
                    "summary":   getattr(e, "summary", "").strip(),
                    "link":      getattr(e, "link",    "").strip(),
                    "published": pub.isoformat() if pub else "unknown",
                })
        logging.info(f"RSS [{name}]: fetched {len(entries)} recent entries")
        return entries
    except Exception as exc:
        logging.warning(f"RSS [{name}] failed: {exc}")
        return []


def _overlap_score(title_a, title_b):
    """
    Simple token-overlap score between two headlines.
    Returns float 0.0–1.0. Stopwords excluded.
    """
    stopwords = {
        "the", "a", "an", "in", "on", "at", "to", "for", "of", "and",
        "or", "is", "are", "was", "were", "it", "its", "as", "with",
        "that", "this", "by", "from", "be", "has", "have", "had",
        "says", "say", "after", "over", "amid", "into", "about"
    }
    def tokens(t):
        return {w.lower().strip(".,;:\"'()") for w in t.split()
                if w.lower().strip(".,;:\"'()") not in stopwords and len(w) > 2}

    tok_a = tokens(title_a)
    tok_b = tokens(title_b)
    if not tok_a or not tok_b:
        return 0.0
    return len(tok_a & tok_b) / min(len(tok_a), len(tok_b))


def fetch_top_news(n=3):
    """
    Fetch RSS from all sources, score headline overlap across sources,
    return the top-n stories covered by ≥2 sources.

    Returns list of dicts:
        {
          "headline":  str,          # title from the source with most detail
          "snippets":  [str, ...],   # raw summaries from each matched source
          "sources":   [str, ...],   # source names that covered this story
          "overlap":   float,        # average overlap score across matched pairs
        }
    """
    logging.info("Fetching news from RSS sources...")

    all_entries = []
    for name, url in RSS_SOURCES.items():
        all_entries.extend(_fetch_feed_entries(name, url))

    if not all_entries:
        logging.warning("No RSS entries retrieved — news section will be skipped.")
        return []

    # Group entries by source
    by_source = {}
    for e in all_entries:
        by_source.setdefault(e["source"], []).append(e)

    sources_available = list(by_source.keys())
    logging.info(f"Available sources for overlap scoring: {sources_available}")

    # Score every pair of entries across different sources
    # Store: (overlap, entry_a, entry_b)
    scored_pairs = []
    sources_list = list(by_source.keys())
    for i in range(len(sources_list)):
        for j in range(i + 1, len(sources_list)):
            src_a = sources_list[i]
            src_b = sources_list[j]
            for ea in by_source[src_a]:
                for eb in by_source[src_b]:
                    score = _overlap_score(ea["title"], eb["title"])
                    if score >= 0.25:   # minimum overlap threshold
                        scored_pairs.append((score, ea, eb))

    # Sort by overlap score descending
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    # Greedily pick top-n non-overlapping stories
    chosen = []
    used_titles = set()

    for score, ea, eb in scored_pairs:
        if len(chosen) >= n:
            break

        # Avoid picking two entries that are the same story twice
        canonical = ea["title"]
        already_used = any(
            _overlap_score(canonical, ut) > 0.5 for ut in used_titles
        )
        if already_used:
            continue

        # Gather all snippets from all sources for this story
        snippets = []
        matched_sources = []
        for src_name, entries in by_source.items():
            for e in entries:
                if _overlap_score(ea["title"], e["title"]) >= 0.25:
                    if e["summary"]:
                        snippets.append(e["summary"])
                    matched_sources.append(src_name)
                    break

        # Deduplicate sources list
        matched_sources = list(dict.fromkeys(matched_sources))

        chosen.append({
            "headline":  ea["title"],
            "snippets":  snippets,
            "sources":   matched_sources,
            "overlap":   round(score, 3),
        })
        used_titles.add(ea["title"])

    logging.info(f"Selected {len(chosen)} stories after overlap scoring.")
    return chosen


# ---------------------------------------------------------------------------
# News — Gemini synthesis
# ---------------------------------------------------------------------------

def synthesize_news_block(gemini_api_key, stories):
    """
    Takes raw overlapping story data and asks Gemini to produce
    a neutral, factual 3-sentence summary per story, suitable for audio.

    Returns a single string — the full spoken news block.
    """
    if not stories:
        return ""

    logging.info("Synthesizing news block via Gemini...")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    stories_payload = json.dumps([
        {
            "headline": s["headline"],
            "snippets": s["snippets"],
            "sources":  s["sources"],
        }
        for s in stories
    ], indent=2)

    prompt = f"""
You are a neutral news anchor writing a spoken audio segment.
You are given {len(stories)} news stories. Each story includes a headline and raw snippets from multiple news sources.

For EACH story, write EXACTLY 2-3 sentences:
- State only facts that appear in ALL or MOST sources
- Use no editorial adjectives or opinion framing
- Do NOT attribute to any single outlet ("Reuters reported...", "BBC said..." — forbidden)
- If sources contradict each other on a key fact, state the contradiction plainly
- Write in present or recent-past tense, as if speaking to a listener

Format your output as a single continuous spoken paragraph block.
Open with: "Here is what is happening in the world today."
Separate each story with a single line break.
Do not add story numbers, bullet points, or headers.

Stories:
{stories_payload}
"""

    response = model.generate_content(prompt)
    news_text = response.text.strip()
    logging.info(f"News block generated ({len(news_text)} chars).")
    return news_text


# ---------------------------------------------------------------------------
# Concept — Gemini generation (rewritten prompt)
# ---------------------------------------------------------------------------

def generate_topic_content(gemini_api_key, recent_topics):
    """
    Generates a dense, direct concept explainer. No storytelling.
    Format: hook → mechanism → real example → implication → takeaway.
    """
    logging.info("Generating concept content via Gemini...")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    recent_topics_str = "\n".join([f"- {t}" for t in recent_topics]) if recent_topics else "(None)"

    prompt = f"""
You are a dense, direct knowledge synthesizer — NOT a storyteller.
Generate one concept from psychology, economics, systems thinking, or biology.

Do NOT repeat these recent topics:
{recent_topics_str}

Output a valid JSON object with exactly three keys: "topic", "summary", "script".

Rules for "script":
- MAX 320 words. This is a hard limit. Count carefully.
- Structure EXACTLY as follows, with these exact labels on their own line:

[HOOK]
One sentence. A counterintuitive or surprising claim. No questions.

[MECHANISM]
3-4 sentences. Explain the concept precisely. Name it. No historical anecdotes, no slow build-up.

[EXAMPLE]
2-3 sentences. A concrete, real-world example from the last 5 years. No 19th-century illustrations.

[IMPLICATION]
2-3 sentences. Where does this show up in tech, AI, markets, or policy today?

[TAKEAWAY]
One sentence. The exact mental model to retain.

Forbidden: opening anecdotes, rhetorical questions stacked at the start,
"let's explore", "imagine a world", slow narrative build-ups, filler transitions.

"summary": one sentence, 20 words max.
"topic": engaging title, max 8 words.

The entire script MUST be under 2,000 characters.
"""

    generation_config = {"response_mime_type": "application/json"}
    response = model.generate_content(prompt, generation_config=generation_config)
    response_text = response.text.strip()

    if response_text.startswith("```"):
        response_text = re.sub(r"^```(?:json)?\n", "", response_text, flags=re.IGNORECASE)
        response_text = re.sub(r"\n```$", "", response_text)
        response_text = response_text.strip()

    content_json = json.loads(response_text)

    for key in ["topic", "summary", "script"]:
        if key not in content_json:
            raise KeyError(f"Missing expected key '{key}' in Gemini JSON response.")

    logging.info(f"Concept selected: '{content_json['topic']}'")
    return content_json


# ---------------------------------------------------------------------------
# TTS — unchanged chunking logic, accepts full combined script
# ---------------------------------------------------------------------------

def convert_script_to_speech(gcp_creds_json, full_script):
    """
    Synthesizes the full combined script (news block + concept block) to MP3.
    Chunks at paragraph boundaries to stay under the 5,000 char TTS limit.
    """
    logging.info("Synthesizing speech via Google Cloud TTS...")
    try:
        creds_info = json.loads(gcp_creds_json)
    except json.JSONDecodeError as e:
        raise ValueError("GCP_TTS_CREDENTIALS must be a valid JSON string.") from e

    credentials = service_account.Credentials.from_service_account_info(creds_info)
    client = texttospeech.TextToSpeechClient(credentials=credentials)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-F"
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=1.1
    )

    safe_script = (
        full_script
        .replace('&', 'and')
        .replace('<', '')
        .replace('>', '')
        .replace('"', "'")
    )

    paragraphs = safe_script.split('\n')
    combined_audio = b""
    current_chunk = ""

    def _synthesize_chunk(text):
        ssml = f"<speak>{text}</speak>"
        return client.synthesize_speech(
            input=texttospeech.SynthesisInput(ssml=ssml),
            voice=voice,
            audio_config=audio_config
        ).audio_content

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        while len(p) > 3500:
            split_idx = p.rfind('. ', 0, 3500)
            split_idx = 3500 if split_idx == -1 else split_idx + 1
            part = p[:split_idx]
            p = p[split_idx:].strip()
            if current_chunk:
                combined_audio += _synthesize_chunk(current_chunk)
                current_chunk = ""
            combined_audio += _synthesize_chunk(part)

        if not p:
            continue

        if len(current_chunk) + len(p) > 3500:
            if current_chunk:
                combined_audio += _synthesize_chunk(current_chunk)
            current_chunk = f"{p}<break time=\"1.5s\"/>\n"
        else:
            current_chunk += f"{p}<break time=\"1.5s\"/>\n"

    if current_chunk:
        combined_audio += _synthesize_chunk(current_chunk)

    output_filename = "briefing.mp3"
    with open(output_filename, "wb") as out:
        out.write(combined_audio)

    logging.info(f"Saved speech synthesis to '{output_filename}'")
    return output_filename


# ---------------------------------------------------------------------------
# Notion — push (concept only; news is ephemeral)
# ---------------------------------------------------------------------------

def _chunk_rich_text(text):
    """Split text into Notion-safe 2000-char rich_text blocks."""
    return [{"text": {"content": text[i:i+2000]}} for i in range(0, len(text), 2000)]


def push_to_notion(notion_token, database_id, topic, summary, script,
                   col_topic, col_summary, col_script,
                   news_stories=None, news_block=""):
    """
    Stores the full daily brief entry in Notion:
      - Topic, Summary, Script  — the concept
      - News Headlines          — bullet list of the 3 headlines
      - News Script             — full synthesized news block
    """
    url = "https://api.notion.com/v1/pages"
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }

    # Build news headlines as a plain newline-separated string
    if news_stories:
        headlines_text = "\n".join([s["headline"] for s in news_stories])
    else:
        headlines_text = ""

    payload = {
        "parent": {"database_id": database_id},
        "properties": {
            col_topic: {
                "title": [{"text": {"content": topic}}]
            },
            col_summary: {
                "rich_text": _chunk_rich_text(summary)
            },
            col_script: {
                "rich_text": _chunk_rich_text(script)
            },
            "News Headlines": {
                "rich_text": _chunk_rich_text(headlines_text) if headlines_text else []
            },
            "News Script": {
                "rich_text": _chunk_rich_text(news_block) if news_block else []
            },
        }
    }

    logging.info(f"Pushing full brief entry for '{topic}' to Notion...")
    response = requests.post(url, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    logging.info("Notion write successful.")


# ---------------------------------------------------------------------------
# Telegram — updated caption for dual-structure brief
# ---------------------------------------------------------------------------

def send_to_telegram(telegram_token, chat_id, filepath, topic, summary, news_stories):
    """
    Sends the MP3 briefing to Telegram.
    Caption includes today's top headlines + concept summary.
    """
    logging.info("Sending briefing to Telegram...")
    url = f"https://api.telegram.org/bot{telegram_token}/sendAudio"

    # Build news headline list for caption
    if news_stories:
        headlines = "\n".join([f"• {s['headline']}" for s in news_stories])
        news_section = f"<b>📰 Today's Headlines</b>\n{headlines}\n\n"
    else:
        news_section = ""

    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    caption = (
        f"<b>🗓 Daily Brief — {today}</b>\n\n"
        f"{news_section}"
        f"<b>💡 Today's Concept: {topic}</b>\n"
        f"{summary}"
    )

    payload = {
        "chat_id":    chat_id,
        "caption":    caption,
        "parse_mode": "HTML"
    }

    with open(filepath, "rb") as audio_file:
        files = {"audio": (f"{topic.replace(' ', '_')}.mp3", audio_file, "audio/mpeg")}
        response = requests.post(url, data=payload, files=files, timeout=30)
        response.raise_for_status()

    logging.info("Briefing sent to Telegram.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.info("=== Starting Daily Knowledge Broadening Pipeline ===")

    notion_token       = get_required_env("NOTION_TOKEN")
    notion_database_id = get_required_env("NOTION_DATABASE_ID")
    gemini_api_key     = get_required_env("GEMINI_API_KEY")
    gcp_tts_credentials = get_required_env("GCP_TTS_CREDENTIALS")
    telegram_token     = get_required_env("TELEGRAM_BOT_TOKEN")
    telegram_chat_id   = get_required_env("TELEGRAM_CHAT_ID")

    # 1. Notion schema (with retry)
    col_topic, col_summary, col_script = auto_detect_notion_properties(
        notion_token, notion_database_id
    )

    # 2. Recent topics for dedup
    recent_topics = get_recent_topics(notion_token, notion_database_id)

    # 3. Fetch + score news (3 stories, ≥2 source overlap)
    news_stories = fetch_top_news(n=3)

    # 4. Synthesize neutral news block via Gemini
    news_block = synthesize_news_block(gemini_api_key, news_stories)

    # 5. Generate concept content (dense format, no storytelling)
    content = generate_topic_content(gemini_api_key, recent_topics)
    topic   = content["topic"]
    summary = content["summary"]
    concept_script = content["script"]

    # 6. Stitch full spoken script: news first, concept second
    section_break = "\n\nAnd now, today's concept.\n\n"
    if news_block:
        full_script = news_block + section_break + concept_script
    else:
        # Graceful degradation — news fetch failed, concept only
        logging.warning("No news content — proceeding with concept only.")
        full_script = concept_script

    # 7. TTS synthesis
    mp3_path = convert_script_to_speech(gcp_tts_credentials, full_script)

    # 8. Push full brief entry to Notion (concept + news)
    push_to_notion(
        notion_token, notion_database_id,
        topic, summary, concept_script,
        col_topic, col_summary, col_script,
        news_stories=news_stories,
        news_block=news_block,
    )

    # 9. Send to Telegram
    send_to_telegram(
        telegram_token, telegram_chat_id,
        mp3_path, topic, summary, news_stories
    )

    logging.info("=== Pipeline finished successfully. ===")


if __name__ == "__main__":
    main()
