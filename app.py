import streamlit as st
import json
import requests
import boto3
import tempfile
import uuid
import re
from io import BytesIO
from datetime import datetime
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import azure.cognitiveservices.speech as speechsdk
import pandas as pd

# -------------------------------
# Configuration (Streamlit secrets)
# -------------------------------
AZURE_DI_KEY         = st.secrets["AZURE_DI_KEY"]
AZURE_DI_ENDPOINT    = st.secrets["AZURE_DI_ENDPOINT"]
GPT_KEY              = st.secrets["GPT_KEY"]
GPT_ENDPOINT         = st.secrets["GPT_ENDPOINT"]
GPT_DEPLOYMENT       = st.secrets["GPT_DEPLOYMENT"]
GPT_API_VERSION      = st.secrets["GPT_API_VERSION"]
DALE_KEY             = st.secrets["DALE_KEY"]
DALE_ENDPOINT        = st.secrets["DALE_ENDPOINT"]
AZURE_SPEECH_KEY     = st.secrets["AZURE_SPEECH_KEY"]
AZURE_REGION         = st.secrets["AZURE_REGION"]
VOICE_NAME           = st.secrets.get("VOICE_NAME", "en-IN-AaravNeural")
AWS_ACCESS_KEY       = st.secrets["AWS_ACCESS_KEY"]
AWS_SECRET_KEY       = st.secrets["AWS_SECRET_KEY"]
AWS_REGION           = st.secrets.get("AWS_REGION", "ap-south-1")
AWS_BUCKET           = st.secrets["AWS_BUCKET"]
S3_PREFIX            = st.secrets.get("S3_PREFIX", "media")
CDN_BASE             = st.secrets.get("CDN_BASE", "https://media.suvichaar.org")
DEFAULT_ERROR_IMAGE  = st.secrets.get("DEFAULT_ERROR_IMAGE", "https://media.suvichaar.org/default-error.jpg")

# -------------------------------
# Helper functions
# -------------------------------

def extract_text(image_path: str) -> str:
    client = DocumentIntelligenceClient(
        endpoint=AZURE_DI_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DI_KEY)
    )
    with open(image_path, "rb") as f:
        data = f.read()
    poller = client.begin_analyze_document("prebuilt-read", body=data)
    result = poller.result()
    return "\n".join(p.content for p in result.paragraphs)


def call_gpt_system(system_prompt: str, user_content: str, max_tokens: int=1500, temperature: float=0.7) -> str:
    url = f"{GPT_ENDPOINT}/openai/deployments/{GPT_DEPLOYMENT}/chat/completions?api-version={GPT_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": GPT_KEY}
    payload = {"messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content}
    ],
    "temperature": temperature,
    "max_tokens": max_tokens}
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def generate_story(text: str) -> dict:
    system_prompt = (
        "You are a teaching assistant. "
        "Create JSON with keys: storytitle, s2paragraph1…s6paragraph1, and s1alt1…s6alt1."
    )
    content = call_gpt_system(system_prompt, text)
    return json.loads(content)


def generate_seo(story: dict) -> dict:
    slides = "\n".join(f"- {story[f's{i}paragraph1']}" for i in range(2,7))
    seo_prompt = (
        "Generate SEO metadata strictly as JSON. Wrap in ```json``` only.\n" +
        f"Title: {story['storytitle']}\nSlides:\n{slides}\nReturn keys: metadescription, metakeywords."
    )
    content = call_gpt_system("You are an expert SEO assistant.", seo_prompt, max_tokens=300, temperature=0.5)
    m = re.search(r"```json(.*?)```", content, re.S)
    raw = m.group(1).strip() if m else content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.warning("⚠️ SEO parsing failed; using empty metadata.")
        return {"metadescription": "", "metakeywords": ""}


def generate_images(story: dict) -> dict:
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    slug = story['storytitle'].lower().replace(' ', '-')
    for i in range(1, 7):
        prompt = story.get(f's{i}alt1', '')
        try:
            hdr = {"Content-Type": "application/json", "api-key": DALE_KEY}
            pl = {"prompt": prompt, "n": 1, "size": "1024x1024"}
            dr = requests.post(DALE_ENDPOINT, headers=hdr, json=pl)
            dr.raise_for_status()
            url = dr.json()['data'][0]['url']
            img = Image.open(BytesIO(requests.get(url).content))
            img = img.convert("RGB").resize((720, 1200))
        except Exception as e:
            st.error(f"Image gen failed slide {i}: {e}")
            img = Image.open(requests.get(DEFAULT_ERROR_IMAGE, stream=True).raw)
            img = img.convert("RGB").resize((720, 1200))
        buf = BytesIO()
        img.save(buf, "JPEG")
        buf.seek(0)
        key = f"{S3_PREFIX}/{slug}/slide{i}.jpg"
        s3.upload_fileobj(buf, AWS_BUCKET, key)
        story[f's{i}image1'] = f"{CDN_BASE}/{key}"
    return story


def synthesize_audio(story: dict) -> (dict, dict):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = VOICE_NAME
    cdn_urls = {}
    for field, audio_key in {
        'storytitle': 's1audio1',
        's2paragraph1': 's2audio1', 's3paragraph1': 's3audio1',
        's4paragraph1': 's4audio1', 's5paragraph1': 's5audio1',
        's6paragraph1': 's6audio1'
    }.items():
        text = story.get(field)
        if not text:
            continue
        fn = f"{uuid.uuid4().hex}.mp3"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=fn)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
        synthesizer.speak_text_async(text).get()
        key = f"{S3_PREFIX}/audio/{fn}"
        s3.upload_file(fn, AWS_BUCKET, key)
        url = f"{CDN_BASE}/{key}"
        story[audio_key] = url
        cdn_urls[field] = url
    return story, cdn_urls


def fill_template(template: str, story: dict) -> str:
    for k, v in story.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="Notes → Web Story", layout="centered")
st.title("Lecture Notes → AMP Web Story Generator")

# 1. Upload notes image
img_file = st.file_uploader("1. Upload lecture notes image", type=["jpg","png","jpeg"])
# 2. Upload HTML template
template_file = st.file_uploader("2. Upload HTML template (.html)", type=["html"])

if img_file and template_file:
    # Save uploaded template
    template_html = template_file.read().decode('utf-8')

    # Save uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(img_file.read())
        img_path = tmp.name
    st.image(img_path, caption="Notes Image", use_column_width=True)

    # OCR Extract
    with st.spinner("Extracting text..."):
        text = extract_text(img_path)
    st.text_area("Extracted Text", text, height=200)

    # GPT Story + Prompts
    with st.spinner("Generating story & prompts..."):
        story = generate_story(text)
    st.json(story)

    # SEO Metadata
    with st.spinner("Generating SEO metadata..."):
        seo = generate_seo(story)
        story.update(seo)
    st.json(seo)

    # DALL·E Images
    with st.spinner("Generating and uploading images..."):
        story = generate_images(story)
    cols = st.columns(3)
    for i in range(1, 7):
        cols[(i-1) % 3].image(story.get(f's{i}image1'), width=150)

    # TTS Audio
    with st.spinner("Synthesizing audio..."):
        story, cdn_urls = synthesize_audio(story)
    st.table(pd.DataFrame(cdn_urls.items(), columns=["Field","CDN URL"]))

    # Download final JSON
    st.download_button(
        label="Download final JSON",
        data=json.dumps(story, indent=2),
        file_name=f"{story['storytitle'].lower().replace(' ','_')}.json",
        mime="application/json"
    )

    # Fill HTML and download
    final_html = fill_template(template_html, story)
    st.download_button(
        label="Download HTML Web Story",
        data=final_html,
        file_name=f"{story['storytitle'].lower().replace(' ','_')}.html",
        mime="text/html"
    )
    st.balloons()
