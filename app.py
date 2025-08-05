import streamlit as st
import json
import os
import uuid
import tempfile
import re
from io import BytesIO
from datetime import datetime
from PIL import Image
import requests
import boto3
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import azure.cognitiveservices.speech as speechsdk

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
S3_PREFIX            = st.secrets.get("S3_PREFIX", "media/")
CDN_BASE             = st.secrets.get("CDN_BASE", "https://media.suvichaar.org/")
DEFAULT_ERROR_IMAGE  = st.secrets.get("DEFAULT_ERROR_IMAGE", "https://media.suvichaar.org/default-error.jpg")
HTML_TEMPLATE        = "template-v18.html"

# -------------------------------
# Helper functions
# -------------------------------
def extract_text_with_document_intelligence(image_path: str) -> str:
    client = DocumentIntelligenceClient(
        endpoint=AZURE_DI_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DI_KEY)
    )
    with open(image_path, "rb") as f:
        file_data = f.read()
    poller = client.begin_analyze_document("prebuilt-read", body=file_data)
    result = poller.result()
    return "\n".join(p.content for p in result.paragraphs)


def chat_completion(messages: list, max_tokens: int = 1500, temperature: float = 0.7) -> str:
    url = f"{GPT_ENDPOINT}/openai/deployments/{GPT_DEPLOYMENT}/chat/completions?api-version={GPT_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": GPT_KEY}
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


def generate_story(document_text: str) -> dict:
    system = (
        "You are a teaching assistant. Create a JSON with keys: storytitle, s2paragraph1…s6paragraph1, "+
        "and s1alt1…s6alt1."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": document_text}]
    content = chat_completion(messages)
    return json.loads(content)


def generate_seo(data: dict) -> dict:
    prompt = (
        "Generate SEO metadata *strictly* as JSON. Wrap your answer in ```json ... ``` with no extra text.\n\n" +
        f"Title: {data['storytitle']}\nSlides:\n- {data['s2paragraph1']}\n- {data['s3paragraph1']}\n" +
        f"- {data['s4paragraph1']}\n- {data['s5paragraph1']}\n- {data['s6paragraph1']}\nReturn keys: metadescription, metakeywords."
    )
    messages = [{"role": "system", "content": "You are an expert SEO assistant."},
                {"role": "user", "content": prompt}]
    content = chat_completion(messages, max_tokens=300, temperature=0.5)
    m = re.search(r"```json(.*?)```", content, re.S)
    raw = m.group(1).strip() if m else content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.warning("⚠️ SEO JSON parse failed; using empty metadata.")
        return {"metadescription": "", "metakeywords": ""}


def generate_images_and_upload(data: dict) -> dict:
    s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY,
                       aws_secret_access_key=AWS_SECRET_KEY,
                       region_name=AWS_REGION)
    slug = data["storytitle"].lower().replace(" ", "-").replace(":", "")
    for i in range(1, 7):
        prompt = data.get(f"s{i}alt1", "")
        try:
            headers = {"Content-Type":"application/json", "api-key": DALE_KEY}
            payload = {"prompt": prompt, "n": 1, "size": "1024x1024"}
            dr = requests.post(DALE_ENDPOINT, headers=headers, json=payload)
            dr.raise_for_status()
            img_url = dr.json()["data"][0]["url"]
            img_data = requests.get(img_url).content
            img = Image.open(BytesIO(img_data)).convert("RGB").resize((720,1200))
        except Exception as e:
            st.error(f"Image gen failed for slide {i}: {e}")
            img = Image.open(requests.get(DEFAULT_ERROR_IMAGE, stream=True).raw).convert("RGB").resize((720,1200))
        buf = BytesIO(); img.save(buf, "JPEG"); buf.seek(0)
        key = f"{S3_PREFIX}/{slug}/slide{i}.jpg"
        s3.upload_fileobj(buf, AWS_BUCKET, key)
        data[f"s{i}image1"] = f"{CDN_BASE}{key}"
    # portrait cover
    cover_img = img
    buf = BytesIO(); cover_img.save(buf, "JPEG"); buf.seek(0)
    cover_key = f"{S3_PREFIX}/{slug}/portrait_cover.jpg"
    s3.upload_fileobj(buf, AWS_BUCKET, cover_key)
    data["portraitcoverurl"] = f"{CDN_BASE}{cover_key}"
    return data


def synthesize_and_upload_audio(data: dict) -> dict:
    s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY,
                       aws_secret_access_key=AWS_SECRET_KEY,
                       region_name=AWS_REGION)
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
    speech_config.speech_synthesis_voice_name = VOICE_NAME
    for field, audio_key in {
        "storytitle": "s1audio1", "s2paragraph1": "s2audio1",
        "s3paragraph1": "s3audio1", "s4paragraph1": "s4audio1",
        "s5paragraph1": "s5audio1", "s6paragraph1": "s6audio1"
    }.items():
        text = data.get(field)
        if not text: continue
        fn = f"{uuid.uuid4().hex}.mp3"
        audio_config = speechsdk.audio.AudioOutputConfig(filename=fn)
        synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
        synthesizer.speak_text_async(text).get()
        key = f"{S3_PREFIX}/audio/{fn}"
        s3.upload_file(fn, AWS_BUCKET, key)
        data[audio_key] = f"{CDN_BASE}{key}"
    return data


def fill_template(template: str, data: dict) -> str:
    for k, v in data.items():
        template = template.replace(f"{{{{{k}}}}}", v)
    return template

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Notes → Web Story", layout="centered")
st.title("Lecture Notes → AMP Web Story Generator")

uploaded = st.file_uploader("Upload your notes image:", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Please upload an image to get started.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
    tmp.write(uploaded.read())
    img_path = tmp.name

st.image(img_path, caption="Uploaded Notes", use_column_width=True)

with st.spinner("Extracting text…"):
    text = extract_text_with_document_intelligence(img_path)
st.success("Text extracted!")

with st.spinner("Generating story & prompts…"):
    story = generate_story(text)
st.success("Story generated!")

with st.spinner("Creating SEO metadata…"):
    seo = generate_seo(story)
    story.update(seo)
st.success("SEO metadata added!")

with st.spinner("Generating and uploading images…"):
    story = generate_images_and_upload(story)
st.success("Images uploaded!")

with st.spinner("Synthesizing and uploading audio…"):
    story = synthesize_and_upload_audio(story)
st.success("Audio uploaded!")

with open(HTML_TEMPLATE, "r", encoding="utf-8") as f:
    tpl = f.read()
final_html = fill_template(tpl, story)

st.download_button(
    label="Download HTML Web Story",
    data=final_html,
    file_name=f"{story['storytitle'].lower().replace(' ', '_')}.html",
    mime="text/html"
)

st.balloons()
