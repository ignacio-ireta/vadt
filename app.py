import os
import tempfile 

import streamlit as st
import torch
import torchaudio
import librosa
import numpy as np
import requests
from urllib.parse import urlparse
from pytubefix import YouTube
from speechbrain.inference import EncoderClassifier
import time

st.set_page_config(
    page_title="🗣️ Accent Detector",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .accent-badge {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .platform-info {
        background: #e3f2fd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>AI Accent Detector</h1>
    <p>Upload an audio file or provide any public video URL to detect English accents</p>
</div>
""", unsafe_allow_html=True)

# Load the accent detection model
@st.cache_resource
def load_accent_model():
    with st.spinner("Loading AI model... This may take a moment on first run."):
        classifier = EncoderClassifier.from_hparams(
            source="Jzuluaga/accent-id-commonaccent_ecapa",
            savedir="pretrained_models/accent-id"
        )
        classifier.hparams.label_encoder.expect_len(16)
        return classifier

# Convert audio file to 16kHz mono WAV due to model training on 16kHz mono audio
def process_audio_file(audio_path):
    try:
        signal_np, sr = librosa.load(audio_path, sr=None, mono=False)
        if signal_np.ndim == 1:
            signal = torch.tensor(signal_np).unsqueeze(0)
        else:
            signal = torch.tensor(signal_np)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        if signal.shape[0] > 1:
            mono = signal.mean(dim=0, keepdim=True)
        else:
            mono = signal
        processed_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        torchaudio.save(processed_path, mono, 16000)
        return processed_path
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# Funny enough all YouTube download libraries gave 400 errors, but this library seems to be the most up to date
def download_youtube_audio(url):
    try:
        with st.spinner("Downloading from YouTube..."):
            yt = YouTube(url)
            st.markdown(f"""
            <div class="platform-info">
                <strong>Video:</strong> {yt.title}<br>
                <strong>Duration:</strong> {yt.length // 60}:{yt.length % 60:02d}<br>
                <strong>Platform:</strong> YouTube
            </div>
            """, unsafe_allow_html=True)
            ys = yt.streams.get_audio_only()
            audio_path = ys.download(filename=tempfile.NamedTemporaryFile(suffix=".m4a", delete=False).name)
            return audio_path
    except Exception as e:
        st.error(f"Error downloading from YouTube: {str(e)}")
        return None

# For other platforms, we use yt-dlp
def download_with_ytdlp(url):
    try:
        import yt_dlp
        with st.spinner("Downloading from video platform..."):
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': tempfile.NamedTemporaryFile(suffix='.%(ext)s', delete=False).name,
                'noplaylist': True,
                'extract_flat': False,
                'quiet': True,
                'no_warnings': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                st.markdown(f"""
                <div class="platform-info">
                    <strong>Video:</strong> {info.get('title', 'Unknown')}<br>
                    <strong>Duration:</strong> {info.get('duration', 0) // 60 if info.get('duration') else 0}:{info.get('duration', 0) % 60 if info.get('duration') else 0:02d}<br>
                    <strong>Platform:</strong> {info.get('extractor_key', 'Unknown')}
                </div>
                """, unsafe_allow_html=True)
                downloaded_files = ydl.download([url])
                downloaded_file = ydl.prepare_filename(info)
                return downloaded_file
    except ImportError:
        st.error("yt-dlp not installed. Please install with: pip install yt-dlp")
        return None
    except Exception as e:
        st.error(f"Error downloading with yt-dlp: {str(e)}")
        return None

# For direct file links, we use requests
def download_direct_file(url):
    try:
        with st.spinner("Downloading direct media file..."):
            response = requests.head(url, allow_redirects=True, timeout=10)
            content_type = response.headers.get('content-type', '').lower()
            if any(media_type in content_type for media_type in ['audio', 'video']):
                parsed_url = urlparse(url)
                file_ext = os.path.splitext(parsed_url.path)[1] or '.mp4'
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                temp_file = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
                total_size = int(response.headers.get('content-length', 0))
                if total_size > 0:
                    progress_bar = st.progress(0)
                    downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                    if total_size > 0:
                        downloaded += len(chunk)
                        progress_bar.progress(downloaded / total_size)
                temp_file.close()
                st.markdown(f"""
                <div class="platform-info">
                    <strong>File:</strong> {os.path.basename(parsed_url.path) or 'Direct Media File'}<br>
                    <strong>Type:</strong> Direct Download
                </div>
                """, unsafe_allow_html=True)
                return temp_file.name
            else:
                st.error("URL doesn't point to a media file")
                return None
    except Exception as e:
        st.error(f"Error downloading direct file: {str(e)}")
        return None

def is_youtube_url(url):
    return any(domain in url.lower() for domain in ['youtube.com', 'youtu.be', 'm.youtube.com'])

def download_from_any_url(url):
    if is_youtube_url(url):
        st.info("Detected YouTube URL - using optimized YouTube downloader")
        return download_youtube_audio(url)
    try:
        st.info("Trying multi-platform downloader...")
        return download_with_ytdlp(url)
    except Exception as e:
        st.warning(f"Multi-platform downloader failed: {str(e)}")
    st.info("Trying direct file download...")
    return download_direct_file(url)

def detect_accent(audio_path, classifier):
    try:
        with st.spinner("Analyzing audio for accent detection..."):
            out_prob, score, index, text_lab = classifier.classify_file(audio_path)
            return text_lab[0], score.item()
    except Exception as e:
        st.error(f"Error during accent detection: {str(e)}")
        return None, None

def get_accent_info(accent_code):
    accent_map = {
        'us': {'name': 'American English', 'flag': '🇺🇸', 'region': 'North America'},
        'uk': {'name': 'British English', 'flag': '🇬🇧', 'region': 'United Kingdom'},
        'australian': {'name': 'Australian English', 'flag': '🇦🇺', 'region': 'Australia'},
        'canadian': {'name': 'Canadian English', 'flag': '🇨🇦', 'region': 'Canada'},
        'indian': {'name': 'Indian English', 'flag': '🇮🇳', 'region': 'India'},
        'irish': {'name': 'Irish English', 'flag': '🇮🇪', 'region': 'Ireland'},
        'scottish': {'name': 'Scottish English', 'flag': '🏴󠁧󠁢󠁳󠁣󠁴󠁿', 'region': 'Scotland'},
        'south_african': {'name': 'South African English', 'flag': '🇿🇦', 'region': 'South Africa'},
        'welsh': {'name': 'Welsh English', 'flag': '🏴󠁧󠁢󠁷󠁬󠁳󠁿', 'region': 'Wales'},
        'new_zealand': {'name': 'New Zealand English', 'flag': '🇳🇿', 'region': 'New Zealand'},
    }
    return accent_map.get(accent_code, {'name': accent_code.title(), 'flag': '🗣️', 'region': 'Unknown'})

tab1, tab2 = st.tabs(["📁 Upload Audio File", "🌐 Video URL Analysis"])

classifier = load_accent_model()

with tab1:
    st.markdown("### Upload an Audio File")
    st.markdown("*Supported formats: MP3, WAV, M4A, FLAC, OGG*")
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
        help="Upload an audio file containing English speech"
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Audio Preview")
            st.audio(uploaded_file, format='audio/wav')
        with col2:
            if st.button("Detect Accent", key="file_detect", type="primary"):
                processed_path = process_audio_file(temp_path)
                if processed_path:
                    accent, confidence = detect_accent(processed_path, classifier)
                    if accent and confidence is not None:
                        accent_info = get_accent_info(accent)
                        st.markdown("### Detection Results")
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>{accent_info['flag']} Detected Accent</h3>
                            <div class="accent-badge">{accent_info['name']}</div>
                            <p><strong>Region:</strong> {accent_info['region']}</p>
                            <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if 'processed_path' in locals() and os.path.exists(processed_path):
                    os.unlink(processed_path)

with tab2:
    st.markdown("### Video URL Analysis")
    st.markdown("*Enter any public video URL to analyze the audio for accent detection*")
    col1, col2 = st.columns([3, 1])
    with col1:
        video_url = st.text_input(
            "Video URL",
            placeholder="https://www.loom.com/share/... or any video URL",
            help="Paste any public video URL containing English speech"
        )
    with col2:
        detect_button = st.button("Analyze Video", key="video_detect", type="primary")
    if detect_button and video_url:
        if video_url.strip():
            audio_path = download_from_any_url(video_url.strip())
            if audio_path:
                processed_path = process_audio_file(audio_path)
                if processed_path:
                    accent, confidence = detect_accent(processed_path, classifier)
                    if accent and confidence is not None:
                        accent_info = get_accent_info(accent)
                        st.markdown("### Detection Results")
                        st.markdown(f"""
                        <div class="result-card">
                            <h3>{accent_info['flag']} Detected Accent</h3>
                            <div class="accent-badge">{accent_info['name']}</div>
                            <p><strong>Region:</strong> {accent_info['region']}</p>
                            <p><strong>Confidence Score:</strong> {confidence:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(confidence, text=f"Confidence: {confidence:.2%}")
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                if 'processed_path' in locals() and os.path.exists(processed_path):
                    os.unlink(processed_path)
        else:
            st.error("Please enter a valid video URL")

with st.sidebar:
    st.markdown("### About")
    st.markdown("""
    This app uses AI to detect English accents in audio recordings from any video platform or audio file.
    **Supported Accents:**
    - 🇺🇸 American English
    - 🇬🇧 British English
    - 🇦🇺 Australian English
    - 🇨🇦 Canadian English
    - 🇮🇳 Indian English
    - 🇮🇪 Irish English
    - 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scottish English
    - 🇿🇦 South African English
    - 🏴󠁧󠁢󠁷󠁬󠁳󠁿 Welsh English
    - 🇳🇿 New Zealand English
    **Tips for best results:**
    - Use clear audio recordings
    - Ensure speech is in English
    - Longer samples (>10 seconds) work better
    - Minimize background noise
    """)
    st.markdown("### Technical Details")
    st.markdown("""
    - **Model:** SpeechBrain ECAPA-TDNN
    - **Training:** CommonAccent dataset
    - **Sample Rate:** 16kHz mono audio
    - **Processing:** Librosa + PyTorch
    - **YouTube:** pytubefix (optimized)
    - **Other platforms:** yt-dlp
    """)