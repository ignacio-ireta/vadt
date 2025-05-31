# AI Accent Detector

A powerful Streamlit web application that uses AI to detect English accents in audio recordings. Upload audio files or provide YouTube links to identify different English accents with confidence scores.

## Features

- **Audio File Upload**: Support for MP3, WAV, M4A, FLAC, OGG formats
- **YouTube Integration**: Direct analysis of YouTube video audio
- **10+ Accent Detection**: Identifies American, British, Australian, Canadian, Indian, Irish, Scottish, South African, Welsh, and New Zealand English accents
- **Beautiful UI**: Modern, responsive interface with real-time progress indicators
- **Confidence Scoring**: Provides percentage confidence for each prediction
- **Audio Preview**: Built-in audio player for uploaded files

## Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd vadt
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Using uv
   uv run streamlit run app.py
   
   # Or using streamlit directly
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Connect your GitHub account** and select this repository

4. **Set the main file path** to `app.py`

5. **Deploy!** The app will automatically install dependencies and start running

## Technology Stack

- **Frontend**: Streamlit
- **AI Model**: SpeechBrain ECAPA-TDNN
- **Audio Processing**: Librosa, PyTorch, torchaudio
- **YouTube Integration**: pytubefix
- **System Dependencies**: FFmpeg (auto-installed on Streamlit Cloud)

## Project Structure

```
vadt/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── packages.txt        # System packages for Streamlit Cloud
├── pyproject.toml      # uv project configuration
└── README.md           # This file
```

## Supported Accents

- 🇺🇸 **American English** - North America
- 🇬🇧 **British English** - United Kingdom  
- 🇦🇺 **Australian English** - Australia
- 🇨🇦 **Canadian English** - Canada
- 🇮🇳 **Indian English** - India
- 🇮🇪 **Irish English** - Ireland
- 🏴󠁧󠁢󠁳󠁣󠁴󠁿 **Scottish English** - Scotland
- 🇿🇦 **South African English** - South Africa
- 🏴󠁧󠁢󠁷󠁬󠁳󠁿 **Welsh English** - Wales
- 🇳🇿 **New Zealand English** - New Zealand

## Usage Tips

- **Audio Quality**: Use clear recordings with minimal background noise
- **Language**: Ensure the audio contains English speech
- **Duration**: Longer samples (>10 seconds) typically provide better results
- **Format**: The app handles format conversion automatically
- **YouTube**: Use public YouTube videos with clear speech

## Technical Details

- **Model**: Pre-trained SpeechBrain ECAPA-TDNN model
- **Training Data**: CommonAccent dataset
- **Audio Processing**: 16kHz mono audio normalization
- **Backend**: Librosa with FFmpeg fallback for format support

## Troubleshooting

### Audio Loading Issues
If you encounter audio loading problems:
1. Ensure FFmpeg is installed on your system
2. Try converting your audio to WAV format first
3. Check that the audio file isn't corrupted

### YouTube Download Issues
If YouTube videos won't download:
1. Ensure the video is public and available
2. Check your internet connection
3. Some videos may have download restrictions

### Model Loading
The AI model downloads automatically on first run (~100MB). This may take a few minutes initially.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

Built with Streamlit and SpeechBrain
