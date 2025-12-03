# Video Semantic Search with RAG

A semantic search system for YouTube videos using GPU-accelerated speech-to-text (Whisper) and vector similarity search (FAISS).

## Features

- **GPU-Accelerated Transcription**: Uses OpenAI's Whisper model with CUDA support for fast audio transcription
- **Semantic Search**: Converts text to embeddings using SentenceTransformers for meaning-based search
- **Vector Database**: FAISS for efficient similarity search across video transcripts
- **YouTube Integration**: Automatic audio download from YouTube videos
- **Persistent Storage**: Saves transcripts and embeddings to disk for reuse

## Architecture

```
YouTube Video → Audio Download → Whisper (GPU) → Text Chunks → 
Embeddings (CPU) → FAISS Index → Semantic Search
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4+ (for GPU acceleration)
- FFmpeg

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-semantic-search.git
cd video-semantic-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```python
# 1. Ingest a video
from video_rag import ingest_video

video_url = "https://www.youtube.com/watch?v=VIDEO_ID"
video_id = ingest_video(video_url)

# 2. Search the video
from video_rag import search

query = "What is machine learning?"
search(query, n_results=5)
```

### Jupyter Notebook

Open `video_rag.ipynb` and run the cells sequentially:

1. Install dependencies
2. Configure GPU/CPU settings
3. Load models (Whisper + SentenceTransformer)
4. Ingest videos
5. Search transcripts

## How It Works

### 1. Transcription
- Downloads audio from YouTube using `yt-dlp`
- Transcribes using Whisper model on GPU (or CPU fallback)
- Splits transcript into 30-second chunks with timestamps

### 2. Embedding Generation
- Converts text chunks to 384-dimensional vectors using `all-MiniLM-L6-v2`
- Runs on CPU to avoid GPU memory conflicts

### 3. Vector Storage
- Stores embeddings in FAISS index (L2 distance)
- Saves metadata (timestamps, video IDs) separately
- Persists to disk in `faiss_index/` directory

### 4. Search
- Converts search query to embedding
- Finds K-nearest neighbors in FAISS index
- Returns ranked results with timestamps and confidence scores

## Configuration

Key parameters in the notebook:

```python
DEVICE = "cuda"  # or "cpu"
MODEL_SIZE = "openai/whisper-base"  # or tiny, small, medium, large
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 5
```

## Memory Optimization

- **Whisper on GPU**: ~1.5 GB VRAM (base model)
- **Embeddings on CPU**: Avoids GPU memory conflicts
- **Chunk Processing**: Processes audio in 30-second chunks to prevent OOM errors

## Project Structure

```
.
├── video_rag.ipynb          # Main notebook (clean version)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── faiss_index/            # FAISS index and metadata (created at runtime)
└── audio_cache/            # Downloaded audio files (created at runtime)
```

## Performance

On NVIDIA GTX 1650 (4GB VRAM):
- **Transcription**: ~30 seconds for 10-minute video
- **Embedding**: ~1 second for 20 chunks
- **Search**: <100ms for queries

## Limitations

- Requires internet connection for YouTube downloads
- GPU memory limited by model size (use `whisper-tiny` for 2GB GPUs)
- Accuracy depends on audio quality and Whisper model size

## Technologies Used

- **[OpenAI Whisper](https://github.com/openai/whisper)**: Automatic speech recognition
- **[FAISS](https://github.com/facebookresearch/faiss)**: Vector similarity search
- **[SentenceTransformers](https://www.sbert.net/)**: Text embeddings
- **[yt-dlp](https://github.com/yt-dlp/yt-dlp)**: YouTube audio download
- **[PyTorch](https://pytorch.org/)**: Deep learning framework

## License

MIT License

## Acknowledgments

- OpenAI for Whisper model
- Facebook Research for FAISS
- Sentence Transformers team
