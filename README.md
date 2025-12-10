# 🔍 Image Similarity Search

A full-stack image retrieval system that finds visually similar images using CLIP embeddings and FAISS vector search — wrapped in a lightweight FastAPI backend and a clean React frontend.

This project demonstrates real-world ML engineering: preprocessing, embedding generation, vector indexing, API design, and UI integration.

## 🌟 Features

- **🧠 CLIP-based Embeddings** - Extracts 512-dimensional embeddings from images using OpenAI CLIP
- **⚡ FAISS Vector Index** - Fast cosine-similarity search across thousands of image embeddings
- **🎯 Accurate Similarity Results** - Returns top-k nearest images with similarity scores
- **🚀 FastAPI Backend** - Clean REST endpoint: upload an image → get similar images
- **💡 React Frontend (MVP)** - Upload an image and instantly visualize similarity matches
- **🗃️ Metadata Mapping** - Every indexed embedding links back to its image path
- **🔄 Extensible Architecture** - Add more models, more datasets, or deploy as a microservice

## 📁 Project Structure

```
ImageSimilaritySearch/
│
├── backend/
│   ├── main.py              # FastAPI server
│   ├── embeddings.py        # CLIP embedding generation
│   ├── faiss_index.py       # FAISS index utilities
│   ├── index_images.py      # Dataset indexing script
│   ├── test_client.py       # Test script for API
│   ├── metadata.json        # Maps index IDs to file paths
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── src/
│   │   ├── App.js           # UI logic & search integration
│   │   ├── api.js           # API client
│   │   ├── components/
│   │   │   └── SearchBox.jsx
│   │   └── ...
│   ├── public/
│   ├── package.json
│   └── ...
│
└── README.md
```

## 🧠 How It Works (High-Level Architecture)

```
User Uploads Image → Frontend → FastAPI → CLIP Embedding → FAISS Search →
→ Top-k Similar Images → Frontend Grid Display
```

### Under the hood:

1. **Indexing**
   - Read dataset images
   - Compute embeddings with CLIP
   - Normalize vectors
   - Store them in a FAISS index
   - Save metadata for ID→filepath mapping

2. **Searching**
   - User uploads query image
   - Generate embedding using CLIP
   - Normalize & search FAISS index
   - Return top matches with similarity scores

3. **Displaying results**
   - React fetches JSON from backend
   - Renders images in a responsive grid

## 🏗️ Setup Instructions

### 1️⃣ Create virtual environment (optional)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3️⃣ Index your images

```bash
python backend/index_images.py \
    --folder /path/to/dataset \
    --index-path backend/faiss_index.bin \
    --meta-path backend/metadata.json
```

This will automatically:
- scan images
- generate embeddings
- create/update FAISS index

### 4️⃣ Run FastAPI server

```bash
uvicorn backend.main:app --reload --port 8000
```

Visit API docs:  
👉 http://localhost:8000/docs

### 5️⃣ Run frontend

```bash
cd frontend
npm install
npm start
```

Frontend runs on:  
👉 http://localhost:3000

## 🧪 Testing the Search API

Use the provided test client:

```bash
python backend/test_client.py path/to/query.jpg
```

Or via cURL:

```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/search-image
```

## 📸 Demo (Screenshots)

_Add screenshots here once the UI is running — I can help you generate perfect ones._

## 🚀 Future Improvements

- Swap placeholder embedding function with integrated CLIP encoder (already implemented in `embeddings.py`)
- Add batching for faster indexing
- Add GPU support for embedding generation
- Deploy backend to Render / AWS / GCP
- Containerize using Docker
- Add authentication or rate limiting
- Add support for multimodal search

## 🤝 Contributing

Pull requests, suggestions, and improvements are welcome. This codebase is intentionally modular, making it easy to extend or experiment.

## 📝 License

MIT License — free to use, modify, and distribute.