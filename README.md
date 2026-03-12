# Color Sketchy Retrieval

An image retrieval platform supporting multiple datasets and embedding models (CLIP, SigLIP2).

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/mattiamondo/color_sketchy_retrieval.git
cd color_sketchy_retrieval
conda env create -f environment.yml
conda activate sketchy_retrieval
```

### 2. Build the frontend

```bash
cd frontend
npx vite build
cd ..
```

### 3. Set up data and embeddings

Images and `.npy` embedding files are not tracked in git. Populate them manually following this structure:

```
data/
├── color/
│   ├── images/
│   └── metadata.json
├── sketchy_test/
│   ├── images/
│   └── metadata.json
└── flickr30k/
    ├── flickr30k_images/
    └── metadata.json

embeddings/
├── color/
│   ├── siglip2_image.npy
│   └── siglip2_text.npy
├── sketchy_test/
│   ├── siglip2_image.npy
│   └── siglip2_text.npy
└── flickr30k_first_caption/
    ├── siglip2_image.npy
    ├── siglip2_text.npy
    └── selected_records_first_caption.jsonl
```

### 4. Run the backend

```bash
python api.py --port 8083
```

If `--port` is omitted, the server defaults to port `8000`.
