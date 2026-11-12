# Backend — Flask model server

Serves a fine-tuned EfficientNetB0 sports classifier over HTTP.

## Setup

```bash
cd backend
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Put the trained model in `model/` (see `model/README.md`), then:

```bash
python app.py
```

The server listens on `http://localhost:5000`.

> **macOS:** port 5000 is used by the AirPlay Receiver. Either disable it in
> System Settings → General → AirDrop & Handoff, or run `PORT=5001 python app.py`
> and set `NEXT_PUBLIC_API_URL=http://localhost:5001` in the frontend.

## Endpoints

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_file": "sports_model.pkl",
  "num_classes": 7,
  "input_size": 224,
  "error": null
}
```

### `GET /classes`

Returns the label list in model index order.

### `POST /predict`

Multipart form upload with the image in the `image` field.

```bash
curl -F "image=@cricket.jpg" http://localhost:5000/predict
```

```json
{
  "sport": "Cricket",
  "confidence": 0.9412,
  "category": "Team Sport",
  "origin": "England",
  "top_k": [
    { "sport": "Cricket", "confidence": 0.9412 },
    { "sport": "Soccer", "confidence": 0.0331 }
  ],
  "description": "...",
  "meta": {
    "model": "sports_model.pkl",
    "backbone": "EfficientNetB0 (ImageNet, frozen)",
    "input_size": 224,
    "num_classes": 7,
    "inference_ms": 128
  }
}
```

## Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask application and the three routes |
| `model_loader.py` | Locates, unpickles and warms the model; image preprocessing |
| `sports_metadata.py` | Default label list and the per-sport description table |
| `train_model.py` | Training script that produces the `.pkl` |
| `model/` | Where the trained model file goes |

## Training

```bash
python train_model.py --data-dir /path/to/dataset --epochs 20
```

The dataset must be one folder per class:

```
dataset/
  Badminton/
  Cricket/
  Karate/
  Soccer/
  Swimming/
  Tennis/
  Wrestling/
```

The script writes `sports_model.pkl`, `sports_model.keras`, `class_names.json` and
`training_history.json` into `model/`.

## Production

```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

One worker is deliberate — each worker holds its own copy of the model in memory.
