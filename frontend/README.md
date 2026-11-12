# Frontend — Next.js

The user interface: image upload, preview, and the prediction returned by the Flask
model server.

## Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000. The Flask backend must already be running.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | `http://localhost:5000` | Base URL of the Flask server |

## Behaviour

- On load the page calls `GET /health` and shows a status pill — green when the model
  is loaded, red when the server is unreachable.
- The chosen image is previewed locally and sent to `POST /predict` as multipart form
  data.
- The result view shows the predicted sport, its softmax confidence, the sport's
  category and origin, the full top-5 probability distribution, and the inference
  metadata returned by the server.
