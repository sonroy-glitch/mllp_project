# MLLP Project — Sports Identification from Images

VIDEO LINK : https://drive.google.com/drive/folders/1RDLAmeSu90a9BJATeyg5CkJZwCMdh59M?usp=sharing


**Task 22 — Sports (Kabaddi, Kho-Kho, Hockey, etc.)**
Experiential Learning Group Assignment: *Language Processing Using Machine Learning and
LLMs for Audio, Image, Text and Other Applications*

A web application that identifies the sport shown in a photograph. The image is
classified by a transfer-learning model built on EfficientNetB0, served from a Flask
API and consumed by a Next.js frontend. The classifier covers seven sports: cricket,
wrestling, tennis, badminton, soccer, swimming and karate.

## Dataset

Seven sport classes, 8,227 labelled images in total.

| Class | Images | Share |
| --- | --- | --- |
| Cricket | 1,556 | 18.9% |
| Wrestling | 1,471 | 17.9% |
| Tennis | 1,445 | 17.6% |
| Badminton | 1,394 | 16.9% |
| Soccer | 1,188 | 14.4% |
| Swimming | 595 | 7.2% |
| Karate | 578 | 7.0% |
| **Total** | **8,227** | **100%** |

The split is 80% training (6,582 images) and 20% validation (1,645 images), taken with a
fixed seed. The largest class holds 2.7 times as many images as the smallest, so
macro-averaged F1 is reported alongside overall accuracy.

Class order is alphabetical — `Badminton, Cricket, Karate, Soccer, Swimming, Tennis,
Wrestling` — because `image_dataset_from_directory` sorts the folder names. That order
defines the softmax index of every class and must match `class_names.json`.

## Architecture

```
┌──────────────────────┐        multipart/form-data        ┌────────────────────────┐
│  Next.js frontend    │  ──────  POST /predict  ───────▶  │  Flask model server    │
│  localhost:3000      │                                   │  localhost:5000        │
│                      │  ◀─────  JSON prediction  ──────  │                        │
│  upload · preview    │                                   │  EfficientNetB0 + head │
│  confidence · top-5  │        GET /health, /classes      │  224×224×3 → softmax   │
└──────────────────────┘                                   └────────────────────────┘
```

## Processing pipeline

| Stage | Operation |
| --- | --- |
| 1. Acquisition | User uploads a JPEG/PNG/WebP/GIF image (max 5 MB) |
| 2. Transport | Image posted as multipart form data to `POST /predict` |
| 3. Validation | Extension, size and decodability checks |
| 4. Preprocessing | RGB conversion → resize to 224×224 → `efficientnet.preprocess_input` → batch axis |
| 5. Feature extraction | Frozen EfficientNetB0 convolutional base (ImageNet weights) |
| 6. Classification head | GlobalAveragePooling2D → BatchNormalization → Dense(256, ReLU) → Dropout(0.4) → Dense(n, softmax) |
| 7. Post-processing | Top-5 ranking, label lookup, category/origin enrichment |
| 8. Presentation | Confidence bar, probability distribution and inference metadata |

## Model

| Property | Value |
| --- | --- |
| Backbone | `EfficientNetB0`, ImageNet weights, `include_top=False`, frozen |
| Input | 224 × 224 × 3 |
| Head | GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.4) → Dense(num_classes, softmax) |
| Optimiser | Adam, learning rate 1e-3 |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Artefact | `backend/model/sports_model.pkl` |

## Repository layout

```
mllp_project/
├── backend/
│   ├── app.py                 Flask application — /health, /classes, /predict
│   ├── model_loader.py        Model discovery, unpickling, image preprocessing
│   ├── sports_metadata.py     Label list and per-sport descriptions
│   ├── train_model.py         Training script that produces the model artefact
│   ├── requirements.txt
│   └── model/                 Trained model goes here (see model/README.md)
├── frontend/
│   ├── app/page.tsx           Upload, preview, prediction view
│   ├── app/layout.tsx
│   └── app/globals.css
├── docs/
│   ├── Project_Report.docx       Project report in the prescribed format
│   ├── Project_Report.pdf        The same report, for submission
│   ├── Project_Presentation.pptx Presentation deck, 19 slides with speaker notes
│   └── figures/                  Dataset, training and evaluation figures
└── README.md
```

## Running the project

Two terminals.

**Terminal 1 — backend**

```bash
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python app.py
```

**Terminal 2 — frontend**

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000.

> On macOS the AirPlay Receiver occupies port 5000. Run `PORT=5001 python app.py` and
> set `NEXT_PUBLIC_API_URL=http://localhost:5001` in `frontend/.env.local`.

## Results

| Metric | Value |
| --- | --- |
| Validation accuracy | **97.45%** (1,604 of 1,646 correct) |
| Macro-averaged F1 | **97.61%** |
| Weighted-average F1 | 97.45% |
| Validation errors | 42 images |
| Lowest class recall | 0.9619 (Tennis) |
| Highest class recall | 0.9914 (Karate) |
| Epochs trained | 30 |

Per-class performance on the 1,646-image validation split:

| Class | Precision | Recall | F1 | Support | Errors |
| --- | --- | --- | --- | --- | --- |
| Badminton | 0.9610 | 0.9713 | 0.9661 | 279 | 8 |
| Cricket | 0.9775 | 0.9775 | 0.9775 | 311 | 7 |
| Karate | 0.9664 | 0.9914 | 0.9787 | 116 | 1 |
| Soccer | 0.9833 | 0.9874 | 0.9853 | 238 | 3 |
| Swimming | 1.0000 | 0.9748 | 0.9872 | 119 | 3 |
| Tennis | 0.9521 | 0.9619 | 0.9570 | 289 | 11 |
| Wrestling | 0.9930 | 0.9694 | 0.9811 | 294 | 9 |
| **Macro average** | **0.9762** | **0.9762** | **0.9761** | **1,646** | **42** |

The errors are concentrated rather than diffuse. Badminton ↔ Tennis alone accounts for 10
of the 42 (23.8%), and just over 40% of all errors involve Tennis on one side or the other
— a fine-grained discrimination failure between two sports that share a court, a net and a
swing. Resolving that one pair would lift overall accuracy to approximately 98.06%.

## Documentation

- `docs/Project_Report.docx` — the full project report
- `backend/README.md` — API reference and training instructions
- `backend/model/README.md` — model file format and class-name contract
- `frontend/README.md` — frontend configuration
