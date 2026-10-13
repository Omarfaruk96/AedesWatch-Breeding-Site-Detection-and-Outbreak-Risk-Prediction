# AedesWatch 🦟
### Aedes Mosquito Breeding-Site Detection & Outbreak Risk Prediction

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)

> Aedes mosquito breeding-site detection (YOLOv8) + dengue/Zika outbreak risk
> prediction (ML) for public health surveillance. Includes a synthetic data
> pipeline, a trained risk model, and a Streamlit demo app.

**Topics:** `machine-learning` `computer-vision` `yolov8` `public-health` `dengue` `mosquito-detection` `scikit-learn` `streamlit` `python`

---

AedesWatch combines two complementary ML pipelines to support mosquito-borne
disease prevention programs:

1. **Detection (Computer Vision)** — a YOLOv8-based pipeline that spots
   standing-water breeding sites (tires, buckets, flowerpots, drains) and
   adult *Aedes aegypti* / *Aedes albopictus* mosquitoes in field photos.
2. **Outbreak Risk Prediction (Tabular / Time-Series)** — a gradient-boosted
   model that estimates short-term dengue/Zika outbreak risk per
   neighborhood from breeding-site density, rainfall, temperature, and
   humidity.

A lightweight **Streamlit demo app** ties both pipelines together: upload a
photo, get a breeding-site flag, and see the current risk score for the
selected region.

> Built as a portfolio project. The repo ships with a **synthetic data
> generator** so every pipeline runs end-to-end with zero downloads. Swap in
> real data (see [Data Sources](#data-sources)) to get a research-grade
> result.

---

## Why this project

Dengue and other Aedes-borne diseases are a recurring public health burden
in South and Southeast Asia, including Bangladesh, where seasonal dengue
outbreaks strain hospitals every monsoon. Most breeding-site surveillance is
still done manually by community health workers walking door to door.
AedesWatch demonstrates how a simple CV + tabular ML pipeline could support
that fieldwork — flagging likely breeding sites from phone photos and
surfacing which neighborhoods need priority inspection.

It's also a deliberately broad showcase: object detection, tabular ML,
a small FastAPI/Streamlit serving layer, tests, and CI in one repo.

---

## Repository structure

```
AedesWatch/
├── app/                      # Streamlit demo app
│   └── streamlit_app.py
├── data/
│   ├── sample/                # tiny synthetic sample outputs (checked in)
│   └── generate_synthetic_data.py
├── src/
│   ├── detection/              # CV pipeline (YOLOv8)
│   │   ├── dataset_prep.py
│   │   ├── train.py
│   │   ├── infer.py
│   │   └── config/aedeswatch.yaml
│   └── risk_prediction/        # Tabular risk model
│       ├── features.py
│       ├── train.py
│       ├── predict.py
│       └── model.py
├── notebooks/
│   └── 01_eda_risk_data.ipynb
├── tests/
│   ├── test_features.py
│   └── test_risk_predict.py
├── .github/workflows/ci.yml
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/AedesWatch.git
cd AedesWatch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate synthetic risk-prediction data (no external downloads needed)
python data/generate_synthetic_data.py --out data/sample

# 2. Train the risk prediction model
python src/risk_prediction/train.py --data data/sample/risk_training_data.csv \
                                     --out models/risk_model.joblib

# 3. Predict risk for a sample region/week
python src/risk_prediction/predict.py --model models/risk_model.joblib \
                                       --input data/sample/risk_inference_example.csv

# 4. (Optional, needs a GPU + real images) Train the detection model
python src/detection/train.py --data src/detection/config/aedeswatch.yaml --epochs 50

# 5. Launch the demo app
streamlit run app/streamlit_app.py
```

The Streamlit app works out of the box using the risk model from step 2. The
detection half will show a clear "no weights found — using mock detector"
message until you've trained or supplied a YOLO checkpoint — it doesn't fail,
it just demonstrates the input/output contract.

---

## Detection pipeline

- Model: **YOLOv8n/s** (via `ultralytics`), fine-tuned for two use cases:
  - **Breeding-site detection**: tires, buckets, flowerpots, open drains,
    discarded containers holding standing water.
  - **Mosquito species ID** (stretch goal): distinguishing *Aedes aegypti*
    (lyre-shaped white markings on thorax) from *Aedes albopictus* (single
    white dorsal stripe) and from *Culex*/*Anopheles*.
- `src/detection/dataset_prep.py` converts a folder of labeled images
  (COCO or YOLO-format annotations) into the YOLO training layout and
  writes the `data.yaml` the trainer expects.
- `src/detection/train.py` is a thin wrapper around `ultralytics.YOLO`
  so you can run `python src/detection/train.py --data ... --epochs 50`
  without touching the Ultralytics CLI directly.
- `src/detection/infer.py` runs inference on a single image or folder and
  returns bounding boxes + confidence, used by the demo app.

### Data sources for real training

| Source | What it provides |
|---|---|
| [Mosquito Alert](https://www.mosquitoalert.com/en/) | Citizen-science photos of mosquitoes & bites, Spain-led but global submissions, with species labels |
| [AIcrowd "Mosquito Alert" challenge dataset](https://www.aicrowd.com/) | ~15k labeled mosquito images from the 2023/2024 challenge, ready for YOLO/classification fine-tuning |
| [iNaturalist research-grade observations](https://www.inaturalist.org/) (filter: genus *Aedes*) | Large volume of geotagged mosquito photos, exportable via their API |
| Your own field photos | Standing-water containers around your area — even a few hundred labeled with [LabelImg](https://github.com/HumanSignal/labelImg) or [CVAT](https://www.cvat.ai/) will noticeably improve the breeding-site detector |

This repo does **not** bundle any third-party images (to respect licensing)
— `data/generate_synthetic_data.py` only produces synthetic tabular data for
the risk model. See `src/detection/dataset_prep.py` docstring for the exact
folder layout to drop real images into.

---

## Risk prediction pipeline

A gradient-boosted tree model (`scikit-learn`'s `HistGradientBoostingRegressor`,
swappable for XGBoost/LightGBM) predicts a 0–100 **outbreak risk score** for
a given region-week from:

- `breeding_site_density` — detections per 100 inspected properties (from
  the detection pipeline, or manual surveillance counts)
- `rainfall_mm_7d`, `rainfall_mm_14d` — rolling rainfall (standing water lag)
- `avg_temp_c`, `humidity_pct` — Aedes breeding is temperature/humidity
  sensitive (optimal ~25–30°C, high humidity)
- `population_density`, `prior_cases_4wk` — prior outbreak momentum
- `season_week` — cyclical week-of-year encoding (captures monsoon
  seasonality)

`src/risk_prediction/features.py` holds all feature engineering so it's
unit-testable and reusable between training and inference.

### Real-world data to plug in

- **Weather**: [Open-Meteo](https://open-meteo.com/) (free, no API key) or
  your national meteorological department
- **Case counts**: Bangladesh DGHS dengue dashboard, Brazil's SINAN, WHO's
  [Global Arbovirus Dashboard](https://worldhealthorg.shinyapps.io/dengue_global/),
  or your local health authority's open data portal

---

## Demo app

`app/streamlit_app.py`:
- **Tab 1 – Inspect a photo**: upload an image → detection pipeline flags
  likely breeding sites with bounding boxes.
- **Tab 2 – Regional risk**: pick a region + week → risk model returns a
  score and the top contributing features (via SHAP-style feature
  importances).

Run it with `streamlit run app/streamlit_app.py`.

---

## Tests & CI

`tests/` covers feature engineering and the prediction interface with
`pytest`. `.github/workflows/ci.yml` runs linting (`ruff`) and tests on
every push — swap in your own GitHub username to activate it once pushed.

```bash
pytest -q
```

---

## Roadmap / ways to extend for a stronger portfolio story

- [ ] Fine-tune detection on a real labeled dataset (Mosquito Alert / own photos)
- [ ] Add species classification head (Aedes aegypti vs albopictus vs Culex)
- [ ] Swap synthetic risk data for real DGHS/WHO case + Open-Meteo weather data
- [ ] Add a map view (folium/plotly) showing risk by neighborhood
- [ ] Containerize with Docker + deploy demo on Streamlit Community Cloud / HF Spaces
- [ ] Add model explainability (SHAP) panel to the risk tab

## License

MIT — see `LICENSE`.
