# 🌽 MaizeGuard: maize leaf disease detection with a retrainable ML pipeline

**Video demo (YouTube):** `ADD LINK`
**Live app:** `http://<EC2-PUBLIC-IP>:8501` · **API docs:** `http://<EC2-PUBLIC-IP>/docs`

## Project description

My Introduction to Machine Learning summative predicted which Rwandan farmers use improved seed. This project extends that use case to **image data**. Improved, disease-tolerant seed only pays off when farmers can spot disease early, so MaizeGuard classifies a photo of a maize leaf as **Blight, Common Rust, Gray Leaf Spot or Healthy** and gives short advice, including when to choose resistant seed next season.

It is a complete ML pipeline:

| Stage | Where |
|---|---|
| Data acquisition | Kaggle "Corn or Maize Leaf Disease Dataset" via `kagglehub`, split by `scripts/prepare_data.py` |
| Data processing | `src/preprocessing.py` (validate, decode, resize, augment, batch) |
| Model creation | `src/model.py` (MobileNetV2 transfer learning, dropout, L2, Adam, early stopping) |
| Model testing | `notebook/maizeguard.ipynb` (accuracy, loss, precision, recall, F1, confusion matrix) |
| Retraining | `src/model.py::retrain_model`, triggered by a button or automatically after 50 new uploads |
| API | `api/main.py` (FastAPI) |
| UI | `ui/app.py` (Streamlit: up-time, visualisations, predict, upload, retrain) |
| Storage | SQLite (`src/database.py`): uploaded images, retraining runs, prediction log |
| Deployment | Docker Compose on AWS EC2, nginx load balancer in front of N API containers |
| Load testing | `locust/locustfile.py` |

```
MaizeGuard/
├── README.md
├── notebook/maizeguard.ipynb
├── src/            preprocessing.py  model.py  prediction.py  database.py  insights.py
├── api/main.py
├── ui/app.py
├── data/           train/  test/
├── models/         maize_leaf_model.h5  metrics.json  data_insights.json
├── locust/locustfile.py
├── nginx/nginx.conf
├── scripts/prepare_data.py
├── Dockerfile  docker-compose.yml  requirements.txt
```

### How retraining works
1. **Upload** many images (or a zip with one folder per class) in the UI. Each file is validated, converted to RGB JPEG and **saved in the SQLite database**.
2. **Trigger**: press *Retrain model now*, or it starts automatically when 50 new images are waiting.
3. **Preprocess**: uploads are read back from the database and pass through the same pipeline as the training data.
4. **Retrain**: the current `maize_leaf_model.h5` is loaded as the **pre-trained model** and fine-tuned (small learning rate, early stopping), mixed with a sample of the original data.
5. **Evaluate in production**: test-set accuracy before and after is recorded. The new model replaces the old one only if it is not worse; every API container reloads it automatically.

## Set-up

### 1. Train the model (Google Colab, GPU)
Open `notebook/maizeguard.ipynb` in Colab, set your repo URL in the first cell and run all cells. Download `artifacts.zip` at the end and put the three files in `models/`. Also keep `data/train` and `data/test` in the repo.

Local alternative:
```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/prepare_data.py <path-to-downloaded-dataset>
python -m src.insights
python -m src.model --epochs 10
```

### 2. Run locally without Docker
```bash
uvicorn api.main:app --port 8000          # terminal 1  -> http://localhost:8000/docs
streamlit run ui/app.py                   # terminal 2  -> http://localhost:8501
```

### 3. Run with Docker
```bash
docker compose up -d --build              # UI :8501, API (through nginx) :80
docker compose up -d --scale api=3 && docker compose restart nginx    # 3 API containers
```

### 4. Deploy on AWS EC2
1. Launch an **Ubuntu 24.04** instance, type **t3.medium** (TensorFlow needs about 2 GB of RAM or more), 20 GB disk.
2. Security group inbound rules: 22 (SSH), 80 (API), 8501 (UI).
3. On the server:
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
git clone https://github.com/<your-username>/MaizeGuard.git && cd MaizeGuard
docker compose up -d --build
```
4. Open `http://<EC2-PUBLIC-IP>:8501`. Stop the instance when you are not demonstrating to avoid charges.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model up-time, version, answering container |
| POST | `/predict` | classify one image |
| POST | `/upload` | bulk upload for retraining (images + `label`, or zip) |
| POST | `/retrain` | trigger retraining |
| GET | `/retrain/status` | history of retraining runs with accuracy before/after |
| GET | `/metrics` | test-set evaluation of the production model |
| GET | `/stats` | predictions served, latency, pending uploads |
| GET | `/insights` | dataset features for the dashboard |

## Model results (test set)

Fill in from section 6 of the notebook.

| Accuracy | Loss | Precision | Recall | F1 |
|---|---|---|---|---|
| `...` | `...` | `...` | `...` | `...` |

## Results from the flood request simulation (Locust)

Run from your laptop against the EC2 server, once per container count:
```bash
pip install locust
docker compose up -d --scale api=1 && docker compose restart nginx      # on the server; then 2, then 4
locust -f locust/locustfile.py --host http://<EC2-PUBLIC-IP> --headless -u 100 -r 10 -t 2m --csv locust/results/containers_1
```
Settings: 100 users, spawn rate 10/s, 2 minutes, 5 `/predict` calls for every `/health` call.

| API containers | Requests | Failures | Median (ms) | Average (ms) | 95th pct (ms) | Requests/s |
|---|---|---|---|---|---|---|
| 1 | `...` | `...` | `...` | `...` | `...` | `...` |
| 2 | `...` | `...` | `...` | `...` | `...` | `...` |
| 4 | `...` | `...` | `...` | `...` | `...` | `...` |

Add Locust chart screenshots to `locust/results/` and two or three sentences on what changed as containers were added (and why gains flatten once all CPU cores are busy).

## Dataset credit
Corn or Maize Leaf Disease Dataset, Smaranjit Ghose, Kaggle (built from PlantVillage and PlantDoc).
