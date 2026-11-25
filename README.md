# MaizeGuard: maize leaf disease detection with a retrainable ML pipeline

| | |
|---|---|
| **Video demo (YouTube)** | `<ADD YOUTUBE LINK HERE>` |
| **GitHub repository** | https://github.com/itmuvunyi/maizeguard |
| **Live app (UI)** | http://13.63.166.158:8501 |
| **API docs (Swagger)** | http://13.63.166.158/docs |

> The live app runs on an AWS EC2 server (region eu-north-1). The server is stopped when it is not being demonstrated to avoid charges, and its public IP changes after a restart. The video shows the full system running.

## Project description

My Introduction to Machine Learning summative predicted which Rwandan farmers use improved seed. This project extends that use case to **image data**. Improved, disease-tolerant seed only pays off when farmers can spot disease early, so MaizeGuard classifies a photo of a maize leaf as **Blight, Common Rust, Gray Leaf Spot or Healthy** and gives short advice, including when to choose resistant seed next season.

### Repository structure

```
maizeguard/
├── README.md
├── notebook/maizeguard.ipynb
├── src/            preprocessing.py  model.py  prediction.py  database.py  insights.py
├── api/main.py
├── ui/app.py
├── data/           train/  test/
├── models/         maize_leaf_model.h5  metrics.json  data_insights.json
├── locust/         locustfile.py  results/
├── nginx/nginx.conf
├── scripts/prepare_data.py
├── Dockerfile  docker-compose.yml  requirements.txt  requirements-dev.txt
```

## How to use the app

1. **Status tab**: model up-time, version, the container that answered, and the evaluation metrics of the model in production.
2. **Data insights tab**: class balance, colour and brightness/contrast charts with interpretations.
3. **Predict tab**: upload one leaf photo and get the disease, the confidence and advice.
4. **Upload & retrain tab**: pick a class, upload many images (saved to the database), then press *Retrain model now* and watch the run history (accuracy before and after).

## Set-up (step by step)

### 1. Get the code and the data
```bash
git clone https://github.com/itmuvunyi/maizeguard.git
cd maizeguard
```
The repository already contains the trained model in `models/` and the images in `data/train` and `data/test`.

To rebuild the dataset yourself, download the Kaggle dataset `smaranjitghose/corn-or-maize-leaf-disease-dataset` **outside** the project folder and split it:
```bash
python scripts/prepare_data.py <path-to-downloaded-dataset>
```
This keeps the four class folders (`Blight`, `Common_Rust`, `Gray_Leaf_Spot`, `Healthy`), drops corrupt files and makes a fixed 80/20 train/test split.

### 2. Train the model (optional, the trained model is included)
* **Google Colab (recommended, free GPU):** open `notebook/maizeguard.ipynb` in Colab, set the runtime to GPU and run all cells. Download `artifacts.zip` at the end and unzip the three files into `models/`.
* **Locally on CPU:**
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt -r requirements-dev.txt
python -m src.model --epochs 10
python -m src.insights
```
Use Python 3.11 (3.12 also works). Run every command from the project root.

### 3. Run locally without Docker
```bash
uvicorn api.main:app --port 8000          # terminal 1  -> http://localhost:8000/docs
streamlit run ui/app.py                   # terminal 2  -> http://localhost:8501
```
Start the API first, otherwise the UI shows "Model is DOWN".

### 4. Run with Docker
```bash
docker compose build
docker compose up -d                      # UI on :8501, API (through nginx) on :80
docker compose up -d --scale api=3 && docker compose restart nginx    # 3 API containers
```
Always restart nginx after changing the number of API containers.

### 5. Deploy on AWS EC2
1. In the EC2 console launch an **Ubuntu 24.04** instance with at least 4 GB of RAM and 20 GB of disk (this project ran on an instance with 2 vCPUs and 7.6 GiB of RAM in `eu-north-1`).
2. Security group inbound rules: **22** (SSH), **80** (API), **8501** (UI).
3. Open a terminal on the server (EC2 Instance Connect) and run:
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
git clone https://github.com/itmuvunyi/maizeguard.git && cd maizeguard
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
docker compose build
docker compose up -d
docker compose ps
```
The 4 GB swap file protects the server from running out of memory when several TensorFlow containers run at once.
4. Open `http://<EC2-PUBLIC-IP>:8501` (UI) and `http://<EC2-PUBLIC-IP>/docs` (API). Stop the instance when you are not demonstrating to avoid charges.

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

## Model results

Evaluation on the held-out test set (836 images the model never saw during training), from section 6 of the notebook. Precision, recall and F1 are macro averages, so every class counts equally.

| Accuracy | Loss | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.9222 | 0.2112 | 0.9006 | 0.8996 | 0.8999 |

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Blight | 0.849 | 0.886 | 0.868 | 229 |
| Common_Rust | 0.992 | 0.962 | 0.977 | 261 |
| Gray_Leaf_Spot | 0.761 | 0.754 | 0.758 | 114 |
| Healthy | 1.000 | 0.996 | 0.998 | 232 |

## Results from the flood request simulation (Locust)

I simulated a flood of prediction requests against the deployed API on AWS EC2 (2 vCPUs, 7.6 GiB RAM) through nginx, with **1, 2, 3 and 4 API containers**. Every run used the same load, so the results can be compared.

Settings: 20 simulated users, 2 new users per second, 2 minutes per run, 5 `/predict` calls for every `/health` call (each `/predict` sends a random leaf image from `data/test`).

On the server I set the number of containers, then ran Locust from locally:
```bash
docker compose up -d --scale api=1 && docker compose restart nginx      # on the server (then 2, 3, 4)
locust -f locust/locustfile.py --host http://<EC2-PUBLIC-IP> --headless -u 20 -r 2 -t 2m --csv locust/results/containers_1
```

| API containers | Total requests | Failures | Requests/s | `/predict` median (ms) | `/predict` average (ms) | `/predict` 95th pct (ms) |
|---|---|---|---|---|---|---|
| 1 | 834 | 0 | 6.98 | 1400 | 1720 | 4200 |
| 2 | 788 | 0 | 6.60 | 700 | 1542 | 4200 |
| 3 | 979 | 0 | 8.19 | 780 | 1221 | 3400 |
| 4 | 1020 | 0 | 8.54 | 810 | 1173 | 3100 |

The raw results are in `locust/results/` (`containers_N_stats.csv`, one set per container count).

## Dataset credit
Corn or Maize Leaf Disease Dataset, Smaranjit Ghose, Kaggle (built from PlantVillage and PlantDoc).
