"""Flood-request simulation.

  locust -f locust/locustfile.py --host http://<SERVER-IP>            (web UI on :8089)
  locust -f locust/locustfile.py --host http://<SERVER-IP> --headless \
         -u 100 -r 10 -t 2m --csv locust/results/containers_1
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

TEST_DIR = Path(__file__).resolve().parent.parent / "data" / "test"
IMAGES = [p for p in TEST_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:50]
if not IMAGES:
    raise SystemExit("Put some images in data/test first.")
PAYLOADS = [(p.name, p.read_bytes()) for p in IMAGES]


class FarmerUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def predict(self):
        name, data = random.choice(PAYLOADS)
        self.client.post("/predict", files={"file": (name, data, "image/jpeg")}, name="/predict")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
