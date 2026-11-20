"""MaizeGuard dashboard (Streamlit).   Run:  streamlit run ui/app.py"""
import os

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

st.set_page_config(page_title="MaizeGuard", page_icon="🌽", layout="wide")
st.title("🌽 MaizeGuard")
st.caption("Maize leaf disease detection for Rwandan farmers, with a retrainable model.")


def api(method, path, **kw):
    """Call the API. Returns (json, error_message)."""
    try:
        r = requests.request(method, API + path, timeout=kw.pop("timeout", 60), **kw)
        if r.ok:
            return r.json(), None
        try:
            return None, r.json().get("detail", r.text)
        except ValueError:
            return None, r.text
    except requests.RequestException as exc:
        return None, f"API not reachable: {exc}"


def fmt_uptime(sec):
    d, rem = divmod(int(sec), 86400)
    h, rem = divmod(rem, 3600)
    return f"{d}d {h}h {rem // 60}m {rem % 60}s"


tab_status, tab_predict, tab_viz, tab_retrain = st.tabs(
    ["📡 Status", "🔍 Predict", "📊 Data insights", "🔁 Upload & retrain"])

# --------------------------------------------------------------------- status
with tab_status:
    st.button("Refresh", key="refresh_status")
    health, err = api("GET", "/health", timeout=10)
    if err:
        st.error(f"🔴 Model is DOWN. {err}")
    else:
        ok = health["status"] == "ok"
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model status", "🟢 Online" if ok else "🟠 No model file")
        c2.metric("Up-time", fmt_uptime(health["uptime_seconds"]))
        c3.metric("Model version", health["model_version"] or "none")
        c4.metric("Answered by container", health["container"])
        st.caption(f"Running since {health['started_at']}. With several containers, "
                   "refresh to see different containers answer.")

    stats, _ = api("GET", "/stats", timeout=10)
    if stats:
        p = stats["predictions"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions served", p["total"])
        c2.metric("Average model latency", f"{p['avg_latency_ms']:.0f} ms" if p["avg_latency_ms"] else "n/a")
        c3.metric("Average confidence", f"{p['avg_confidence']:.1%}" if p["avg_confidence"] else "n/a")
        if p["by_class"]:
            st.subheader("What the model has been predicting in production")
            st.bar_chart(pd.Series({k: v["count"] for k, v in p["by_class"].items()}, name="predictions"))

    m, _ = api("GET", "/metrics", timeout=10)
    if m and m.get("test"):
        t = m["test"]
        st.subheader("Evaluation of the model in production (held-out test set)")
        st.caption(f"{m['note']} · {m['trained_at']} · {t['n_images']} test images")
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "loss"]):
            col.metric(key.capitalize(), f"{t[key]:.3f}")
        st.write("Confusion matrix (rows = true class, columns = predicted class)")
        st.dataframe(pd.DataFrame(t["confusion_matrix"], index=t["class_names"], columns=t["class_names"]))

# -------------------------------------------------------------------- predict
with tab_predict:
    st.subheader("Check one maize leaf")
    img = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"], key="predict_file")
    if img:
        left, right = st.columns(2)
        left.image(img, caption=img.name, use_container_width=True)
        if right.button("Predict", type="primary"):
            with st.spinner("Analysing..."):
                res, err = api("POST", "/predict", files={"file": (img.name, img.getvalue(), img.type)})
            if err:
                right.error(err)
            else:
                right.success(f"Prediction: **{res['label'].replace('_', ' ')}** "
                              f"({res['confidence']:.1%} confident)")
                right.info(res["advice"])
                right.bar_chart(pd.Series(res["probabilities"], name="probability"))
                right.caption(f"Model time {res['latency_ms']} ms · container {res['container']}")

# ------------------------------------------------------------------- insights
with tab_viz:
    ins, err = api("GET", "/insights", timeout=10)
    if err:
        st.warning(err)
    else:
        counts = pd.Series(ins["class_counts"], name="images")
        feats = pd.DataFrame(ins["features"]).T

        st.subheader("1. Class balance: how many images per class?")
        st.bar_chart(counts)
        small, big = counts.idxmin(), counts.idxmax()
        st.markdown(
            f"**Story:** `{small}` has only {counts.min()} training images against {counts.max()} for "
            f"`{big}` ({counts.max() / max(counts.min(), 1):.1f}x more). The model sees the rare class "
            "less often, so that is where recall is most at risk. This is why we report per-class "
            "precision/recall and macro F1 instead of accuracy alone, and why new uploads of the rare "
            "class are the most valuable for retraining.")

        if not feats.empty:
            st.subheader("2. Colour: how green is the leaf?")
            st.bar_chart(feats["green_ratio"])
            g_hi, g_lo = feats["green_ratio"].idxmax(), feats["green_ratio"].idxmin()
            st.markdown(
                f"**Story:** `{g_hi}` leaves have the highest share of green ({feats.loc[g_hi, 'green_ratio']:.1%}) "
                f"and `{g_lo}` the lowest ({feats.loc[g_lo, 'green_ratio']:.1%}). Disease replaces green "
                "chlorophyll with brown, grey or rust-coloured tissue, so loss of green is a direct signal "
                "of infection that the model can learn from colour alone.")
            st.bar_chart(feats[["red", "green", "blue"]])

            st.subheader("3. Brightness and contrast: how patchy is the leaf surface?")
            st.bar_chart(feats[["brightness", "contrast"]])
            c_hi, c_lo = feats["contrast"].idxmax(), feats["contrast"].idxmin()
            st.markdown(
                f"**Story:** `{c_hi}` images have the highest contrast ({feats.loc[c_hi, 'contrast']:.1f}) and "
                f"`{c_lo}` the lowest ({feats.loc[c_lo, 'contrast']:.1f}). Contrast measures how much the "
                "grey level varies inside one photo: lesions and pustules create light/dark patches, "
                "while a clean leaf is more uniform. Classes with similar colour AND similar contrast are "
                "the ones the model confuses most, which you can check in the confusion matrix on the "
                "Status tab.")

    m, _ = api("GET", "/metrics", timeout=10)
    if m and m.get("history", {}).get("loss"):
        st.subheader("Training curves of the current model")
        h = pd.DataFrame(m["history"])
        c1, c2 = st.columns(2)
        c1.line_chart(h[[c for c in ("loss", "val_loss") if c in h]])
        c2.line_chart(h[[c for c in ("accuracy", "val_accuracy") if c in h]])

# -------------------------------------------------------------------- retrain
with tab_retrain:
    st.subheader("Step 1. Upload new labelled images")
    st.caption("Select many images of ONE class, or upload a .zip with one folder per class "
               f"({', '.join(CLASSES)}).")
    label = st.selectbox("Class of these images", CLASSES)
    new_files = st.file_uploader("Images or a zip file", type=["jpg", "jpeg", "png", "zip"],
                                 accept_multiple_files=True, key="bulk_files")
    if st.button("Upload to database", disabled=not new_files):
        with st.spinner("Validating and saving..."):
            res, err = api("POST", "/upload", data={"label": label}, timeout=300,
                           files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                                  for f in new_files])
        if err:
            st.error(err)
        else:
            st.success(f"Saved {res['saved']} images to the database.")
            if res["rejected"]:
                st.warning(f"Rejected {len(res['rejected'])} files")
                st.dataframe(pd.DataFrame(res["rejected"]))
            if res["auto_retrain_started"]:
                st.info("Upload threshold reached: retraining started automatically.")

    stats, _ = api("GET", "/stats", timeout=10)
    if stats:
        pend = stats["pending_uploads"]
        st.write(f"**Images waiting for retraining: {sum(pend.values())}** "
                 f"(automatic retraining starts at {stats['retrain_threshold']})")
        st.bar_chart(pd.Series(pend, name="waiting"))

    st.subheader("Step 2. Trigger retraining")
    if st.button("🔁 Retrain model now", type="primary"):
        res, err = api("POST", "/retrain")
        if err:
            st.error(err)
        else:
            st.success(f"Retraining run #{res['run_id']} started on {res['images']} new images.")

    st.subheader("Step 3. Follow the run")
    st.button("Refresh status", key="refresh_runs")
    runs, _ = api("GET", "/retrain/status", timeout=10)
    if runs and runs["runs"]:
        df = pd.DataFrame(runs["runs"])
        if df.iloc[0]["status"] == "running":
            st.info("⏳ Retraining is running. Predictions keep working with the current model.")
        st.dataframe(df[["id", "status", "triggered_by", "started_at", "finished_at", "n_images",
                         "accuracy_before", "accuracy_after", "promoted", "message"]])
    else:
        st.caption("No retraining runs yet.")
