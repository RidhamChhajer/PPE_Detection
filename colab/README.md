# Google Colab Verification Guide

This directory contains the self-contained verification notebook for the **AI Based PPE Monitoring & Warning** project:

👉 **[`PPE_Detection_and_Tracking_Verification.ipynb`](./PPE_Detection_and_Tracking_Verification.ipynb)**

---

## What This Notebook Verifies

1. **Worker Detection & Real-time Tracking**: Uses `yolo11n.pt` with ByteTrack to track workers across video frames with unique, persistent IDs.
2. **PPE Association**: Pairs detected goggles, wrist accessories, and shoes from `upper.pt` / `lower.pt` with the person who is wearing them.
3. **Compliant vs. Non-Compliant Highlighting**:
   - 🟢 **GREEN box**: `COMPLIANT: Worker #<ID>`
   - 🔴 **RED box**: `VIOLATION: Worker #<ID> (<Reason>)`
4. **Single-Pass Violation Capture & Deduplication**:
   - Captures **only one photo** while a worker is in front of the camera (zero duplicate photos even over 50+ frames).
   - Once the worker walks out of the camera footage (absent for $\ge 3$ seconds), the active track is purged.
   - If that worker walks back in later on a **new pass**, their violation is captured again.
5. **Real-time Metrics**: Live count of workers in frame, total unique workers tracked, total violators flagged, and violation rate %.
6. **PDF Audit Report**: Compiles the official ReportLab PDF shift audit report with the **photos of the violators embedded directly into the report** and downloads it.

---

## Which Video Does It Check?

The notebook supports **two modes**:

### Mode 1: Built-in Automated Verification (Step 5A — Zero Uploads Required)
* Requires **no external video file**.
* Generates a realistic multi-frame factory video stream with workers moving in front of the camera, staying in view for multiple frames, leaving the camera footage, and returning later.
* Automatically verifies that:
  - 1 snapshot is captured on first violation.
  - 0 duplicate photos are saved while the worker stays in frame.
  - The worker is captured again when they return on a new pass.

### Mode 2: Test On Any Real Video File (Step 5B)
* You can test on any **real recorded `.mp4` video** (factory footage, test clip, or webcam recording).
* Upload your video using Colab's left sidebar (Files 📁), set `VIDEO_PATH = "your_video.mp4"`, and run Step 5B.
* It runs real YOLO11n person tracking and real `upper.pt` PPE inspection on every frame, generates `output_tracked.mp4`, saves snapshots, and embeds them into the PDF report!

---

## How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **Upload** (or File $\rightarrow$ Upload notebook) and select:
   `PPE_Detection/colab/PPE_Detection_and_Tracking_Verification.ipynb`
3. In Colab, go to **Runtime $\rightarrow$ Change runtime type** and select **T4 GPU** (free tier).
4. Run all cells from top to bottom (**Runtime $\rightarrow$ Run all** or `Ctrl + F9`).
5. All models (`upper.pt`, `lower.pt`, `yolo11n.pt`) will automatically download, the test stream will execute, the captured snapshots will display, and the generated PDF report will download automatically.
