# Helmet Detection and Violation Tracking

A Python computer-vision project that uses an Ultralytics YOLO model to detect riders wearing helmets and riders without helmets. It can process a still image, a video, or a live webcam feed. A separate tracking mode can count a rider once per track, save evidence, and optionally show or play a warning.

## Project files

| File | Purpose |
|---|---|
| `best.pt` | Trained model weights used by the detection and tracking scripts. |
| `detect.py` | Detects helmet status in an image, video, or webcam stream. |
| `track_violations.py` | Tracks detections, records suspected violations, saves evidence, and supports alerts. |
| `review_violations.py` | Lets a person confirm or correct each recorded violation. |
| `export_review_dataset.py` | Exports reviewed examples in YOLO format for later training. |
| `train_helmet_yolo26.ipynb` | Google Colab notebook for dataset preparation, model training, and evaluation. |
| `requirements.txt` | Python package requirements for the local scripts. |

The project expects the model classes to represent a helmeted rider and a rider without a helmet. The included model reports `Helmet` and `No-Helmet` classes.

## Setup on Windows

Install Python, open PowerShell in this folder, and create an environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The model file `best.pt` is included and should stay next to the Python scripts. If PowerShell blocks environment activation, you can install the requirements with `python -m pip install -r requirements.txt` and run the scripts with your chosen Python installation.

## Run detection

```powershell
# Image (saves the annotated result as output.jpg)
python detect.py --image photo.jpg

# Video (saves an annotated video as output.mp4)
python detect.py --video traffic.mp4 --save

# Webcam
python detect.py --webcam
```

Press **Q** to stop video or webcam mode. Useful options include `--weights best.pt`, `--conf 0.4`, `--imgsz 640`, and `--camera 0`. The image mode supports `--no-show` to save the result without opening a window.

## Track and log violations

Use tracking mode when you want a rider counted once per track and evidence saved:

```powershell
python track_violations.py --video traffic.mp4 --save
python track_violations.py --webcam --warn
python track_violations.py --webcam --warn --speak
```

A rider is counted after at least five no-helmet detections by default, provided at least 60% of that rider's detections are no-helmet. Adjust these thresholds with `--min-frames` and `--bad-ratio`. Other options include `--conf`, `--imgsz`, `--camera`, `--tracker`, and `--out`.

Each run creates a timestamped folder under `violations_output/` containing:

- `violations.csv` and `violations.json` with the recorded detections and timing information.
- `run_info.json` and `summary.txt` with run settings and totals.
- `snapshots/` with annotated frames, `raw/` with unannotated frames, and `crops/` with close-ups.
- `annotated.mp4` when `--save` is used.

The optional `--warn` flag beeps and displays an on-screen warning when a violation is confirmed. On Windows, `--speak` also speaks a warning. Alert timing is recorded. Tracking can lose a rider when they are obscured, which may cause a new track ID and a duplicate count.

## Review detections

Review the newest run with:

```powershell
python review_violations.py
```

Use these keys in the review window:

- **Y** — confirm the rider was without a helmet.
- **H** — correct the result: the rider was wearing a helmet.
- **N** — mark the detection as not a rider or not a head.
- **S** — skip; **B** — go back; **Q** — save and quit.

The decisions are saved to `review.csv` in that run's folder.

## Export corrections for retraining

After reviewing violations, export the reviewed examples:

```powershell
python export_review_dataset.py --zip
```

This creates `review_dataset/` and `review_dataset.zip`. Confirmed no-helmet cases are labeled as no-helmet, corrected helmet cases are labeled as helmet, and non-rider examples are exported without object labels as hard negatives.

To use these examples, upload `review_dataset.zip` to Colab as `/content/review_dataset.zip` and run the training notebook. The notebook adds reviewed examples to the training data. To fine-tune the current model, also upload `best.pt` to Colab and set `START_FROM = "/content/best.pt"` in the training cell. Retraining is a manual step; the local app does not train itself automatically.

## Train or evaluate a model

Open `train_helmet_yolo26.ipynb` in Google Colab, select a GPU runtime, and run the notebook cells in order. The notebook downloads or accepts a dataset, prepares YOLO labels, trains the model, evaluates it, and lets you download the resulting `best.pt`.

The training dataset is not included in this repository. The notebook's default source is the public Kaggle Helmet Detection dataset; it also documents Hugging Face, Roboflow, and YOLO-format ZIP options. Check the license and usage terms for whichever dataset you choose. Model results depend on its labels and how closely its images match the intended camera, lighting, distance, and helmet types.

## Limitations

- This is a detection aid, not a guarantee that every rider or helmet will be identified correctly.
- Small, distant, partially hidden riders and unfamiliar lighting or camera angles can reduce accuracy.
- A false detection can still be logged as a suspected violation; use human review before treating logs as ground truth.
- Tracking IDs may change after long occlusions, so one rider can occasionally be counted more than once.
- The README does not claim benchmark performance; evaluate the trained model on representative footage and report the notebook's measured metrics.

## Licensing

This repository includes a `LICENSE` file. Ultralytics and the selected dataset may have separate license terms; review those terms before redistributing or using the model commercially.
