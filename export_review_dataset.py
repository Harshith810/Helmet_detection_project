"""
Turn reviewed violations into retraining data (YOLO format).

For every reviewed violation, a close-up crop is made from the clean (undrawn) frame:
    confirmed       -> labelled with the No-Helmet class
    wearing_helmet  -> labelled with the Helmet class (the model's mistake, corrected)
    not_a_rider     -> no label at all (a hard negative: the model must learn to ignore it)
Skipped / unreviewed violations are ignored.

    python export_review_dataset.py                       # all runs in violations_output
    python export_review_dataset.py --runs violations_output/webcam_2026* --zip

Output:  review_dataset/{images,labels}  (+ classes.json, and review_dataset.zip with --zip)
Upload the zip to Colab and set REVIEW_ZIP in the training notebook (Step 3).
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys

import cv2


def is_violation(class_name: str) -> bool:
    n = class_name.lower()
    return "without" in n or n.startswith("no")


def read_decisions(run_dir):
    path = os.path.join(run_dir, "review.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {int(r["violation_no"]): r["decision"] for r in csv.DictReader(f)}


def class_ids(run_dir):
    """Return (names_in_order, bad_id, good_id) from run_info.json."""
    info = json.load(open(os.path.join(run_dir, "run_info.json")))
    classes = {int(k): v for k, v in info["classes"].items()}
    names = [classes[i] for i in sorted(classes)]
    bad = next((i for i, n in enumerate(names) if is_violation(n)), None)
    good = next((i for i, n in enumerate(names) if not is_violation(n)), None)
    return names, bad, good


def make_example(raw, box, pad=0.2):
    """Crop around the box with padding. Returns (crop, (cx, cy, w, h) normalised box in the crop)."""
    h, w = raw.shape[:2]
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
    cx2, cy2 = min(w, x2 + px), min(h, y2 + py)
    crop = raw[cy1:cy2, cx1:cx2]
    ch, cw = crop.shape[:2]
    if cw < 20 or ch < 20:
        return None, None
    bx1, by1, bx2, by2 = x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1
    return crop, ((bx1 + bx2) / 2 / cw, (by1 + by2) / 2 / ch, (bx2 - bx1) / cw, (by2 - by1) / ch)


def main():
    p = argparse.ArgumentParser(description="Export reviewed violations as a YOLO training set")
    p.add_argument("--runs", nargs="+", default=["violations_output/*"],
                   help="run folders (wildcards allowed)")
    p.add_argument("--out", default="review_dataset")
    p.add_argument("--zip", action="store_true", help="also create <out>.zip for Colab")
    args = p.parse_args()

    runs = sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)})
    runs = [r for r in runs if os.path.exists(os.path.join(r, "violations.json"))]
    if not runs:
        sys.exit("No run folders with violations.json found.")

    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(os.path.join(args.out, "images"))
    os.makedirs(os.path.join(args.out, "labels"))

    stats = {"confirmed": 0, "wearing_helmet": 0, "not_a_rider": 0, "skipped": 0}
    names0 = None
    for run in runs:
        decisions = read_decisions(run)
        if not decisions:
            print(f"  {os.path.basename(run)}: not reviewed yet - skipped")
            continue
        names, bad_id, good_id = class_ids(run)
        names0 = names0 or names
        for rec in json.load(open(os.path.join(run, "violations.json"))):
            d = decisions.get(rec["violation_no"])
            if d not in stats or d == "skipped":
                stats["skipped"] += 1
                continue
            raw = cv2.imread(os.path.join(run, rec["raw"]))
            if raw is None:
                print(f"  missing raw image for violation {rec['violation_no']} in {run}")
                continue
            crop, box = make_example(raw, rec["box"])
            if crop is None:
                continue
            stem = f"{os.path.basename(run)}_v{rec['violation_no']:03d}"
            cv2.imwrite(os.path.join(args.out, "images", stem + ".jpg"), crop)
            label = ""
            if d == "confirmed" and bad_id is not None:
                label = f"{bad_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
            elif d == "wearing_helmet" and good_id is not None:
                label = f"{good_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n"
            open(os.path.join(args.out, "labels", stem + ".txt"), "w").write(label)
            stats[d] += 1

    if names0:
        json.dump({"names": names0}, open(os.path.join(args.out, "classes.json"), "w"))
    total = stats["confirmed"] + stats["wearing_helmet"] + stats["not_a_rider"]
    print(f"\nExported {total} example(s) to {args.out}/")
    print(f"  confirmed No-Helmet      : {stats['confirmed']}")
    print(f"  corrected to Helmet      : {stats['wearing_helmet']}")
    print(f"  hard negatives (no label): {stats['not_a_rider']}")
    print(f"  skipped / unreviewed     : {stats['skipped']}")
    if args.zip and total:
        shutil.make_archive(args.out, "zip", args.out)
        print(f"Created {args.out}.zip - upload it to Colab.")
    if total == 0:
        print("Nothing exported - review some violations first (python review_violations.py).")


if __name__ == "__main__":
    main()
