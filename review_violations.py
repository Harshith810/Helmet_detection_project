"""
Human review of the violations found by track_violations.py.

A person looks at every logged violation and decides whether the system was right.
The decisions are saved in review.csv inside the run folder, and later turned into
retraining data by export_review_dataset.py.

    python review_violations.py                       # newest run in violations_output
    python review_violations.py --run violations_output/webcam_20261113_101500

Keys (click the window first):
    Y   confirmed - this rider really has NO helmet
    H   wrong - the rider IS wearing a helmet
    N   wrong - not a rider / not a head (for example a bag, a pedestrian)
    S   skip for now          B   go back one          Q   save and quit
"""
import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np

DECISION_KEYS = {"y": "confirmed", "h": "wearing_helmet", "n": "not_a_rider"}
DECISION_TEXT = {
    "confirmed": "CONFIRMED: no helmet",
    "wearing_helmet": "WRONG: wearing a helmet",
    "not_a_rider": "WRONG: not a rider / not a head",
}
DECISION_COLOR = {"confirmed": (0, 200, 0), "wearing_helmet": (0, 140, 255),
                  "not_a_rider": (0, 140, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------ files
def newest_run(base="violations_output"):
    runs = [d for d in glob.glob(os.path.join(base, "*"))
            if os.path.isfile(os.path.join(d, "violations.json"))]
    return max(runs, key=os.path.getmtime) if runs else None


def load_records(run_dir):
    path = os.path.join(run_dir, "violations.json")
    if not os.path.exists(path):
        sys.exit(f"No violations.json in {run_dir}. Run track_violations.py first "
                 "(this review tool needs the newer version of it).")
    return json.load(open(path))


def load_decisions(run_dir):
    path = os.path.join(run_dir, "review.csv")
    decisions = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                decisions[int(row["violation_no"])] = row["decision"]
    return decisions


def save_decisions(run_dir, decisions):
    with open(os.path.join(run_dir, "review.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["violation_no", "decision", "reviewed_at"])
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for no in sorted(decisions):
            w.writerow([no, decisions[no], stamp])


# ------------------------------------------------------------ logic (no window needed)
class ReviewSession:
    def __init__(self, records, decisions=None):
        self.records = records
        self.decisions = dict(decisions or {})
        self.idx = next((i for i, r in enumerate(records)
                         if r["violation_no"] not in self.decisions), len(records))

    @property
    def done(self):
        return self.idx >= len(self.records)

    def current(self):
        return self.records[self.idx]

    def decide(self, decision):
        self.decisions[self.current()["violation_no"]] = decision
        self.idx += 1

    def skip(self):
        self.idx += 1

    def back(self):
        self.idx = max(0, self.idx - 1)

    def counts(self):
        c = {"confirmed": 0, "wearing_helmet": 0, "not_a_rider": 0}
        for d in self.decisions.values():
            if d in c:
                c[d] += 1
        c["pending"] = len(self.records) - sum(c.values())
        return c


# ------------------------------------------------------------ drawing
def fit(img, w, h):
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    return cv2.resize(img, (max(1, int(iw * s)), max(1, int(ih * s))), interpolation=cv2.INTER_AREA)


def paste(canvas, img, x, y):
    h, w = img.shape[:2]
    canvas[y:y + h, x:x + w] = img


def render(run_dir, rec, idx, total, decision, counts):
    canvas = np.full((720, 1280, 3), 32, np.uint8)
    cv2.putText(canvas, f"Violation {idx + 1} of {total}   |   rider id {rec['track_id']}   |   "
                f"video time {rec['video_time_s']}s   |   confidence {rec['avg_confidence']}",
                (20, 36), FONT, 0.7, (255, 255, 255), 2)

    for key, (x, y, w, h, title) in {"snapshot": (20, 70, 760, 520, "Snapshot (as logged)"),
                                     "crop": (800, 70, 460, 360, "Close-up")}.items():
        cv2.putText(canvas, title, (x, y - 8), FONT, 0.5, (170, 170, 170), 1)
        rel = rec.get(key)
        img = cv2.imread(os.path.join(run_dir, rel)) if rel else None
        if img is None:
            cv2.putText(canvas, "(image missing)", (x + 10, y + 40), FONT, 0.6, (0, 0, 255), 2)
        else:
            paste(canvas, fit(img, w, h), x, y)

    if decision:
        cv2.putText(canvas, f"Your decision: {DECISION_TEXT.get(decision, decision)}",
                    (800, 470), FONT, 0.65, DECISION_COLOR.get(decision, (255, 255, 255)), 2)
    else:
        cv2.putText(canvas, "Not reviewed yet", (800, 470), FONT, 0.65, (0, 200, 255), 2)

    lines = ["[Y]  Confirmed - rider has NO helmet", "[H]  Wrong - rider IS wearing a helmet",
             "[N]  Wrong - not a rider / not a head", "[S]  Skip      [B]  Back      [Q]  Save and quit"]
    for i, t in enumerate(lines):
        cv2.putText(canvas, t, (20, 625 + i * 26), FONT, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, f"Confirmed {counts['confirmed']}   |   Wearing helmet {counts['wearing_helmet']}",
                (800, 515), FONT, 0.55, (200, 200, 200), 1)
    cv2.putText(canvas, f"Not a rider {counts['not_a_rider']}   |   Still pending {counts['pending']}",
                (800, 540), FONT, 0.55, (200, 200, 200), 1)
    return canvas


# ------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description="Review logged helmet violations")
    p.add_argument("--run", help="run folder inside violations_output (default: newest)")
    p.add_argument("--base", default="violations_output")
    args = p.parse_args()

    run_dir = args.run or newest_run(args.base)
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit("No run folder found. Run track_violations.py first, or pass --run <folder>.")
    records = load_records(run_dir)
    if not records:
        sys.exit("This run has no violations to review.")

    session = ReviewSession(records, load_decisions(run_dir))
    print(f"Reviewing {len(records)} violation(s) from {run_dir}")
    win = "Review violations"
    while not session.done:
        rec = session.current()
        cv2.imshow(win, render(run_dir, rec, session.idx, len(records),
                               session.decisions.get(rec["violation_no"]), session.counts()))
        k = cv2.waitKey(0) & 0xFF
        ch = chr(k).lower() if k < 128 else ""
        if ch in DECISION_KEYS:
            session.decide(DECISION_KEYS[ch])
            save_decisions(run_dir, session.decisions)
        elif ch == "s":
            session.skip()
        elif ch == "b":
            session.back()
        elif ch == "q":
            break
    cv2.destroyAllWindows()
    save_decisions(run_dir, session.decisions)

    c = session.counts()
    print(f"\nConfirmed violations : {c['confirmed']}\nWrong (had helmet)   : {c['wearing_helmet']}\n"
          f"Wrong (not a rider)  : {c['not_a_rider']}\nStill pending        : {c['pending']}")
    reviewed = c["confirmed"] + c["wearing_helmet"] + c["not_a_rider"]
    if reviewed:
        print(f"False-alarm rate so far: {100 * (reviewed - c['confirmed']) / reviewed:.0f}% of {reviewed} reviewed")
    print(f"Saved: {os.path.join(run_dir, 'review.csv')}")


if __name__ == "__main__":
    main()
