"""
Helmet violation tracking + logging + real-time warning (YOLO26 + Ultralytics tracker).

Every rider gets a track ID, and a rider is counted as a violator ONCE
(not once per frame). For each violation it saves evidence and, with --warn,
raises an alert on the spot (beep + on-screen message, optional spoken warning).

Run from the project folder (best.pt next to this file):

    python track_violations.py --video traffic.mp4 --save
    python track_violations.py --webcam --warn            # live, with warnings
    python track_violations.py --webcam --warn --speak    # also says 'please wear a helmet' (Windows)

Results go to  violations_output/<name>_<date-time>/ :
    violations.csv     one row per violator (for people)
    violations.json    the same violations with box + file details (used by the review tools)
    run_info.json      model classes and settings of the run
    snapshots/         full frame (with boxes) when the violation was confirmed
    raw/               the same frame WITHOUT drawings (used for retraining)
    crops/             clean close-up of the violator
    summary.txt        totals for the run
    annotated.mp4      only if you use --save

Afterwards:  python review_violations.py   (approve / reject each violation)
             python export_review_dataset.py   (turn the reviews into retraining data)

Options:
    --weights best.pt   trained model
    --conf 0.4          minimum detection confidence
    --min-frames 5      a rider must be seen this many frames (as No-Helmet) before it counts
    --bad-ratio 0.6     share of a rider's frames that must say No-Helmet
    --warn              beep + on-screen warning when a violation is confirmed
    --warn-seconds 3    how long the on-screen warning stays
    --speak             spoken warning (Windows only; needs --warn)
    --tracker X.yaml    optional tracker config (default: Ultralytics default)
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2

GREEN = (0, 200, 0)   # BGR
RED = (0, 0, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def is_violation(class_name: str) -> bool:
    """True for classes like 'Without Helmet' / 'No-Helmet' / 'no_helmet'."""
    n = class_name.lower()
    return "without" in n or n.startswith("no")


class TrackStats:
    __slots__ = ("first_frame", "last_frame", "n_bad", "n_ok", "conf_bad", "logged", "first_bad_t")

    def __init__(self, frame_idx):
        self.first_frame = self.last_frame = frame_idx
        self.n_bad = self.n_ok = 0
        self.conf_bad = 0.0
        self.logged = False
        self.first_bad_t = None


class ViolationCounter:
    """Decides, per track ID, when someone counts as a violator (once)."""

    def __init__(self, min_frames=5, ratio=0.6):
        self.min_frames = min_frames
        self.ratio = ratio          # share of frames that must agree (flicker protection)
        self.tracks = {}
        self.n_violations = 0

    def update(self, frame_idx, detections, t=None):
        """detections: list of (track_id, is_bad, conf, box). t = time in seconds (optional).
        Returns the new violation events."""
        events = []
        now = frame_idx if t is None else t
        for tid, bad, conf, box in detections:
            st = self.tracks.get(tid)
            if st is None:
                st = self.tracks[tid] = TrackStats(frame_idx)
            st.last_frame = frame_idx
            if bad:
                if st.n_bad == 0:
                    st.first_bad_t = now
                st.n_bad += 1
                st.conf_bad += conf
            else:
                st.n_ok += 1
            # clear majority + minimum number of frames = stable decision
            total = st.n_bad + st.n_ok
            if not st.logged and st.n_bad >= self.min_frames and st.n_bad / total >= self.ratio:
                st.logged = True
                self.n_violations += 1
                events.append({
                    "violation_no": self.n_violations,
                    "track_id": tid,
                    "frame": frame_idx,
                    "frames_seen": total,
                    "avg_conf": st.conf_bad / st.n_bad,
                    "box": box,
                    "first_bad_t": st.first_bad_t,
                })
        return events

    def helmet_riders(self):
        """Tracks that are (so far) confidently wearing a helmet."""
        return sum(1 for s in self.tracks.values()
                   if not s.logged and s.n_ok >= self.min_frames
                   and s.n_ok / (s.n_ok + s.n_bad) >= self.ratio)

    def is_logged(self, tid):
        st = self.tracks.get(tid)
        return bool(st and st.logged)


def get_tracked_detections(result, names):
    """From one Ultralytics result -> list of (track_id or None, is_bad, conf, (x1,y1,x2,y2), name)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id
    ids = ids.cpu().numpy().astype(int).tolist() if ids is not None else [None] * len(cls)
    out = []
    for b, c, p, i in zip(xyxy, cls, conf, ids):
        name = names[int(c)]
        out.append((i, is_violation(name), float(p),
                    tuple(int(v) for v in b), name))
    return out


def draw(frame, dets, counter):
    for tid, bad, conf, (x1, y1, x2, y2), name in dets:
        confirmed = tid is not None and counter.is_logged(tid)
        color = RED if bad else GREEN
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if confirmed else 2)
        tag = f"#{tid} " if tid is not None else ""
        label = f"{tag}{name} {conf:.2f}" + ("  VIOLATION" if confirmed else "")
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 2)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2), FONT, 0.55, (255, 255, 255), 2)

    banner = f"Violations: {counter.n_violations}   Helmet riders: {counter.helmet_riders()}"
    cv2.rectangle(frame, (0, 0), (380, 30), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 21), FONT, 0.6, (255, 255, 255), 2)
    return frame


# ---------------------------------------------------------------- real-time warning
def draw_alert(frame, text):
    """Red warning bar across the bottom of the frame."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 255), -1)
    cv2.putText(frame, text, (15, h - 20), FONT, 0.8, (255, 255, 255), 2)
    return frame


def play_beep():
    """Three short beeps in a background thread (Windows). Elsewhere: terminal bell."""
    def _run():
        try:
            import winsound
            for freq in (1000, 1400, 1000):
                winsound.Beep(freq, 180)
        except ImportError:
            print("\a", end="", flush=True)
    threading.Thread(target=_run, daemon=True).start()


def speak(text):
    """Spoken warning using the built-in Windows voice. Does nothing on other systems."""
    if platform.system() != "Windows":
        return
    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command",
             "Add-Type -AssemblyName System.Speech; "
             f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception:
        pass


# ---------------------------------------------------------------- saving evidence
def save_event(event, raw, annotated, out_dir, writer, source, t_sec, delay_s,
               records, bad_name, fps):
    n = event["violation_no"]
    h, w = raw.shape[:2]
    x1, y1, x2, y2 = event["box"]
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    pad_x, pad_y = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
    crop = raw[max(y1 - pad_y, 0):min(y2 + pad_y, h), max(x1 - pad_x, 0):min(x2 + pad_x, w)]

    fname = f"violation_{n:03d}_id{event['track_id']}.jpg"
    cv2.imwrite(os.path.join(out_dir, "snapshots", fname), annotated)
    cv2.imwrite(os.path.join(out_dir, "raw", fname), raw)
    if crop.size:
        cv2.imwrite(os.path.join(out_dir, "crops", fname), crop)

    logged_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    writer.writerow([n, event["track_id"], f"{t_sec:.2f}", f"{delay_s:.2f}", logged_at,
                     source, event["frames_seen"], f"{event['avg_conf']:.3f}",
                     f"snapshots/{fname}", f"crops/{fname}"])
    records.append({
        "violation_no": n, "track_id": event["track_id"], "frame": event["frame"],
        "video_time_s": round(t_sec, 2), "alert_delay_s": round(delay_s, 2),
        "logged_at": logged_at, "predicted_class": bad_name,
        "avg_confidence": round(event["avg_conf"], 3), "frames_seen": event["frames_seen"],
        "box": [x1, y1, x2, y2], "frame_size": [w, h],
        "snapshot": f"snapshots/{fname}", "raw": f"raw/{fname}",
        "crop": f"crops/{fname}" if crop.size else None,
    })
    with open(os.path.join(out_dir, "violations.json"), "w") as f:
        json.dump(records, f, indent=1)


def main():
    p = argparse.ArgumentParser(description="Helmet violation tracking, logging and warning (YOLO26)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--webcam", action="store_true", help="use the webcam")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--min-frames", type=int, default=5,
                   help="frames a rider must be seen as No-Helmet before counting")
    p.add_argument("--bad-ratio", type=float, default=0.6,
                   help="share of frames that must say No-Helmet to count a violation")
    p.add_argument("--warn", action="store_true", help="beep + on-screen warning on each violation")
    p.add_argument("--warn-seconds", type=float, default=3.0, help="how long the warning stays on screen")
    p.add_argument("--speak", action="store_true", help="spoken warning (Windows; needs --warn)")
    p.add_argument("--tracker", default=None, help="optional tracker yaml, e.g. bytetrack.yaml")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--save", action="store_true", help="save the annotated video")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    p.add_argument("--out", default="violations_output", help="output folder")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"Model file not found: {args.weights}\nPut best.pt in this folder.")
    if args.video and not os.path.exists(args.video):
        sys.exit(f"Video not found: {args.video}")

    from ultralytics import YOLO  # imported here so --help is instant
    model = YOLO(args.weights)
    names = model.names
    print("Model classes:", names)
    bad_names = [n for n in names.values() if is_violation(n)]
    bad_name = bad_names[0] if bad_names else "No-Helmet"

    source_name = os.path.splitext(os.path.basename(args.video))[0] if args.video else "webcam"
    run_dir = os.path.join(args.out, f"{source_name}_{datetime.now():%Y%m%d_%H%M%S}")
    for sub in ("snapshots", "raw", "crops"):
        os.makedirs(os.path.join(run_dir, sub))
    print("Saving results to:", run_dir)

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("Could not open the video source.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    vw = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(os.path.join(run_dir, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    with open(os.path.join(run_dir, "run_info.json"), "w") as f:
        json.dump({"classes": {str(k): v for k, v in names.items()}, "source": source_name,
                   "fps": fps, "weights": args.weights, "conf": args.conf,
                   "min_frames": args.min_frames, "bad_ratio": args.bad_ratio}, f, indent=1)

    counter = ViolationCounter(args.min_frames, args.bad_ratio)
    track_kwargs = {"tracker": args.tracker} if args.tracker else {}
    csv_file = open(os.path.join(run_dir, "violations.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["violation_no", "track_id", "video_time_s", "alert_delay_s", "logged_at",
                     "source", "frames_seen", "avg_confidence", "snapshot", "crop"])

    records, delays = [], []
    alert_text, alert_until = "", 0.0
    start = time.time()
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, persist=True, conf=args.conf, imgsz=args.imgsz,
                                 verbose=False, **track_kwargs)[0]
            dets = get_tracked_detections(result, names)
            t_sec = frame_idx / fps if args.video else time.time() - start

            # only boxes that already have a track ID are counted
            events = counter.update(
                frame_idx, [(i, bad, c, box) for i, bad, c, box, _ in dets if i is not None],
                t=t_sec)

            raw = frame.copy() if events else None     # clean copy before drawing
            frame = draw(frame, dets, counter)

            for ev in events:
                delay = max(0.0, t_sec - (ev["first_bad_t"] if ev["first_bad_t"] is not None else t_sec))
                delays.append(delay)
                save_event(ev, raw, frame, run_dir, writer, source_name, t_sec, delay,
                           records, bad_name, fps)
                csv_file.flush()
                print(f"VIOLATION #{ev['violation_no']}  (rider id {ev['track_id']}, "
                      f"{t_sec:.1f}s, conf {ev['avg_conf']:.2f}, alert after {delay:.2f}s)")
                if args.warn:
                    alert_text = f"NO HELMET - PLEASE WEAR A HELMET  (rider #{ev['track_id']})"
                    alert_until = time.time() + args.warn_seconds
                    play_beep()
                    if args.speak:
                        speak("Please wear a helmet")

            if args.warn and time.time() < alert_until:
                draw_alert(frame, alert_text)

            if vw:
                vw.write(frame)
            if not args.no_show:
                cv2.imshow("Helmet violation tracking (q to stop)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_idx += 1
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        cap.release()
        if vw:
            vw.release()
        csv_file.close()
        if not args.no_show:
            cv2.destroyAllWindows()

    delay_line = (f"Alert delay (s)   : mean {sum(delays) / len(delays):.2f}, max {max(delays):.2f}\n"
                  if delays else "")
    summary = (
        f"Source            : {source_name}\n"
        f"Frames processed  : {frame_idx}\n"
        f"Unique violators  : {counter.n_violations}\n"
        f"Helmet riders     : {counter.helmet_riders()}\n"
        f"Tracks seen (all) : {len(counter.tracks)}\n"
        f"{delay_line}"
        f"Settings          : conf={args.conf}, min_frames={args.min_frames}, bad_ratio={args.bad_ratio}, "
        f"warn={args.warn}, weights={args.weights}\n"
    )
    open(os.path.join(run_dir, "summary.txt"), "w").write(summary)
    print("\n" + summary + f"Log + images saved in: {run_dir}")


if __name__ == "__main__":
    main()
