"""
Helmet / No Helmet detection with a trained YOLO26 model (Ultralytics).

Run from the project folder (put your trained best.pt next to this file):

    python detect.py --image photo.jpg
    python detect.py --video clip.mp4 --save
    python detect.py --webcam                  # press 'q' to quit

Options:
    --weights best.pt     path to the trained model
    --conf 0.4            minimum confidence to show a detection
    --imgsz 640           inference size (use the same size you trained with)
"""
import argparse
import os
import sys
import time

import cv2

GREEN = (0, 200, 0)   # BGR: helmet
RED = (0, 0, 255)     # BGR: no helmet


def is_violation(class_name: str) -> bool:
    """True for classes like 'Without Helmet' / 'No Helmet' / 'no_helmet'."""
    n = class_name.lower()
    return "without" in n or n.startswith("no")


def get_detections(result):
    """Return a list of (xyxy, class_id, confidence) from one Ultralytics result."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    return list(zip(xyxy, cls, conf))


def annotate(frame, result, names):
    """Draw boxes + labels on the frame and a count banner. Returns (frame, n_ok, n_bad)."""
    n_ok = n_bad = 0
    for (x1, y1, x2, y2), cid, conf in get_detections(result):
        name = names[cid]
        bad = is_violation(name)
        n_bad += bad
        n_ok += not bad
        color = RED if bad else GREEN
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1 - th - 8, 0)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + 8), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    banner = f"Helmet: {n_ok}   No helmet: {n_bad}"
    cv2.rectangle(frame, (0, 0), (300, 30), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    return frame, n_ok, n_bad


def run_stream(model, source, args, save_path=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Could not open video source: {source}")

    writer = None
    if save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    last = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        frame, _, _ = annotate(frame, result, model.names)

        now = time.time()
        fps_text = f"{1 / max(now - last, 1e-6):.1f} FPS"
        last = now
        cv2.putText(frame, fps_text, (frame.shape[1] - 110, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if writer:
            writer.write(frame)
        cv2.imshow("Helmet Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved video to {save_path}")
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Helmet / No Helmet detection (YOLO26)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image")
    src.add_argument("--video", help="path to a video file")
    src.add_argument("--webcam", action="store_true", help="use the webcam")
    p.add_argument("--weights", default="best.pt", help="trained model file")
    p.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size")
    p.add_argument("--camera", type=int, default=0, help="webcam index")
    p.add_argument("--save", action="store_true", help="save the annotated video")
    p.add_argument("--no-show", action="store_true", help="do not open a window (image mode)")
    args = p.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"Model file not found: {args.weights}\n"
                 "Train it with the Colab notebook and put best.pt in this folder.")

    from ultralytics import YOLO  # imported here so --help is instant
    model = YOLO(args.weights)
    print("Model classes:", model.names)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Could not read image: {args.image}")
        result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        frame, n_ok, n_bad = annotate(frame, result, model.names)
        cv2.imwrite("output.jpg", frame)
        print(f"Helmet: {n_ok} | No helmet: {n_bad}  ->  saved output.jpg")
        if not args.no_show:
            cv2.imshow("Helmet Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    elif args.video:
        run_stream(model, args.video, args, "output.mp4" if args.save else None)
    else:
        run_stream(model, args.camera, args, "output.mp4" if args.save else None)


if __name__ == "__main__":
    main()
