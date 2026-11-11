from pathlib import Path
import cv2, argparse, re

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help='Participant name, e.g. "Soujanya"')
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.name.strip()).strip("_")
    if not name or args.samples < 5:
        raise SystemExit("Use a valid name and request at least 5 samples.")
    folder = DATASET / name
    folder.mkdir(parents=True, exist_ok=True)
    detector = cv2.CascadeClassifier(CASCADE)
    camera = cv2.VideoCapture(0)
    if detector.empty() or not camera.isOpened():
        raise SystemExit("Could not load face detector or webcam. Check camera permissions.")
    count = 0
    print("Look at the camera and slowly turn your head slightly. Press Q to stop.")
    while count < args.samples:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        for (x, y, w, h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            count += 1
            cv2.imwrite(str(folder / f"{name}_{count:03d}.jpg"), face)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"Saved {count}/{args.samples}", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,0), 2)
            break
        cv2.imshow("Face registration - Q to quit", frame)
        if cv2.waitKey(150) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} samples in {folder}")

if __name__ == "__main__":
    main()
