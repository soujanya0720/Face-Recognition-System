from pathlib import Path
import cv2, json

BASE = Path(__file__).resolve().parent
TRAINER = BASE / "trainer"
MODEL, LABELS = TRAINER / "face_model.yml", TRAINER / "labels.json"
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 65  # Lower LBPH distance generally means a closer match.

def main():
    if not hasattr(cv2, "face"):
        raise SystemExit("cv2.face missing. Install opencv-contrib-python.")
    if not MODEL.exists() or not LABELS.exists():
        raise SystemExit("Model missing. Run python train.py first.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL))
    names = json.loads(LABELS.read_text(encoding="utf-8"))
    detector = cv2.CascadeClassifier(CASCADE)
    camera = cv2.VideoCapture(0)
    if detector.empty() or not camera.isOpened():
        raise SystemExit("Could not load detector or webcam. Check camera permissions.")
    print("Recognition started. Press Q in the camera window to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
        for (x,y,w,h) in faces:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            label, distance = recognizer.predict(face)
            if distance <= CONFIDENCE_THRESHOLD and str(label) in names:
                text, color = f"{names[str(label)]} ({distance:.0f})", (0,180,0)
            else:
                text, color = f"Unknown ({distance:.0f})", (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, .7, color, 2)
        cv2.imshow("Face recognition - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
