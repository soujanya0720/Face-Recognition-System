from pathlib import Path
import cv2, json
import numpy as np

BASE = Path(__file__).resolve().parent
DATASET, TRAINER = BASE / "dataset", BASE / "trainer"

def main():
    if not hasattr(cv2, "face"):
        raise SystemExit("cv2.face missing. Install opencv-contrib-python in your active venv.")
    images, labels, names = [], [], {}
    folders = sorted(p for p in DATASET.iterdir() if p.is_dir())
    if not folders:
        raise SystemExit("No registered faces. Run register.py first.")
    for label, folder in enumerate(folders):
        names[str(label)] = folder.name
        for path in folder.glob("*.jpg"):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(cv2.resize(image, (200, 200)))
                labels.append(label)
    if len(images) < 5:
        raise SystemExit("Not enough face samples. Register at least 5 images.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels, dtype=np.int32))
    TRAINER.mkdir(exist_ok=True)
    recognizer.write(str(TRAINER / "face_model.yml"))
    (TRAINER / "labels.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(f"Training complete. Images: {len(images)}; people: {len(names)}")
    print("Model saved in trainer/face_model.yml")

if __name__ == "__main__":
    main()
