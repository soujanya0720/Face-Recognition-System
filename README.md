# Face Recognition System using Python and OpenCV

Beginner-friendly local demo using Haar Cascade face detection and LBPH face recognition.

## Setup (Windows PowerShell in VS Code)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.

## Run

1. Register your own face (or another participant who clearly consents):

   `python register.py --name "Soujanya" --samples 30`

   Look at the webcam and slowly change your head angle. Press Q to stop early.

2. Train the model:

   `python train.py`

3. Recognize faces:

   `python recognize.py`

   Press Q in the webcam window to quit. Register another consenting person by repeating step 1 with a different name, then retrain.

## Files

- `register.py`: captures grayscale face samples into `dataset/`
- `train.py`: trains an LBPH recognizer and saves it under `trainer/`
- `recognize.py`: detects and labels registered faces or Unknown
- `requirements.txt`: required packages

## How it works

Haar Cascade locates faces. LBPH (Local Binary Patterns Histograms) learns visual patterns from the saved samples and predicts a label for a new face. LBPH confidence is a distance; lower usually means a closer match. The threshold in `recognize.py` may need tuning for your lighting and camera.

## Responsible use and limitations

This is an educational prototype, not secure authentication. Lighting, pose, and camera quality affect results. A match is not proof of identity. Do not use it for access control, attendance decisions, surveillance, or high-impact decisions. Only capture images with clear consent, keep the local dataset private, and delete it when no longer needed. Images are not uploaded to a server.

## Troubleshooting

- If `cv2.face` is missing, use `opencv-contrib-python` in the active virtual environment. Do not install both `opencv-python` and `opencv-contrib-python`.
- If webcam does not open, close other camera apps and check Windows camera permissions.
- If recognition is poor, register more varied samples and adjust `CONFIDENCE_THRESHOLD` carefully.
