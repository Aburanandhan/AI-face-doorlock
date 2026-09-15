import cv2
import numpy as np
import os

# ==============================
# SETTINGS
# ==============================

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"
KNOWN_FACES_DIR = "known_faces"

# ==============================
# LOAD MODELS
# ==============================

print("Loading YuNet...")

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 240),
    0.6,
    0.3,
    5000
)

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

# ==============================
# LOAD AUTHORIZED FEATURES
# ==============================

authorized_faces = {}

for file in os.listdir(KNOWN_FACES_DIR):

    if file.lower().endswith(".npy"):

        path = os.path.join(
            KNOWN_FACES_DIR,
            file
        )

        name = os.path.splitext(file)[0]

        feature = np.load(path)

        authorized_faces[name] = feature

        print(f"Loaded: {name}")

print()

# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(
    1,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("❌ Camera failed")
    exit()

print("====================================")
print("RECOGNITION DIAGNOSTIC")
print("====================================")
print()
print("Look at the camera.")
print("Press Q to quit.")
print()

# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None and len(faces) == 1:

        face = faces[0]

        # Align
        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        # Feature
        feature = recognizer.feature(
            aligned_face
        )

        print("------------------------------------")

        # Compare with every authorized user
        for name, known_feature in authorized_faces.items():

            score = recognizer.match(
                known_feature,
                feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            print(
                f"{name}: {score:.4f}"
            )

        # Draw face
        x, y, fw, fh = face[:4].astype(int)

        cv2.rectangle(
            frame,
            (x, y),
            (x + fw, y + fh),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FACE DETECTED",
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    elif faces is not None and len(faces) > 1:

        cv2.putText(
            frame,
            "MULTIPLE FACES",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Recognition Diagnostic",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()