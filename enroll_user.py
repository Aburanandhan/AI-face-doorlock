import cv2
import numpy as np
import os

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

os.makedirs("known_faces", exist_ok=True)

name = input("Enter person's name: ").strip()

if not name:
    print("❌ Name cannot be empty")
    exit()

# Make filename safe
filename = "".join(
    c for c in name
    if c.isalnum() or c in (" ", "_", "-")
).strip()

filename = filename.replace(" ", "_")

# Load models
detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 240),
    0.6,
    0.3,
    5000
)

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

# Open camera
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print()
print(f"Enrolling: {name}")
print("Look directly at the camera.")
print("Press SPACE to save the face.")
print("Press Q to cancel.")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera frame failed")
        break

    h, w = frame.shape[:2]

    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is not None:

        for face in faces:

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
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        "SPACE = Save | Q = Cancel",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Enroll Authorized User", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Enrollment cancelled.")
        break

    if key == 32:

        if faces is None or len(faces) == 0:
            print("❌ No face detected")
            continue

        if len(faces) > 1:
            print("❌ Multiple faces detected")
            continue

        face = faces[0]

        # Align face
        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        # Generate SFace feature
        feature = recognizer.feature(
            aligned_face
        )

        # Save feature
        save_path = f"known_faces/{filename}.npy"

        np.save(
            save_path,
            feature
        )

        print()
        print("================================")
        print("✅ USER ENROLLED")
        print("================================")
        print(f"Name: {name}")
        print(f"Saved: {save_path}")
        print()

        break

cap.release()
cv2.destroyAllWindows()