import cv2
import os
import numpy as np

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

# Create face detector
detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 240),
    0.6,
    0.3,
    5000
)

# Create SFace recognizer
recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("====================================")
print("      AI DOOR LOCK - ENROLLMENT")
print("====================================")
print()
print("Look directly at the camera.")
print("Keep your face inside the green box.")
print("Press SPACE to enroll your face.")
print("Press Q to cancel.")
print()

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read camera")
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
        "SPACE = Enroll | Q = Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Door Lock - Enrollment", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        print("Enrollment cancelled.")
        break

    if key == 32:  # SPACE
        if faces is None or len(faces) == 0:
            print("❌ No face detected. Try again.")
            continue

        if len(faces) > 1:
            print("❌ Multiple faces detected. Only one person should be visible.")
            continue

        # Get the first detected face
        face = faces[0]

        # Align and crop the face for SFace
        aligned_face = recognizer.alignCrop(frame, face)

        # Generate face feature/embedding
        feature = recognizer.feature(aligned_face)

        # Save authorized face feature
        os.makedirs("known_faces", exist_ok=True)

        np.save(
            "known_faces/authorized_face.npy",
            feature
        )

        print()
        print("====================================")
        print("✅ FACE ENROLLED SUCCESSFULLY")
        print("====================================")
        print("Saved to:")
        print("known_faces/authorized_face.npy")
        print()

        break

cap.release()
cv2.destroyAllWindows()