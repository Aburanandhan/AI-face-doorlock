import cv2

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Load YuNet
detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 240),
    0.6,
    0.3,
    5000
)

# Open working webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break

    # YuNet requires the current image size
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    # Detect faces
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

    cv2.imshow("YuNet Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()