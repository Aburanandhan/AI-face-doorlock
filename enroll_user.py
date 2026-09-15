import cv2
import numpy as np
import os


# ============================================================
# CONFIGURATION
# ============================================================

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

KNOWN_FACES_DIR = "known_faces"


# ============================================================
# CHECK REQUIRED MODELS
# ============================================================

required_files = [
    DETECTOR_MODEL,
    RECOGNITION_MODEL
]

for file in required_files:

    if not os.path.exists(file):

        print(f"❌ Missing file: {file}")
        input("Press ENTER to exit...")
        exit()


# ============================================================
# CAMERA AUTO-DETECTION
# ============================================================

def open_camera():

    print()
    print("🔍 Searching for webcam...")
    print()

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]

    # Try camera indexes 0 to 4
    for index in range(5):

        for backend_name, backend in backends:

            print(
                f"   Trying camera {index} "
                f"({backend_name})..."
            )

            camera = cv2.VideoCapture(
                index,
                backend
            )

            if not camera.isOpened():

                camera.release()
                continue

            # Test whether the camera can actually
            # capture a frame
            ret, frame = camera.read()

            if ret and frame is not None:

                print()
                print(
                    f"✅ Webcam found: "
                    f"Camera {index} "
                    f"({backend_name})"
                )
                print()

                return camera

            camera.release()

    return None


# ============================================================
# CREATE FACE DETECTOR
# ============================================================

print("Loading YuNet...")

detector = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL,
    "",
    (320, 240),
    0.6,
    0.3,
    5000
)

print("✅ YuNet loaded")


# ============================================================
# CREATE SFACE RECOGNIZER
# ============================================================

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

print("✅ SFace loaded")


# ============================================================
# GET USER NAME
# ============================================================

print()
print("====================================")
print("      AI DOOR LOCK - ENROLLMENT")
print("====================================")
print()

name = input("Enter person's name: ").strip()

if not name:

    print("❌ Name cannot be empty")
    input("Press ENTER to exit...")
    exit()


# ============================================================
# MAKE SAFE FILE NAME
# ============================================================

filename = "".join(
    c for c in name
    if c.isalnum() or c in (" ", "_", "-")
).strip()

filename = filename.replace(
    " ",
    "_"
)

if not filename:

    print("❌ Invalid name")
    input("Press ENTER to exit...")
    exit()


# ============================================================
# CREATE KNOWN FACES DIRECTORY
# ============================================================

os.makedirs(
    KNOWN_FACES_DIR,
    exist_ok=True
)


# ============================================================
# OPEN CAMERA
# ============================================================

cap = open_camera()

if cap is None:

    print()
    print("❌ No working webcam was found.")
    print()
    print("Please check:")
    print("• Webcam is connected")
    print("• Camera permission is enabled")
    print("• Another application is not using the webcam")
    print()

    input("Press ENTER to exit...")
    exit()


# ============================================================
# ENROLLMENT INSTRUCTIONS
# ============================================================

print()
print(f"Enrolling: {name}")
print()
print("Look directly at the camera.")
print("Keep your face inside the green box.")
print("Press SPACE to enroll your face.")
print("Press Q to cancel.")
print()


# ============================================================
# MAIN ENROLLMENT LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("❌ Failed to read camera")
        break


    # --------------------------------------------------------
    # IMAGE SIZE
    # --------------------------------------------------------

    h, w = frame.shape[:2]

    detector.setInputSize(
        (w, h)
    )


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    _, faces = detector.detect(
        frame
    )


    # --------------------------------------------------------
    # DRAW DETECTED FACES
    # --------------------------------------------------------

    if faces is not None:

        for face in faces:

            x, y, fw, fh = (
                face[:4].astype(int)
            )

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
                (
                    x,
                    max(y - 10, 20)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # INSTRUCTIONS ON SCREEN
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "SPACE = Enroll | Q = Quit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "AI Door Lock - Enrollment",
        frame
    )


    # --------------------------------------------------------
    # KEYBOARD
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        print("Enrollment cancelled.")
        break


    # --------------------------------------------------------
    # ENROLL
    # --------------------------------------------------------

    if key == 32:

        # No face
        if faces is None or len(faces) == 0:

            print(
                "❌ No face detected. "
                "Try again."
            )

            continue


        # Multiple faces
        if len(faces) > 1:

            print(
                "❌ Multiple faces detected. "
                "Only one person should be visible."
            )

            continue


        # ----------------------------------------------------
        # GET FACE
        # ----------------------------------------------------

        face = faces[0]


        # ----------------------------------------------------
        # ALIGN FACE
        # ----------------------------------------------------

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )


        # ----------------------------------------------------
        # GENERATE SFACE FEATURE
        # ----------------------------------------------------

        feature = recognizer.feature(
            aligned_face
        )


        # ----------------------------------------------------
        # SAVE FEATURE
        # ----------------------------------------------------

        save_path = os.path.join(
            KNOWN_FACES_DIR,
            f"{filename}.npy"
        )

        np.save(
            save_path,
            feature
        )


        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("====================================")
        print("✅ FACE ENROLLED SUCCESSFULLY")
        print("====================================")
        print(f"Name: {name}")
        print(f"Saved to: {save_path}")
        print()

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()