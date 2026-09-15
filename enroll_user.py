import cv2
import os
import numpy as np
import sys

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"

KNOWN_FACES_DIR = "known_faces"

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


# ============================================================
# CAMERA AUTO DETECTION
# ============================================================

def open_camera():

    print()
    print("Opening camera...")
    print("🔍 Searching for webcam...")
    print()

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]

    for index in range(6):

        for backend_name, backend in backends:

            print(
                f"   Trying camera {index} ({backend_name})..."
            )

            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            ret = False
            frame = None

            # Give camera some time to initialize
            for _ in range(5):

                ret, frame = cap.read()

                if ret and frame is not None:
                    break

            if not ret or frame is None:

                print("      ❌ No frame")
                cap.release()
                continue

            # Check whether the camera actually provides
            # a useful image.
            mean_value = float(np.mean(frame))
            max_value = int(np.max(frame))
            min_value = int(np.min(frame))

            print(
                f"      Frame: mean={mean_value:.2f}, "
                f"min={min_value}, max={max_value}"
            )

            # Reject completely black / empty cameras
            if max_value <= 5 or mean_value <= 1:

                print("      ❌ Empty/black camera - skipping")
                cap.release()
                continue

            print()
            print(
                f"✅ Webcam found: Camera {index} "
                f"({backend_name})"
            )
            print()

            return cap

    return None


# ============================================================
# GET USER NAME
# ============================================================

name = input("Enter person's name: ").strip()

if not name:

    print("❌ Name cannot be empty")
    input("\nPress Enter to exit...")
    sys.exit()


# ============================================================
# MAKE SAFE FILE NAME
# ============================================================

filename = "".join(
    c for c in name
    if c.isalnum() or c in (" ", "_", "-")
).strip()

filename = filename.replace(" ", "_")

if not filename:

    print("❌ Invalid name")
    input("\nPress Enter to exit...")
    sys.exit()


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print()
print("Loading face detection model...")

try:

    detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL,
        "",
        (320, 240),
        0.6,
        0.3,
        5000
    )

except Exception as e:

    print("❌ Failed to load face detector")
    print(e)
    input("\nPress Enter to exit...")
    sys.exit()


print("✅ Face detector loaded")


# ============================================================
# LOAD FACE RECOGNIZER
# ============================================================

print("Loading face recognition model...")

try:

    recognizer = cv2.FaceRecognizerSF.create(
        RECOGNITION_MODEL,
        ""
    )

except Exception as e:

    print("❌ Failed to load face recognizer")
    print(e)
    input("\nPress Enter to exit...")
    sys.exit()


print("✅ Face recognizer loaded")


# ============================================================
# OPEN CAMERA
# ============================================================

cap = open_camera()

if cap is None:

    print()
    print("❌ No working webcam was found.")
    print()
    print("Please check:")
    print("1. Webcam is connected")
    print("2. Camera permission is enabled")
    print("3. No other application is using the camera")
    print()

    input("Press Enter to exit...")
    sys.exit()


# ============================================================
# ENROLLMENT
# ============================================================

print("================================")
print(f"Enrolling: {name}")
print("================================")
print()
print("Look directly at the camera.")
print("Press SPACE to save the face.")
print("Press Q to cancel.")
print()


while True:

    ret, frame = cap.read()

    if not ret or frame is None:

        print("❌ Camera frame failed")
        break


    h, w = frame.shape[:2]

    detector.setInputSize((w, h))


    # ========================================================
    # FACE DETECTION
    # ========================================================

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


    # ========================================================
    # DISPLAY INSTRUCTIONS
    # ========================================================

    cv2.putText(
        frame,
        "SPACE = Save | Q = Cancel",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Enroll Authorized User",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # CANCEL
    # ========================================================

    if key == ord("q"):

        print()
        print("Enrollment cancelled.")
        break


    # ========================================================
    # SAVE FACE
    # ========================================================

    if key == 32:

        print()
        print("Processing face...")


        # ----------------------------------------------------
        # CHECK FACE
        # ----------------------------------------------------

        if faces is None or len(faces) == 0:

            print("❌ No face detected")
            print("Please position your face inside the camera.")
            continue


        if len(faces) > 1:

            print("❌ Multiple faces detected")
            print("Only one person should be visible.")
            continue


        face = faces[0]


        # ----------------------------------------------------
        # ALIGN FACE
        # ----------------------------------------------------

        try:

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

        except Exception as e:

            print("❌ Face alignment failed")
            print(e)
            continue


        if aligned_face is None:

            print("❌ Aligned face is empty")
            continue


        # ----------------------------------------------------
        # GENERATE FACE FEATURE
        # ----------------------------------------------------

        try:

            feature = recognizer.feature(
                aligned_face
            )

        except Exception as e:

            print("❌ Face feature generation failed")
            print(e)
            continue


        if feature is None:

            print("❌ Recognition feature is empty")
            continue


        print(
            f"✅ Feature generated: shape={feature.shape}"
        )


        # ----------------------------------------------------
        # SAVE FEATURE
        # ----------------------------------------------------

        save_path = os.path.abspath(
            os.path.join(
                KNOWN_FACES_DIR,
                filename + ".npy"
            )
        )


        print()
        print("Saving face feature...")
        print(f"Path: {save_path}")


        try:

            np.save(
                save_path,
                feature
            )

        except Exception as e:

            print("❌ Failed to save face feature")
            print(e)
            continue


        # ----------------------------------------------------
        # VERIFY FILE
        # ----------------------------------------------------

        if os.path.exists(save_path):

            file_size = os.path.getsize(
                save_path
            )

            print()
            print("================================")
            print("       ✅ USER ENROLLED")
            print("================================")
            print()
            print(f"Name: {name}")
            print(f"File: {save_path}")
            print(f"Size: {file_size} bytes")
            print()
            print("You can now start the Door Lock.")
            print()

        else:

            print()
            print("❌ File was not created!")
            print()
            print("Expected path:")
            print(save_path)

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

input("Press Enter to continue...")