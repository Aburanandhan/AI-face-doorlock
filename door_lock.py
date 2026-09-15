import cv2
import numpy as np
import os
import time
import math
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


# ============================================================
# CONFIGURATION
# ============================================================

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"
LANDMARK_MODEL = "models/face_landmarker.task"

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "logs/access_log.txt"

RECOGNITION_THRESHOLD = 0.45
RECOGNITION_MARGIN = 0.08

CAMERA_MAX_INDEX = 5

# Liveness
YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
OPEN_BASELINE_FRAMES = 10
OPEN_AFTER_BLINK_FRAMES = 3

CHALLENGE_TIMEOUT = 15

# Number of recognition samples during ONE authentication attempt
RECOGNITION_SAMPLES = 5


# ============================================================
# CAMERA SEARCH
# ============================================================

def open_working_camera():

    print("\nSearching for a working camera...")

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for index in range(CAMERA_MAX_INDEX + 1):

        for backend_name, backend in backends:

            print(
                f"  Trying camera {index} using {backend_name}..."
            )

            cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            # Give camera time to initialize
            time.sleep(0.3)

            ret, frame = cap.read()

            if not ret or frame is None:
                cap.release()
                continue

            if frame.size == 0:
                cap.release()
                continue

            mean_value = float(np.mean(frame))
            min_value = int(np.min(frame))
            max_value = int(np.max(frame))

            # Reject completely black camera
            if max_value <= 5 or mean_value < 2:
                cap.release()
                continue

            print("\n[OK] Working camera found!")
            print(f"     Camera index : {index}")
            print(f"     Backend      : {backend_name}")
            print(
                f"     Resolution   : "
                f"{frame.shape[1]}x{frame.shape[0]}"
            )

            return cap

    return None


# ============================================================
# EAR CALCULATION
# ============================================================

def calculate_ear(landmarks, eye_indices):

    points = []

    for index in eye_indices:

        landmark = landmarks[index]

        points.append(
            np.array(
                [
                    landmark.x,
                    landmark.y
                ],
                dtype=np.float32
            )
        )

    p1, p2, p3, p4, p5, p6 = points

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)

    horizontal = np.linalg.norm(p1 - p4)

    if horizontal == 0:
        return 0.0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return float(ear)


# ============================================================
# APPROXIMATE HEAD YAW
# ============================================================

def calculate_yaw(landmarks):

    # Approximate nose position
    nose = landmarks[1]

    # Left and right cheek / face points
    left_face = landmarks[234]
    right_face = landmarks[454]

    face_width = abs(
        right_face.x - left_face.x
    )

    if face_width < 0.001:
        return 0.0

    nose_position = (
        nose.x - left_face.x
    ) / face_width

    # Convert normalized position to approximate yaw
    yaw = (
        nose_position - 0.5
    ) * 90.0

    return float(yaw)


# ============================================================
# LOAD AUTHORIZED FACES
# ============================================================

def load_known_faces():

    known_faces = {}

    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    files = os.listdir(KNOWN_FACES_DIR)

    for filename in files:

        if not filename.lower().endswith(".npy"):
            continue

        path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        try:

            feature = np.load(path)

            name = os.path.splitext(filename)[0]

            known_faces[name] = feature

            print(
                f"[OK] Loaded authorized user: {name}"
            )

        except Exception as e:

            print(
                f"[ERROR] Could not load {filename}: {e}"
            )

    print(
        f"\nTotal authorized users: "
        f"{len(known_faces)}"
    )

    return known_faces


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(
    frame,
    detector,
    recognizer,
    known_faces,
    print_scores=True
):

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None, 0.0, 0.0

    # Use largest detected face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    try:

        aligned = recognizer.alignCrop(
            frame,
            face
        )

        feature = recognizer.feature(
            aligned
        )

    except Exception:
        return None, 0.0, 0.0

    scores = {}

    for name, known_feature in known_faces.items():

        try:

            score = recognizer.match(
                feature,
                known_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            scores[name] = float(score)

        except Exception:
            continue

    if not scores:
        return None, 0.0, 0.0

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_name, best_score = sorted_scores[0]

    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
    else:
        second_score = 0.0

    margin = best_score - second_score

    if print_scores:

        print("\n[RECOGNITION]")

        for name, score in sorted_scores:
            print(
                f"  {name}: {score:.4f}"
            )

        print(
            f"  Best: {best_name}"
        )

        print(
            f"  Score: {best_score:.4f}"
        )

        print(
            f"  Margin: {margin:.4f}"
        )

    return best_name, best_score, margin


# ============================================================
# LOG ACCESS
# ============================================================

def log_access(name, score, granted):

    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    status = (
        "GRANTED"
        if granted
        else "DENIED"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            f"{timestamp} | "
            f"{name} | "
            f"{status} | "
            f"{score:.4f}\n"
        )


# ============================================================
# MEDIAPIPE LANDMARKER
# ============================================================

def create_landmarker():

    base_options = python.BaseOptions(
        model_asset_path=LANDMARK_MODEL
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    return landmarker


# ============================================================
# MEDIAPIPE LANDMARK DETECTION
# ============================================================

def get_landmarks(frame, landmarker):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = landmarker.detect(
        mp_image
    )

    if not result.face_landmarks:
        return None

    return result.face_landmarks[0]


# ============================================================
# LIVENESS
# ============================================================

def perform_liveness(
    cap,
    landmarker
):

    print("\n============================================================")
    print("LIVENESS VERIFICATION")
    print("============================================================")

    print("Please look straight at the camera.")

    start_time = time.time()

    open_frames = 0
    closed_frames = 0

    blink_detected = False

    yaw_challenge = None
    yaw_completed = False

    # --------------------------------------------------------
    # Step 1: Wait for eyes-open baseline
    # --------------------------------------------------------

    while time.time() - start_time < CHALLENGE_TIMEOUT:

        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = get_landmarks(
            frame,
            landmarker
        )

        if landmarks is None:
            cv2.imshow(
                "AI Face Door Lock",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

            continue

        left_ear = calculate_ear(
            landmarks,
            [33, 160, 158, 133, 153, 144]
        )

        right_ear = calculate_ear(
            landmarks,
            [362, 385, 387, 263, 373, 380]
        )

        ear = (
            left_ear + right_ear
        ) / 2.0

        if ear > EAR_OPEN_THRESHOLD:

            open_frames += 1

        else:

            open_frames = 0

        if open_frames >= OPEN_BASELINE_FRAMES:
            break

        cv2.putText(
            frame,
            "Look at camera",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    # --------------------------------------------------------
    # Step 2: Blink
    # --------------------------------------------------------

    print("\n[LIVENESS] Blink once.")

    blink_start = time.time()

    while time.time() - blink_start < CHALLENGE_TIMEOUT:

        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = get_landmarks(
            frame,
            landmarker
        )

        if landmarks is None:
            continue

        left_ear = calculate_ear(
            landmarks,
            [33, 160, 158, 133, 153, 144]
        )

        right_ear = calculate_ear(
            landmarks,
            [362, 385, 387, 263, 373, 380]
        )

        ear = (
            left_ear + right_ear
        ) / 2.0

        if ear < EAR_CLOSED_THRESHOLD:

            closed_frames += 1

        else:

            if closed_frames >= BLINK_CLOSED_FRAMES:

                blink_detected = True

            closed_frames = 0

        if blink_detected:

            break

        cv2.putText(
            frame,
            "BLINK",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    if not blink_detected:

        print("[LIVENESS] Blink not detected.")

        return False

    print("[LIVENESS] Blink detected.")

    # --------------------------------------------------------
    # Step 3: Head turn
    # --------------------------------------------------------

    yaw_challenge = np.random.choice(
        ["LEFT", "RIGHT"]
    )

    print(
        f"[LIVENESS] Turn your head "
        f"{yaw_challenge}"
    )

    turn_start = time.time()

    while time.time() - turn_start < CHALLENGE_TIMEOUT:

        ret, frame = cap.read()

        if not ret:
            continue

        landmarks = get_landmarks(
            frame,
            landmarker
        )

        if landmarks is None:
            continue

        yaw = calculate_yaw(
            landmarks
        )

        cv2.putText(
            frame,
            f"TURN {yaw_challenge}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        if yaw_challenge == "LEFT":

            if yaw < -YAW_REQUIRED:

                yaw_completed = True
                break

        else:

            if yaw > YAW_REQUIRED:

                yaw_completed = True
                break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    if not yaw_completed:

        print(
            "[LIVENESS] Head movement not detected."
        )

        return False

    print("[LIVENESS] Head movement detected.")
    print("[LIVENESS] PASSED")

    return True


# ============================================================
# ONE AUTHENTICATION ATTEMPT
# ============================================================

def authenticate(
    cap,
    detector,
    recognizer,
    landmarker,
    known_faces
):

    print("\n")
    print("============================================================")
    print("AUTHENTICATION")
    print("============================================================")

    print("Look at the camera.")

    # --------------------------------------------------------
    # Collect recognition samples ONCE
    # --------------------------------------------------------

    recognition_results = []

    start_time = time.time()

    while (
        len(recognition_results)
        < RECOGNITION_SAMPLES
        and time.time() - start_time < 8
    ):

        ret, frame = cap.read()

        if not ret:
            continue

        name, score, margin = recognize_face(
            frame,
            detector,
            recognizer,
            known_faces,
            print_scores=False
        )

        if name is not None:

            recognition_results.append(
                (name, score, margin)
            )

        cv2.putText(
            frame,
            "VERIFYING FACE...",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    if not recognition_results:

        print(
            "[ERROR] No face could be recognized."
        )

        return False

    # --------------------------------------------------------
    # Determine best identity from collected samples
    # --------------------------------------------------------

    grouped = {}

    for name, score, margin in recognition_results:

        if name not in grouped:
            grouped[name] = []

        grouped[name].append(score)

    average_scores = {}

    for name, scores in grouped.items():

        average_scores[name] = float(
            np.mean(scores)
        )

    best_name = max(
        average_scores,
        key=average_scores.get
    )

    best_average = average_scores[
        best_name
    ]

    # Find second-best average
    sorted_average = sorted(
        average_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if len(sorted_average) > 1:

        second_average = sorted_average[1][1]

    else:

        second_average = 0.0

    final_margin = (
        best_average - second_average
    )

    # --------------------------------------------------------
    # Display ONE recognition result
    # --------------------------------------------------------

    print("\n============================================================")
    print("FACE VERIFICATION RESULT")
    print("============================================================")

    for name, scores in grouped.items():

        print(
            f"{name}: "
            f"average={np.mean(scores):.4f}, "
            f"best={np.max(scores):.4f}"
        )

    print(
        f"\nBest match : {best_name}"
    )

    print(
        f"Average score : {best_average:.4f}"
    )

    print(
        f"Margin : {final_margin:.4f}"
    )

    print(
        f"Threshold : {RECOGNITION_THRESHOLD:.4f}"
    )

    # --------------------------------------------------------
    # LIVENESS
    # --------------------------------------------------------

    liveness_passed = perform_liveness(
        cap,
        landmarker
    )

    if not liveness_passed:

        print("\n[ACCESS DENIED]")
        print("[REASON] Liveness verification failed.")

        log_access(
            best_name,
            best_average,
            False
        )

        return False

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    recognized = (
        best_average >= RECOGNITION_THRESHOLD
        and final_margin >= RECOGNITION_MARGIN
    )

    if recognized:

        print("\n============================================================")
        print("[ACCESS GRANTED]")
        print(f"Authorized user: {best_name}")
        print(
            f"Match score: {best_average:.4f}"
        )
        print("[DOOR] UNLOCKED")
        print("============================================================")

        log_access(
            best_name,
            best_average,
            True
        )

        return True

    else:

        print("\n============================================================")
        print("[ACCESS DENIED]")
        print(
            f"Best match: {best_name}"
        )
        print(
            f"Match score: {best_average:.4f}"
        )
        print(
            f"Required: {RECOGNITION_THRESHOLD:.4f}"
        )
        print("[DOOR] LOCKED")
        print("============================================================")

        log_access(
            best_name,
            best_average,
            False
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("============================================================")
    print("           AI FACE DOOR LOCK")
    print("============================================================")

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    if not os.path.exists(DETECTOR_MODEL):

        print(
            f"[ERROR] Missing detector model:\n"
            f"{DETECTOR_MODEL}"
        )

        return

    if not os.path.exists(RECOGNITION_MODEL):

        print(
            f"[ERROR] Missing recognition model:\n"
            f"{RECOGNITION_MODEL}"
        )

        return

    if not os.path.exists(LANDMARK_MODEL):

        print(
            f"[ERROR] Missing landmark model:\n"
            f"{LANDMARK_MODEL}"
        )

        return

    # --------------------------------------------------------
    # Load YuNet
    # --------------------------------------------------------

    try:

        detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        print(
            "[OK] YuNet face detector loaded."
        )

    except Exception as e:

        print(
            f"[ERROR] YuNet loading failed: {e}"
        )

        return

    # --------------------------------------------------------
    # Load SFace
    # --------------------------------------------------------

    try:

        recognizer = cv2.FaceRecognizerSF.create(
            RECOGNITION_MODEL,
            ""
        )

        print(
            "[OK] SFace face recognizer loaded."
        )

    except Exception as e:

        print(
            f"[ERROR] SFace loading failed: {e}"
        )

        return

    # --------------------------------------------------------
    # Load MediaPipe
    # --------------------------------------------------------

    try:

        landmarker = create_landmarker()

        print(
            "[OK] MediaPipe face landmarker loaded."
        )

    except Exception as e:

        print(
            f"[ERROR] MediaPipe loading failed: {e}"
        )

        return

    # --------------------------------------------------------
    # Load authorized users
    # --------------------------------------------------------

    known_faces = load_known_faces()

    if not known_faces:

        print(
            "\n[ERROR] No authorized users found."
        )

        print(
            "Please enroll a user first."
        )

        return

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = open_working_camera()

    if cap is None:

        print(
            "\n[ERROR] No working camera found."
        )

        return

    print("\n")
    print("============================================================")
    print("DOOR LOCK ACTIVE")
    print("============================================================")
    print("Look at the camera.")
    print("Blink and turn your head when requested.")
    print("Press Q to exit.")
    print("============================================================")

    door_locked = True

    try:

        while True:

            # ------------------------------------------------
            # WAIT FOR FACE
            # ------------------------------------------------

            ret, frame = cap.read()

            if not ret:
                continue

            detector.setInputSize(
                (
                    frame.shape[1],
                    frame.shape[0]
                )
            )

            _, faces = detector.detect(
                frame
            )

            face_present = (
                faces is not None
                and len(faces) > 0
            )

            if face_present:

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(
                    "AI Face Door Lock",
                    frame
                )

                cv2.waitKey(500)

                # ============================================
                # ONE AUTHENTICATION ATTEMPT
                # ============================================

                granted = authenticate(
                    cap,
                    detector,
                    recognizer,
                    landmarker,
                    known_faces
                )

                if granted:

                    door_locked = False

                    print(
                        "\n[DOOR] UNLOCKED"
                    )

                    print(
                        "Door will remain unlocked for 5 seconds."
                    )

                    time.sleep(5)

                    door_locked = True

                    print(
                        "[DOOR] Locked again."
                    )

                else:

                    door_locked = True

                    print(
                        "\n[DOOR] Remains LOCKED."
                    )

                    # Short cooldown so the same face
                    # isn't immediately processed again
                    time.sleep(2)

            else:

                cv2.putText(
                    frame,
                    "DOOR LOCKED - WAITING FOR FACE",
                    (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "AI Face Door Lock",
                    frame
                )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

    except KeyboardInterrupt:

        print(
            "\nProgram interrupted by user."
        )

    finally:

        cap.release()

        cv2.destroyAllWindows()

        print(
            "\n[OK] Camera released."
        )

        print(
            "[OK] Door lock stopped."
        )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()