import cv2
import numpy as np
import os
import time
import math

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"
LANDMARK_MODEL = "models/face_landmarker.task"

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "logs/access_log.txt"

# Recognition
RECOGNITION_THRESHOLD = 0.45
RECOGNITION_MARGIN = 0.08
RECOGNITION_SAMPLES = 5

# Liveness
YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
OPEN_BASELINE_FRAMES = 15
OPEN_AFTER_BLINK_FRAMES = 5

CHALLENGE_TIMEOUT = 15

# Camera
MAX_CAMERA_INDEX = 5


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def log_event(message):
    ensure_directories()

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"[WARNING] Could not write log: {e}")


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    """
    Automatically searches camera indexes 0-5.

    Rejects cameras that return black/empty frames.
    """

    print("\n[CAMERA] Searching for working camera...")

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for index in range(MAX_CAMERA_INDEX + 1):

        for backend_name, backend in backends:

            try:
                cap = cv2.VideoCapture(index, backend)

                if not cap.isOpened():
                    cap.release()
                    continue

                time.sleep(0.3)

                ret, frame = cap.read()

                if not ret or frame is None:
                    cap.release()
                    continue

                # Check whether the frame is basically black
                mean_value = float(np.mean(frame))
                max_value = int(np.max(frame))

                if mean_value < 5 or max_value < 10:
                    cap.release()
                    continue

                print(
                    f"[CAMERA] Using camera {index} "
                    f"({backend_name})"
                )

                return cap

            except Exception:
                try:
                    cap.release()
                except Exception:
                    pass

    return None


# ============================================================
# LOAD MODELS
# ============================================================

def load_models():

    print("\n[MODEL] Loading YuNet face detector...")

    if not os.path.exists(DETECTOR_MODEL):
        raise FileNotFoundError(
            f"Missing detector model:\n{DETECTOR_MODEL}"
        )

    detector = cv2.FaceDetectorYN.create(
        DETECTOR_MODEL,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    print("[OK] YuNet loaded.")

    print("\n[MODEL] Loading SFace recognizer...")

    if not os.path.exists(RECOGNITION_MODEL):
        raise FileNotFoundError(
            f"Missing recognition model:\n{RECOGNITION_MODEL}"
        )

    recognizer = cv2.FaceRecognizerSF.create(
        RECOGNITION_MODEL,
        ""
    )

    print("[OK] SFace loaded.")

    print("\n[MODEL] Loading MediaPipe Face Landmarker...")

    if not os.path.exists(LANDMARK_MODEL):
        raise FileNotFoundError(
            f"Missing landmark model:\n{LANDMARK_MODEL}"
        )

    base_options = python.BaseOptions(
        model_asset_path=LANDMARK_MODEL
    )

    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    landmarker = vision.FaceLandmarker.create_from_options(
        options
    )

    print("[OK] MediaPipe Face Landmarker loaded.")

    return detector, recognizer, landmarker


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(detector, frame):

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    # Select largest face
    best_face = None
    best_area = 0

    for face in faces:

        x, y, w, h = face[:4]

        area = w * h

        if area > best_area:
            best_area = area
            best_face = face

    return best_face


# ============================================================
# FACE RECOGNITION
# ============================================================

def get_face_feature(recognizer, frame, face):

    try:

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )

        return feature

    except Exception as e:

        print(
            f"[WARNING] Could not extract face feature: {e}"
        )

        return None


def load_known_faces():

    known_faces = {}

    ensure_directories()

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
                f"[WARNING] Could not load {filename}: {e}"
            )

    return known_faces


def recognize_face(
    recognizer,
    feature,
    known_faces
):

    scores = {}

    for name, known_feature in known_faces.items():

        try:

            score = recognizer.match(
                feature,
                known_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            scores[name] = float(score)

        except Exception as e:

            print(
                f"[WARNING] Recognition error for {name}: {e}"
            )

    if not scores:
        return None, 0.0, 0.0, scores

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    best_name = sorted_scores[0][0]
    best_score = sorted_scores[0][1]

    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
    else:
        second_score = 0.0

    margin = best_score - second_score

    return (
        best_name,
        best_score,
        margin,
        scores
    )


# ============================================================
# LANDMARK / LIVENESS
# ============================================================

# MediaPipe Face Mesh landmark indexes
LEFT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]

RIGHT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]


def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ear(landmarks, eye):

    p1 = landmarks[eye[0]]
    p2 = landmarks[eye[1]]
    p3 = landmarks[eye[2]]
    p4 = landmarks[eye[3]]
    p5 = landmarks[eye[4]]
    p6 = landmarks[eye[5]]

    horizontal = distance(p1, p4)

    vertical1 = distance(p2, p6)
    vertical2 = distance(p3, p5)

    if horizontal == 0:
        return 0

    ear = (
        vertical1 + vertical2
    ) / (2.0 * horizontal)

    return ear


def get_landmarks(landmarker, frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = vision.Image(
        image_format=vision.ImageFormat.SRGB,
        data=rgb_frame
    )

    try:

        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    except Exception:

        return None


# ============================================================
# BLINK DETECTION
# ============================================================

def detect_blink(
    landmarker,
    frame,
    state
):

    landmarks = get_landmarks(
        landmarker,
        frame
    )

    if landmarks is None:
        return False, None

    # Convert normalized coordinates
    points = []

    height, width = frame.shape[:2]

    for landmark in landmarks:

        points.append(
            (
                landmark.x * width,
                landmark.y * height
            )
        )

    left_ear = calculate_ear(
        points,
        LEFT_EYE
    )

    right_ear = calculate_ear(
        points,
        RIGHT_EYE
    )

    ear = (
        left_ear + right_ear
    ) / 2.0

    blink_detected = False

    # Build open-eye baseline
    if not state["baseline_ready"]:

        if ear > EAR_OPEN_THRESHOLD:

            state["open_frames"] += 1

        else:

            state["open_frames"] = 0

        if state["open_frames"] >= OPEN_BASELINE_FRAMES:

            state["baseline_ready"] = True

    # Blink detection
    if state["baseline_ready"]:

        if ear < EAR_CLOSED_THRESHOLD:

            state["closed_frames"] += 1

        else:

            if (
                state["closed_frames"]
                >= BLINK_CLOSED_FRAMES
            ):

                state["blink_seen"] = True
                blink_detected = True

            state["closed_frames"] = 0

    return blink_detected, ear


# ============================================================
# HEAD TURN
# ============================================================

def estimate_head_yaw(landmarks, frame):

    if landmarks is None:
        return None

    height, width = frame.shape[:2]

    points = []

    for landmark in landmarks:

        points.append(
            (
                landmark.x * width,
                landmark.y * height
            )
        )

    # Approximate nose and eye positions
    nose = points[1]

    left_eye = points[33]
    right_eye = points[263]

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2,
        (left_eye[1] + right_eye[1]) / 2
    )

    face_width = distance(
        left_eye,
        right_eye
    )

    if face_width == 0:
        return None

    horizontal_offset = (
        nose[0] - eye_center[0]
    )

    yaw = (
        horizontal_offset
        / face_width
    ) * 90.0

    return yaw


# ============================================================
# LIVENESS CHALLENGE
# ============================================================

def perform_liveness(
    cap,
    landmarker
):

    print("\n----------------------------------------")
    print("LIVENESS CHECK")
    print("----------------------------------------")

    print("Please:")
    print("1. Keep your eyes open.")
    print("2. Blink once.")
    print("3. Turn your head LEFT or RIGHT.")
    print("----------------------------------------")

    start_time = time.time()

    state = {
        "baseline_ready": False,
        "open_frames": 0,
        "closed_frames": 0,
        "blink_seen": False,
    }

    head_turn_seen = False
    yaw_direction = None

    while True:

        elapsed = time.time() - start_time

        if elapsed > CHALLENGE_TIMEOUT:

            print("\n[LIVENESS] Timeout.")

            log_event(
                "LIVENESS FAILED - TIMEOUT"
            )

            return False

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face_present = True

        # Blink
        blink_now, ear = detect_blink(
            landmarker,
            frame,
            state
        )

        if blink_now:

            print("[LIVENESS] Blink detected.")

        # Head turn
        landmarks = get_landmarks(
            landmarker,
            frame
        )

        yaw = estimate_head_yaw(
            landmarks,
            frame
        )

        if yaw is not None:

            if abs(yaw) >= YAW_REQUIRED:

                if not head_turn_seen:

                    head_turn_seen = True

                    if yaw < 0:
                        yaw_direction = "LEFT"
                    else:
                        yaw_direction = "RIGHT"

                    print(
                        f"[LIVENESS] Head turned "
                        f"{yaw_direction}."
                    )

        # Determine final result
        if (
            state["blink_seen"]
            and head_turn_seen
        ):

            print("\n[LIVENESS] PASSED")

            log_event(
                f"LIVENESS PASSED - HEAD TURN {yaw_direction}"
            )

            # Show success briefly
            cv2.putText(
                frame,
                "LIVENESS PASSED",
                (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "AI Face Door Lock",
                frame
            )

            cv2.waitKey(500)

            return True

        # Display status
        cv2.putText(
            frame,
            "LIVENESS CHECK",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        blink_status = (
            "YES"
            if state["blink_seen"]
            else "NO"
        )

        turn_status = (
            "YES"
            if head_turn_seen
            else "NO"
        )

        cv2.putText(
            frame,
            f"Blink: {blink_status}",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Head Turn: {turn_status}",
            (30, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print("\n[LIVENESS] Cancelled.")

            return False


# ============================================================
# AUTHENTICATION
# ============================================================

def authenticate(
    cap,
    detector,
    recognizer,
    landmarker,
    known_faces
):

    print("\n")
    print("========================================")
    print("          AUTHENTICATION")
    print("========================================")

    if not known_faces:

        print(
            "[ERROR] No authorized users enrolled."
        )

        print(
            "Run enroll_user.py first."
        )

        return False

    print(
        f"[AUTH] Collecting "
        f"{RECOGNITION_SAMPLES} face samples..."
    )

    samples = []

    sample_start = time.time()

    while len(samples) < RECOGNITION_SAMPLES:

        if time.time() - sample_start > 10:

            print(
                "[AUTH] Recognition timeout."
            )

            log_event(
                "ACCESS DENIED - RECOGNITION TIMEOUT"
            )

            return False

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = detect_face(
            detector,
            frame
        )

        if face is None:

            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "AI Face Door Lock",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

            continue

        # Face feature
        feature = get_face_feature(
            recognizer,
            frame,
            face
        )

        if feature is None:
            continue

        samples.append(feature)

        print(
            f"[AUTH] Sample "
            f"{len(samples)}/{RECOGNITION_SAMPLES}"
        )

        # Draw face box
        x, y, w, h = face[:4]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Collecting sample "
            f"{len(samples)}/{RECOGNITION_SAMPLES}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        key = cv2.waitKey(150) & 0xFF

        if key == ord("q"):
            return False

    # --------------------------------------------------------
    # Average recognition features
    # --------------------------------------------------------

    feature_stack = np.vstack(samples)

    average_feature = np.mean(
        feature_stack,
        axis=0,
        keepdims=True
    )

    print("\n[AUTH] Recognition scores:")

    best_name = None
    best_score = -1.0

    all_scores = {}

    for name, known_feature in known_faces.items():

        try:

            score = recognizer.match(
                average_feature,
                known_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            score = float(score)

            all_scores[name] = score

            print(
                f"  {name}: {score:.4f}"
            )

            if score > best_score:

                best_score = score
                best_name = name

        except Exception as e:

            print(
                f"[WARNING] Could not compare "
                f"with {name}: {e}"
            )

    if best_name is None:

        print(
            "[ACCESS DENIED] No recognition result."
        )

        log_event(
            "ACCESS DENIED - NO RECOGNITION RESULT"
        )

        return False

    # --------------------------------------------------------
    # Calculate margin
    # --------------------------------------------------------

    sorted_scores = sorted(
        all_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if len(sorted_scores) > 1:

        second_score = sorted_scores[1][1]

    else:

        second_score = 0.0

    margin = (
        best_score - second_score
    )

    print(
        f"\n[AUTH] Best match: {best_name}"
    )

    print(
        f"[AUTH] Match score: {best_score:.4f}"
    )

    print(
        f"[AUTH] Margin: {margin:.4f}"
    )

    # --------------------------------------------------------
    # Recognition decision
    # --------------------------------------------------------

    if best_score < RECOGNITION_THRESHOLD:

        print(
            "\n[ACCESS DENIED]"
        )

        print(
            f"Score {best_score:.4f} "
            f"is below threshold "
            f"{RECOGNITION_THRESHOLD:.2f}"
        )

        log_event(
            f"ACCESS DENIED - "
            f"UNKNOWN USER - "
            f"BEST={best_name} "
            f"SCORE={best_score:.4f}"
        )

        return False

    # If there are multiple users, require margin
    if (
        len(sorted_scores) > 1
        and margin < RECOGNITION_MARGIN
    ):

        print(
            "\n[ACCESS DENIED]"
        )

        print(
            f"Recognition margin "
            f"{margin:.4f} is too small."
        )

        log_event(
            f"ACCESS DENIED - "
            f"LOW MARGIN - "
            f"BEST={best_name} "
            f"SCORE={best_score:.4f} "
            f"MARGIN={margin:.4f}"
        )

        return False

    print(
        "\n[RECOGNITION PASSED]"
    )

    print(
        f"Authorized user candidate: "
        f"{best_name}"
    )

    # --------------------------------------------------------
    # Liveness
    # --------------------------------------------------------

    liveness_passed = perform_liveness(
        cap,
        landmarker
    )

    if not liveness_passed:

        print(
            "\n[ACCESS DENIED]"
        )

        print(
            "Liveness verification failed."
        )

        log_event(
            f"ACCESS DENIED - "
            f"LIVENESS FAILED - "
            f"USER={best_name}"
        )

        return False

    # --------------------------------------------------------
    # FINAL ACCESS GRANTED
    # --------------------------------------------------------

    print("\n")
    print("========================================")
    print("          ACCESS GRANTED")
    print("========================================")

    print(
        f"Authorized user: {best_name}"
    )

    print(
        f"Match score: {best_score:.4f}"
    )

    print(
        "[LIVENESS] Passed"
    )

    print(
        "[DOOR] UNLOCKED"
    )

    print(
        "========================================"
    )

    log_event(
        f"ACCESS GRANTED - "
        f"USER={best_name} "
        f"SCORE={best_score:.4f} "
        f"MARGIN={margin:.4f}"
    )

    return True


# ============================================================
# DRAW LOCKED SCREEN
# ============================================================

def show_locked_screen(
    frame,
    face_detected=False
):

    height, width = frame.shape[:2]

    # Dark red lock status
    cv2.putText(
        frame,
        "DOOR LOCKED",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3
    )

    if face_detected:

        cv2.putText(
            frame,
            "FACE DETECTED",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Press SPACE to authenticate",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "Waiting for face...",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        "Q = Exit",
        (30, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )


# ============================================================
# DRAW UNLOCKED SCREEN
# ============================================================

def show_unlocked_screen(frame):

    height, width = frame.shape[:2]

    cv2.putText(
        frame,
        "DOOR UNLOCKED",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "ACCESS GRANTED",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Authentication successful",
        (30, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Door will remain unlocked.",
        (30, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to exit",
        (30, height - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("\n")
    print("========================================")
    print("       AI FACE DOOR LOCK SYSTEM")
    print("========================================")

    ensure_directories()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    try:

        detector, recognizer, landmarker = load_models()

    except Exception as e:

        print("\n[ERROR] Model loading failed.")
        print(e)

        input("\nPress Enter to exit...")
        return

    # --------------------------------------------------------
    # Load authorized users
    # --------------------------------------------------------

    known_faces = load_known_faces()

    if not known_faces:

        print("\n")
        print("[ERROR] No authorized users found.")
        print()
        print(
            "Please enroll a user first using:"
        )
        print()
        print(
            "python enroll_user.py"
        )

        input("\nPress Enter to exit...")
        return

    print(
        f"\n[OK] {len(known_faces)} "
        f"authorized user(s) loaded."
    )

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:

        print(
            "\n[ERROR] No working camera found."
        )

        input("\nPress Enter to exit...")
        return

    # --------------------------------------------------------
    # Door state
    # --------------------------------------------------------

    door_locked = True

    print("\n")
    print("========================================")
    print("SYSTEM READY")
    print("========================================")
    print()
    print("DOOR STATUS: LOCKED")
    print()
    print("Look at the camera.")
    print("Press SPACE when ready to authenticate.")
    print("Press Q to exit.")
    print()

    # ========================================================
    # LOCKED STATE
    # ========================================================

    while door_locked:

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = detect_face(
            detector,
            frame
        )

        face_detected = face is not None

        # Draw face box
        if face is not None:

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2
            )

        show_locked_screen(
            frame,
            face_detected
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if key == ord("q"):

            print("\n[EXIT] Program closed.")
            break

        # ----------------------------------------------------
        # Authentication starts ONLY when SPACE is pressed
        # ----------------------------------------------------

        if key == 32:

            if not face_detected:

                print(
                    "\n[AUTH] No face detected."
                )

                continue

            # ------------------------------------------------
            # ONE authentication attempt
            # ------------------------------------------------

            authenticated = authenticate(
                cap,
                detector,
                recognizer,
                landmarker,
                known_faces
            )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if authenticated:

                door_locked = False

                print("\n")
                print("########################################")
                print("#                                      #")
                print("#          DOOR IS UNLOCKED            #")
                print("#                                      #")
                print("#   NO MORE AUTHENTICATION REQUIRED    #")
                print("#                                      #")
                print("########################################")

                # IMPORTANT:
                # Do NOT set door_locked = True again.
                # The program now moves to the unlocked state.

            else:

                print("\n")
                print("----------------------------------------")
                print("ACCESS DENIED")
                print("DOOR REMAINS LOCKED")
                print("----------------------------------------")
                print()
                print(
                    "You may try again."
                )

    # ========================================================
    # UNLOCKED STATE
    # ========================================================

    while not door_locked:

        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        show_unlocked_screen(
            frame
        )

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # ----------------------------------------------------
        # IMPORTANT:
        # No authentication here.
        # No automatic relocking.
        # ----------------------------------------------------

        if key == ord("q"):

            print("\n")
            print("========================================")
            print("Program terminated.")
            print("Door was left in UNLOCKED state.")
            print("========================================")

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()

    try:
        landmarker.close()
    except Exception:
        pass


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()