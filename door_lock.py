import cv2
import numpy as np
import os
import time
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AI FACE DOOR LOCK
# Guided flow:
# FACE DETECTED -> BLINK -> HEAD TURN -> RECOGNITION
# -> ACCESS GRANTED / ACCESS DENIED
#
# After ACCESS GRANTED:
#   Door stays unlocked until Q is pressed.
#   No more authentication is performed.
# ============================================================

DETECTOR_MODEL = "models/face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "models/face_recognition_sface_2021dec.onnx"
LANDMARK_MODEL = "models/face_landmarker.task"

KNOWN_FACES_DIR = "known_faces"
LOG_FILE = "logs/access_log.txt"

CAMERA_MAX_INDEX = 5

RECOGNITION_THRESHOLD = 0.45
RECOGNITION_MARGIN = 0.08
RECOGNITION_SAMPLES = 5

YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19
BLINK_CLOSED_FRAMES = 3
OPEN_BASELINE_FRAMES = 10

BLINK_TIMEOUT = 12
HEAD_TURN_TIMEOUT = 12
RECOGNITION_TIMEOUT = 10
RESULT_DISPLAY_SECONDS = 3

WINDOW_NAME = "AI Face Door Lock"


# ============================================================
# FILE / LOG HELPERS
# ============================================================

def ensure_directories():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)


def log_event(message):
    ensure_directories()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as exc:
        print(f"[WARNING] Log write failed: {exc}")


# ============================================================
# CAMERA
# ============================================================

def open_camera():
    print("\n[CAMERA] Searching for a working camera...")

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    for index in range(CAMERA_MAX_INDEX + 1):
        for backend_name, backend in backends:
            cap = None

            try:
                cap = cv2.VideoCapture(index, backend)

                if not cap.isOpened():
                    cap.release()
                    continue

                time.sleep(0.25)
                ret, frame = cap.read()

                if not ret or frame is None:
                    cap.release()
                    continue

                mean_value = float(np.mean(frame))
                max_value = int(np.max(frame))

                # Reject black / unusable cameras.
                if mean_value < 5 or max_value < 10:
                    cap.release()
                    continue

                print(
                    f"[CAMERA] Using camera {index} "
                    f"({backend_name})"
                )
                return cap

            except Exception:
                if cap is not None:
                    cap.release()

    return None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    print("\n[MODEL] Loading YuNet...")

    if not os.path.exists(DETECTOR_MODEL):
        raise FileNotFoundError(
            f"Missing model: {DETECTOR_MODEL}"
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

    print("[MODEL] Loading SFace...")

    if not os.path.exists(RECOGNITION_MODEL):
        raise FileNotFoundError(
            f"Missing model: {RECOGNITION_MODEL}"
        )

    recognizer = cv2.FaceRecognizerSF.create(
        RECOGNITION_MODEL,
        ""
    )

    print("[OK] SFace loaded.")

    print("[MODEL] Loading MediaPipe Face Landmarker...")

    if not os.path.exists(LANDMARK_MODEL):
        raise FileNotFoundError(
            f"Missing model: {LANDMARK_MODEL}"
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

    # Use the largest detected face.
    best_face = None
    best_area = 0

    for face in faces:
        x, y, w, h = face[:4]
        area = float(w * h)

        if area > best_area:
            best_area = area
            best_face = face

    return best_face


def draw_face_box(frame, face, color=(0, 255, 255)):
    if face is None:
        return

    x, y, w, h = [int(v) for v in face[:4]]

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        2
    )


# ============================================================
# MEDIAPIPE LANDMARKS
# ============================================================

def get_landmarks(landmarker, frame):
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # IMPORTANT:
    # Image belongs to mediapipe, not mediapipe.tasks.python.vision.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    try:
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    except Exception as exc:
        print(f"[WARNING] Landmark detection error: {exc}")
        return None


def landmarks_to_pixels(landmarks, frame):
    height, width = frame.shape[:2]

    return [
        (
            landmark.x * width,
            landmark.y * height
        )
        for landmark in landmarks
    ]


# ============================================================
# BLINK
# ============================================================

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def point_distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1]
    )


def calculate_ear(points, eye):
    p1 = points[eye[0]]
    p2 = points[eye[1]]
    p3 = points[eye[2]]
    p4 = points[eye[3]]
    p5 = points[eye[4]]
    p6 = points[eye[5]]

    horizontal = point_distance(p1, p4)

    if horizontal <= 0:
        return 0.0

    vertical_1 = point_distance(p2, p6)
    vertical_2 = point_distance(p3, p5)

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def get_ear(landmarks, frame):
    if landmarks is None:
        return None

    points = landmarks_to_pixels(
        landmarks,
        frame
    )

    left = calculate_ear(
        points,
        LEFT_EYE
    )

    right = calculate_ear(
        points,
        RIGHT_EYE
    )

    return (left + right) / 2.0


# ============================================================
# HEAD YAW
# ============================================================

def estimate_head_yaw(landmarks, frame):
    if landmarks is None:
        return None

    points = landmarks_to_pixels(
        landmarks,
        frame
    )

    # Nose tip and eye corners.
    nose = points[1]
    left_eye = points[33]
    right_eye = points[263]

    eye_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0
    )

    eye_distance = point_distance(
        left_eye,
        right_eye
    )

    if eye_distance <= 0:
        return None

    # Approximate yaw.
    yaw = (
        (nose[0] - eye_center[0])
        / eye_distance
    ) * 90.0

    return yaw


# ============================================================
# UI
# ============================================================

def put_centered_text(
    frame,
    text,
    y,
    scale=1.0,
    thickness=2,
    color=(255, 255, 255)
):
    width = frame.shape[1]

    size, _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness
    )

    x = max(10, (width - size[0]) // 2)

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def show_locked(frame, message="Look at the camera"):
    cv2.putText(
        frame,
        "DOOR LOCKED",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )

    put_centered_text(
        frame,
        message,
        110,
        0.85,
        2,
        (255, 255, 255)
    )

    cv2.putText(
        frame,
        "SPACE = Authenticate    Q = Exit",
        (30, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


def show_unlocked(frame, user_name):
    cv2.putText(
        frame,
        "DOOR UNLOCKED",
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    put_centered_text(
        frame,
        f"Welcome {user_name}",
        125,
        0.9,
        2,
        (0, 255, 0)
    )

    put_centered_text(
        frame,
        "ACCESS GRANTED",
        170,
        0.75,
        2,
        (255, 255, 255)
    )

    put_centered_text(
        frame,
        "Door will remain unlocked",
        215,
        0.65,
        2,
        (255, 255, 255)
    )

    cv2.putText(
        frame,
        "Q = Exit",
        (30, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


def show_challenge(
    frame,
    title,
    status,
    face=None,
    status_color=(255, 255, 255)
):
    draw_face_box(
        frame,
        face,
        (0, 255, 255)
    )

    cv2.putText(
        frame,
        "AUTHENTICATION",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    put_centered_text(
        frame,
        title,
        115,
        1.0,
        3,
        (255, 255, 255)
    )

    put_centered_text(
        frame,
        status,
        165,
        0.75,
        2,
        status_color
    )

    cv2.putText(
        frame,
        "Q = Cancel",
        (30, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA
    )


# ============================================================
# BLINK CHALLENGE
# ============================================================

def blink_challenge(cap, landmarker, detector):
    print("\n========================================")
    print("STEP 1/2 - BLINK")
    print("========================================")
    print("Blink once.")

    start = time.time()

    open_frames = 0
    closed_frames = 0
    baseline_ready = False

    while time.time() - start < BLINK_TIMEOUT:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = detect_face(
            detector,
            frame
        )

        if face is None:
            show_challenge(
                frame,
                "BLINK NOW",
                "Face not detected",
                None,
                (0, 0, 255)
            )
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

            continue

        landmarks = get_landmarks(
            landmarker,
            frame
        )

        ear = get_ear(
            landmarks,
            frame
        )

        if ear is not None:

            if not baseline_ready:
                if ear > EAR_OPEN_THRESHOLD:
                    open_frames += 1
                else:
                    open_frames = 0

                if open_frames >= OPEN_BASELINE_FRAMES:
                    baseline_ready = True

            else:
                if ear < EAR_CLOSED_THRESHOLD:
                    closed_frames += 1
                else:
                    if closed_frames >= BLINK_CLOSED_FRAMES:
                        print("[LIVENESS] Blink detected.")

                        show_challenge(
                            frame,
                            "BLINK VERIFIED",
                            "Get ready to turn your head",
                            face,
                            (0, 255, 0)
                        )

                        cv2.imshow(
                            WINDOW_NAME,
                            frame
                        )
                        cv2.waitKey(600)

                        return True

                    closed_frames = 0

        show_challenge(
            frame,
            "BLINK NOW",
            "Close and open your eyes once",
            face,
            (255, 255, 255)
        )

        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    print("[LIVENESS] Blink timeout.")
    return False


# ============================================================
# HEAD TURN CHALLENGE
# ============================================================

def head_turn_challenge(cap, landmarker, detector):
    print("\n========================================")
    print("STEP 2/2 - HEAD TURN")
    print("========================================")
    print("Turn your head LEFT or RIGHT.")

    start = time.time()

    while time.time() - start < HEAD_TURN_TIMEOUT:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = detect_face(
            detector,
            frame
        )

        if face is None:
            show_challenge(
                frame,
                "TURN HEAD",
                "Face not detected",
                None,
                (0, 0, 255)
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False

            continue

        landmarks = get_landmarks(
            landmarker,
            frame
        )

        yaw = estimate_head_yaw(
            landmarks,
            frame
        )

        direction_text = "LEFT or RIGHT"

        if yaw is not None:
            if yaw < -YAW_REQUIRED:
                direction_text = "LEFT DETECTED"

                print(
                    f"[LIVENESS] Head turned LEFT "
                    f"(yaw={yaw:.1f})"
                )

                show_challenge(
                    frame,
                    "HEAD TURN VERIFIED",
                    "LEFT detected",
                    face,
                    (0, 255, 0)
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )
                cv2.waitKey(600)

                return True

            if yaw > YAW_REQUIRED:
                direction_text = "RIGHT DETECTED"

                print(
                    f"[LIVENESS] Head turned RIGHT "
                    f"(yaw={yaw:.1f})"
                )

                show_challenge(
                    frame,
                    "HEAD TURN VERIFIED",
                    "RIGHT detected",
                    face,
                    (0, 255, 0)
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )
                cv2.waitKey(600)

                return True

        show_challenge(
            frame,
            "TURN HEAD",
            direction_text,
            face,
            (255, 255, 255)
        )

        if yaw is not None:
            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False

    print("[LIVENESS] Head-turn timeout.")
    return False


# ============================================================
# FACE RECOGNITION
# ============================================================

def load_known_faces():
    ensure_directories()

    known_faces = {}

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith(".npy"):
            continue

        path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        try:
            feature = np.load(path)

            name = os.path.splitext(
                filename
            )[0]

            known_faces[name] = feature

            print(
                f"[OK] Loaded authorized user: {name}"
            )

        except Exception as exc:
            print(
                f"[WARNING] Could not load "
                f"{filename}: {exc}"
            )

    return known_faces


def get_face_feature(
    recognizer,
    frame,
    face
):
    try:
        aligned = recognizer.alignCrop(
            frame,
            face
        )

        return recognizer.feature(
            aligned
        )

    except Exception as exc:
        print(
            f"[WARNING] Feature extraction failed: "
            f"{exc}"
        )
        return None


def recognize_average_feature(
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

        except Exception as exc:
            print(
                f"[WARNING] Recognition failed "
                f"for {name}: {exc}"
            )

    if not scores:
        return None, 0.0, 0.0, {}

    ordered = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    best_name = ordered[0][0]
    best_score = ordered[0][1]

    second_score = (
        ordered[1][1]
        if len(ordered) > 1
        else 0.0
    )

    margin = best_score - second_score

    return (
        best_name,
        best_score,
        margin,
        scores
    )


def recognition_step(
    cap,
    detector,
    recognizer,
    known_faces
):
    print("\n========================================")
    print("FACE RECOGNITION")
    print("========================================")

    samples = []
    start = time.time()

    while (
        len(samples) < RECOGNITION_SAMPLES
        and time.time() - start < RECOGNITION_TIMEOUT
    ):
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        face = detect_face(
            detector,
            frame
        )

        if face is None:
            show_challenge(
                frame,
                "RECOGNIZING",
                "Face not detected",
                None,
                (0, 0, 255)
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False, None, 0.0

            continue

        feature = get_face_feature(
            recognizer,
            frame,
            face
        )

        if feature is not None:
            samples.append(feature)

        show_challenge(
            frame,
            "RECOGNIZING",
            f"Analyzing {len(samples)}/{RECOGNITION_SAMPLES}",
            face,
            (0, 255, 255)
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        if cv2.waitKey(100) & 0xFF == ord("q"):
            return False, None, 0.0

    if not samples:
        print("[ACCESS DENIED] Recognition timeout.")
        return False, None, 0.0

    average_feature = np.mean(
        np.vstack(samples),
        axis=0,
        keepdims=True
    )

    best_name, best_score, margin, scores = (
        recognize_average_feature(
            recognizer,
            average_feature,
            known_faces
        )
    )

    print("\n[AUTH] Recognition scores:")

    for name, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"  {name}: {score:.4f}")

    print(
        f"\n[AUTH] Best match: {best_name}"
    )
    print(
        f"[AUTH] Match score: {best_score:.4f}"
    )
    print(
        f"[AUTH] Margin: {margin:.4f}"
    )

    if best_name is None:
        return False, None, best_score

    if best_score < RECOGNITION_THRESHOLD:
        print(
            f"[ACCESS DENIED] Score "
            f"{best_score:.4f} < "
            f"{RECOGNITION_THRESHOLD:.2f}"
        )

        log_event(
            f"ACCESS DENIED - UNKNOWN USER - "
            f"BEST={best_name} SCORE={best_score:.4f}"
        )

        return False, best_name, best_score

    # Only require a margin when multiple authorized
    # users are enrolled.
    if (
        len(scores) > 1
        and margin < RECOGNITION_MARGIN
    ):
        print(
            f"[ACCESS DENIED] Margin "
            f"{margin:.4f} < "
            f"{RECOGNITION_MARGIN:.2f}"
        )

        log_event(
            f"ACCESS DENIED - LOW MARGIN - "
            f"BEST={best_name} SCORE={best_score:.4f} "
            f"MARGIN={margin:.4f}"
        )

        return False, best_name, best_score

    return True, best_name, best_score


# ============================================================
# RESULT SCREENS
# ============================================================

def show_result(
    cap,
    granted,
    user_name=None,
    score=0.0
):
    start = time.time()

    while time.time() - start < RESULT_DISPLAY_SECONDS:
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        if granted:
            cv2.putText(
                frame,
                "ACCESS GRANTED",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                3,
                cv2.LINE_AA
            )

            put_centered_text(
                frame,
                "DOOR UNLOCKED",
                145,
                1.0,
                3,
                (0, 255, 0)
            )

            put_centered_text(
                frame,
                f"Welcome {user_name}",
                195,
                0.8,
                2,
                (255, 255, 255)
            )

        else:
            cv2.putText(
                frame,
                "ACCESS DENIED",
                (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

            put_centered_text(
                frame,
                "DOOR LOCKED",
                145,
                1.0,
                3,
                (0, 0, 255)
            )

            put_centered_text(
                frame,
                "Authentication failed",
                195,
                0.75,
                2,
                (255, 255, 255)
            )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


# ============================================================
# ONE COMPLETE AUTHENTICATION ATTEMPT
# ============================================================

def authenticate_once(
    cap,
    detector,
    recognizer,
    landmarker,
    known_faces
):
    # --------------------------------------------------------
    # STEP 1: BLINK
    # --------------------------------------------------------
    if not blink_challenge(
        cap,
        landmarker,
        detector
    ):
        print("\n[ACCESS DENIED] Blink challenge failed.")
        log_event(
            "ACCESS DENIED - BLINK CHALLENGE FAILED"
        )
        show_result(cap, False)
        return False, None

    # --------------------------------------------------------
    # STEP 2: HEAD TURN
    # --------------------------------------------------------
    if not head_turn_challenge(
        cap,
        landmarker,
        detector
    ):
        print(
            "\n[ACCESS DENIED] "
            "Head-turn challenge failed."
        )

        log_event(
            "ACCESS DENIED - HEAD TURN CHALLENGE FAILED"
        )

        show_result(cap, False)
        return False, None

    # --------------------------------------------------------
    # STEP 3: RECOGNITION
    # --------------------------------------------------------
    print("\n[LIVENESS] PASSED")
    print("[AUTH] Starting face recognition...")

    granted, user_name, score = recognition_step(
        cap,
        detector,
        recognizer,
        known_faces
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------
    if granted:
        print("\n========================================")
        print("          ACCESS GRANTED")
        print("========================================")
        print(f"Authorized user: {user_name}")
        print(f"Match score: {score:.4f}")
        print("[LIVENESS] Passed")
        print("[DOOR] UNLOCKED")
        print("========================================")

        log_event(
            f"ACCESS GRANTED - USER={user_name} "
            f"SCORE={score:.4f}"
        )

        show_result(
            cap,
            True,
            user_name,
            score
        )

        return True, user_name

    print("\n========================================")
    print("          ACCESS DENIED")
    print("========================================")
    print("[DOOR] LOCKED")
    print("========================================")

    show_result(
        cap,
        False,
        user_name,
        score
    )

    return False, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n")
    print("========================================")
    print("       AI FACE DOOR LOCK SYSTEM")
    print("========================================")

    ensure_directories()

    try:
        detector, recognizer, landmarker = load_models()
    except Exception as exc:
        print("\n[ERROR] Model loading failed.")
        print(exc)
        input("\nPress Enter to exit...")
        return

    known_faces = load_known_faces()

    if not known_faces:
        print("\n[ERROR] No authorized users found.")
        print("Run:")
        print("python enroll_user.py")
        input("\nPress Enter to exit...")
        return

    print(
        f"\n[OK] {len(known_faces)} "
        f"authorized user(s) loaded."
    )

    cap = open_camera()

    if cap is None:
        print("\n[ERROR] No working camera found.")
        input("\nPress Enter to exit...")
        return

    door_locked = True
    authenticated_user = None

    print("\n========================================")
    print("SYSTEM READY")
    print("========================================")
    print("DOOR STATUS: LOCKED")
    print("Look at the camera.")
    print("Press SPACE to start authentication.")
    print("Press Q to exit.")
    print()

    try:
        # ====================================================
        # LOCKED STATE
        # ====================================================
        while door_locked:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            face = detect_face(
                detector,
                frame
            )

            draw_face_box(
                frame,
                face
            )

            if face is None:
                show_locked(
                    frame,
                    "Look at the camera"
                )
            else:
                show_locked(
                    frame,
                    "Face detected - Press SPACE"
                )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n[EXIT] Program closed.")
                return

            # SPACE starts exactly ONE guided authentication.
            if key == 32 and face is not None:
                print("\n[AUTH] Starting authentication...")

                granted, user_name = authenticate_once(
                    cap,
                    detector,
                    recognizer,
                    landmarker,
                    known_faces
                )

                if granted:
                    # IMPORTANT:
                    # Never set this back to True automatically.
                    door_locked = False
                    authenticated_user = user_name

                    print("\n")
                    print("########################################")
                    print("#          DOOR IS UNLOCKED            #")
                    print("#                                      #")
                    print("#  NO MORE AUTHENTICATION REQUIRED     #")
                    print("#                                      #")
                    print("#  Press Q to exit                     #")
                    print("########################################")

                else:
                    # Stay locked.
                    print(
                        "\n[STATE] Door remains LOCKED."
                    )
                    print(
                        "[STATE] Press SPACE to try again "
                        "or Q to exit."
                    )

        # ====================================================
        # UNLOCKED STATE
        # ====================================================
        #
        # NOTHING is authenticated here.
        # No blink check.
        # No head-turn check.
        # No face recognition.
        #
        # The door stays unlocked until Q.
        # ====================================================
        while not door_locked:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            show_unlocked(
                frame,
                authenticated_user
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n========================================")
                print("Program terminated.")
                print(
                    "Door was left UNLOCKED "
                    "for this simulation."
                )
                print("========================================")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

        try:
            landmarker.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
