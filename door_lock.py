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
# UI - CAMERA LEFT / CONTROL PANEL RIGHT
# ============================================================

PANEL_WIDTH = 390
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 600

# OpenCV uses BGR.
WHITE = (245, 245, 245)
MUTED = (165, 175, 185)
CYAN = (255, 220, 80)
GREEN = (80, 220, 120)
RED = (80, 80, 235)
AMBER = (60, 190, 255)
DARK_PANEL = (28, 32, 38)
DARKER = (20, 23, 28)
BORDER = (55, 62, 72)


def fit_camera(frame):
    return cv2.resize(
        frame,
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        interpolation=cv2.INTER_AREA
    )


def draw_panel_card(canvas, x, y, w, h):
    cv2.rectangle(
        canvas,
        (x, y),
        (x + w, y + h),
        BORDER,
        1,
        cv2.LINE_AA
    )
    cv2.rectangle(
        canvas,
        (x + 1, y + 1),
        (x + w - 1, y + h - 1),
        DARK_PANEL,
        -1
    )


def panel_text(
    canvas,
    text,
    x,
    y,
    scale=0.62,
    color=WHITE,
    thickness=1
):
    cv2.putText(
        canvas,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def panel_header(canvas, title, subtitle=None):
    panel_text(
        canvas,
        title,
        24,
        48,
        0.82,
        WHITE,
        2
    )

    if subtitle:
        panel_text(
            canvas,
            subtitle,
            24,
            75,
            0.48,
            MUTED,
            1
        )

    cv2.line(
        canvas,
        (24, 92),
        (PANEL_WIDTH - 24, 92),
        BORDER,
        1,
        cv2.LINE_AA
    )


def make_layout(camera_frame):
    camera = fit_camera(camera_frame)

    canvas = np.full(
        (CAMERA_HEIGHT, CAMERA_WIDTH + PANEL_WIDTH, 3),
        DARKER,
        dtype=np.uint8
    )

    canvas[:, :CAMERA_WIDTH] = camera

    # Camera border.
    cv2.rectangle(
        canvas,
        (0, 0),
        (CAMERA_WIDTH - 1, CAMERA_HEIGHT - 1),
        BORDER,
        1
    )

    # Right panel.
    cv2.rectangle(
        canvas,
        (CAMERA_WIDTH, 0),
        (CAMERA_WIDTH + PANEL_WIDTH, CAMERA_HEIGHT),
        DARKER,
        -1
    )

    cv2.line(
        canvas,
        (CAMERA_WIDTH, 0),
        (CAMERA_WIDTH, CAMERA_HEIGHT),
        BORDER,
        2
    )

    return canvas


def show_panel(
    frame,
    title="DOOR LOCKED",
    subtitle="AI SECURITY SYSTEM",
    status="Waiting for authentication",
    status_color=AMBER,
    step=0,
    step1="Blink",
    step1_done=False,
    step2="Turn head",
    step2_done=False,
    user_name=None,
    score=None,
    unlocked=False,
    instruction="SPACE  Authenticate",
    extra=None
):
    canvas = make_layout(frame)

    x = CAMERA_WIDTH + 1

    # Header
    panel_header(
        canvas[:, x:],
        title,
        subtitle
    )

    # Status card
    draw_panel_card(
        canvas,
        x + 18,
        115,
        PANEL_WIDTH - 36,
        100
    )

    panel_text(
        canvas,
        "STATUS",
        x + 36,
        143,
        0.45,
        MUTED,
        1
    )

    panel_text(
        canvas,
        status,
        x + 36,
        177,
        0.63,
        status_color,
        2
    )

    if extra:
        panel_text(
            canvas,
            extra,
            x + 36,
            198,
            0.43,
            MUTED,
            1
        )

    # Authentication card
    draw_panel_card(
        canvas,
        x + 18,
        232,
        PANEL_WIDTH - 36,
        190
    )

    panel_text(
        canvas,
        "AUTHENTICATION",
        x + 36,
        260,
        0.58,
        WHITE,
        2
    )

    # Step 1
    if step1_done:
        icon1 = "✓"
        color1 = GREEN
        detail1 = "Verified"
    elif step == 1:
        icon1 = "●"
        color1 = AMBER
        detail1 = "Blink once"
    else:
        icon1 = "○"
        color1 = MUTED
        detail1 = "Waiting"

    panel_text(
        canvas,
        icon1,
        x + 38,
        304,
        0.72,
        color1,
        2
    )
    panel_text(
        canvas,
        "01  " + step1,
        x + 70,
        304,
        0.58,
        WHITE if step == 1 else MUTED,
        2
    )
    panel_text(
        canvas,
        detail1,
        x + 70,
        327,
        0.43,
        color1,
        1
    )

    # Step 2
    if step2_done:
        icon2 = "✓"
        color2 = GREEN
        detail2 = "Verified"
    elif step == 2:
        icon2 = "●"
        color2 = AMBER
        detail2 = "Turn LEFT or RIGHT"
    else:
        icon2 = "○"
        color2 = MUTED
        detail2 = "Waiting"

    panel_text(
        canvas,
        icon2,
        x + 38,
        367,
        0.72,
        color2,
        2
    )
    panel_text(
        canvas,
        "02  " + step2,
        x + 70,
        367,
        0.58,
        WHITE if step == 2 else MUTED,
        2
    )
    panel_text(
        canvas,
        detail2,
        x + 70,
        390,
        0.43,
        color2,
        1
    )

    # Recognition result
    if user_name is not None or score is not None:
        draw_panel_card(
            canvas,
            x + 18,
            438,
            PANEL_WIDTH - 36,
            82
        )

        panel_text(
            canvas,
            "FACE MATCH",
            x + 36,
            463,
            0.43,
            MUTED,
            1
        )

        if user_name is not None:
            panel_text(
                canvas,
                str(user_name),
                x + 36,
                495,
                0.62,
                GREEN if unlocked else RED,
                2
            )

        if score is not None:
            panel_text(
                canvas,
                f"Score  {score:.3f}",
                x + 185,
                495,
                0.48,
                WHITE,
                1
            )

    # Bottom instruction
    if unlocked:
        bottom = "Q  Exit"
    else:
        bottom = instruction

    panel_text(
        canvas,
        bottom,
        x + 24,
        CAMERA_HEIGHT - 25,
        0.52,
        WHITE,
        2
    )

    return canvas


def show_locked(frame, message="Look at the camera"):
    return show_panel(
        frame,
        title="DOOR LOCKED",
        subtitle="AI SECURITY SYSTEM",
        status=message,
        status_color=AMBER,
        step=0,
        instruction="SPACE  Authenticate     Q  Exit"
    )


def show_unlocked(frame, user_name):
    return show_panel(
        frame,
        title="DOOR UNLOCKED",
        subtitle="ACCESS CONTROL",
        status="ACCESS GRANTED",
        status_color=GREEN,
        step=0,
        step1="Liveness",
        step1_done=True,
        step2="Recognition",
        step2_done=True,
        user_name=user_name,
        unlocked=True,
        instruction="Q  Exit"
    )


def show_challenge(
    frame,
    title,
    status,
    face=None,
    status_color=WHITE
):
    if title == "BLINK NOW":
        step = 1
        step1_done = False
        step2_done = False
        auth_title = "AUTHENTICATION"
    elif title == "BLINK VERIFIED":
        step = 2
        step1_done = True
        step2_done = False
        auth_title = "AUTHENTICATION"
    elif title == "HEAD TURN VERIFIED":
        step = 3
        step1_done = True
        step2_done = True
        auth_title = "AUTHENTICATION"
    elif title == "RECOGNIZING":
        step = 3
        step1_done = True
        step2_done = True
        auth_title = "RECOGNITION"
    else:
        step = 0
        step1_done = False
        step2_done = False
        auth_title = "AUTHENTICATION"

    draw_face_box(
        frame,
        face,
        (0, 255, 255)
    )

    return show_panel(
        frame,
        title=auth_title,
        subtitle="GUIDED VERIFICATION",
        status=status,
        status_color=status_color,
        step=step,
        step1="Blink",
        step1_done=step1_done,
        step2="Turn head",
        step2_done=step2_done,
        instruction="Q  Cancel"
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

        face = detect_face(detector, frame)

        if face is None:
            display = show_challenge(
                frame,
                "BLINK NOW",
                "Face not detected",
                None,
                RED
            )
            cv2.imshow(WINDOW_NAME, display)

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

                        display = show_challenge(
                            frame,
                            "BLINK VERIFIED",
                            "Get ready to turn your head",
                            face,
                            GREEN
                        )
                        cv2.imshow(WINDOW_NAME, display)
                        cv2.waitKey(700)

                        return True

                    closed_frames = 0

        display = show_challenge(
            frame,
            "BLINK NOW",
            "Close and open your eyes once",
            face,
            AMBER
        )

        cv2.imshow(
            WINDOW_NAME,
            display
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

        face = detect_face(detector, frame)

        if face is None:
            display = show_challenge(
                frame,
                "TURN HEAD",
                "Face not detected",
                None,
                RED
            )
            cv2.imshow(WINDOW_NAME, display)

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

        if yaw is not None:
            if yaw < -YAW_REQUIRED:
                print(
                    f"[LIVENESS] Head turned LEFT "
                    f"(yaw={yaw:.1f})"
                )

                display = show_challenge(
                    frame,
                    "HEAD TURN VERIFIED",
                    "LEFT detected",
                    face,
                    GREEN
                )
                cv2.imshow(WINDOW_NAME, display)
                cv2.waitKey(700)

                return True

            if yaw > YAW_REQUIRED:
                print(
                    f"[LIVENESS] Head turned RIGHT "
                    f"(yaw={yaw:.1f})"
                )

                display = show_challenge(
                    frame,
                    "HEAD TURN VERIFIED",
                    "RIGHT detected",
                    face,
                    GREEN
                )
                cv2.imshow(WINDOW_NAME, display)
                cv2.waitKey(700)

                return True

        display = show_challenge(
            frame,
            "TURN HEAD",
            "Turn LEFT or RIGHT",
            face,
            AMBER
        )

        if yaw is not None:
            # Small camera-side diagnostic, not center-screen text.
            cv2.putText(
                display,
                f"Yaw: {yaw:.1f}",
                (25, CAMERA_HEIGHT - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                MUTED,
                1,
                cv2.LINE_AA
            )

        cv2.imshow(
            WINDOW_NAME,
            display
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
            name = os.path.splitext(filename)[0]
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
            display = show_challenge(
                frame,
                "RECOGNIZING",
                "Face not detected",
                None,
                RED
            )

            cv2.imshow(
                WINDOW_NAME,
                display
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

        display = show_challenge(
            frame,
            "RECOGNIZING",
            f"Analyzing {len(samples)}/{RECOGNITION_SAMPLES}",
            face,
            CYAN
        )

        cv2.imshow(
            WINDOW_NAME,
            display
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

    print(f"\n[AUTH] Best match: {best_name}")
    print(f"[AUTH] Match score: {best_score:.4f}")
    print(f"[AUTH] Margin: {margin:.4f}")

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
            display = show_panel(
                frame,
                title="DOOR UNLOCKED",
                subtitle="ACCESS CONTROL",
                status="ACCESS GRANTED",
                status_color=GREEN,
                step=0,
                step1="Blink",
                step1_done=True,
                step2="Head turn",
                step2_done=True,
                user_name=user_name,
                score=score,
                unlocked=True,
                instruction="Q  Exit"
            )
        else:
            display = show_panel(
                frame,
                title="DOOR LOCKED",
                subtitle="ACCESS CONTROL",
                status="ACCESS DENIED",
                status_color=RED,
                step=0,
                step1="Blink",
                step1_done=True,
                step2="Head turn",
                step2_done=True,
                user_name=user_name,
                score=score,
                instruction="R  Try again     Q  Exit"
            )

        cv2.imshow(
            WINDOW_NAME,
            display
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

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )
    cv2.resizeWindow(
        WINDOW_NAME,
        CAMERA_WIDTH + PANEL_WIDTH,
        CAMERA_HEIGHT
    )

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

            if face is None:
                display = show_panel(
                    frame,
                    title="DOOR LOCKED",
                    subtitle="AI SECURITY SYSTEM",
                    status="Look at the camera",
                    status_color=AMBER,
                    step=0,
                    instruction="SPACE  Authenticate     Q  Exit"
                )
            else:
                draw_face_box(
                    frame,
                    face
                )

                display = show_panel(
                    frame,
                    title="DOOR LOCKED",
                    subtitle="AI SECURITY SYSTEM",
                    status="Face detected",
                    status_color=GREEN,
                    step=0,
                    instruction="SPACE  Authenticate     Q  Exit"
                )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n[EXIT] Program closed.")
                return

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
                    # The door NEVER automatically relocks.
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
        while not door_locked:
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            display = show_unlocked(
                frame,
                authenticated_user
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n========================================")
                print("Program terminated.")
                print("Door was left UNLOCKED for this simulation.")
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
