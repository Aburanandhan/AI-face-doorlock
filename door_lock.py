import cv2
import numpy as np
import os
import time
import math
from datetime import datetime

import mediapipe as mp
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

# Recognition settings
RECOGNITION_THRESHOLD = 0.45
RECOGNITION_MARGIN = 0.08

# Liveness settings
YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
OPEN_BASELINE_FRAMES = 15
OPEN_AFTER_BLINK_FRAMES = 5

CHALLENGE_TIMEOUT = 15


# ============================================================
# CAMERA DETECTION
# ============================================================

def open_camera():
    """
    Automatically search for a working webcam.

    Rejects cameras that technically open but only return
    black/empty frames.
    """

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY),
    ]

    print("\nSearching for a working camera...")

    for camera_index in range(6):

        for backend_name, backend in backends:

            print(
                f"  Trying camera {camera_index} "
                f"using {backend_name}..."
            )

            cap = cv2.VideoCapture(camera_index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            # Give the camera a moment to initialize
            time.sleep(0.3)

            working = False

            for _ in range(5):

                ret, frame = cap.read()

                if not ret or frame is None:
                    continue

                if frame.size == 0:
                    continue

                # Check whether frame actually contains image data
                min_value = int(frame.min())
                max_value = int(frame.max())
                mean_value = float(frame.mean())

                # Reject completely black/empty frames
                if max_value <= 5 or mean_value <= 1:
                    continue

                working = True
                break

            if working:

                print(
                    f"\n[OK] Working camera found!"
                )
                print(
                    f"     Camera index : {camera_index}"
                )
                print(
                    f"     Backend      : {backend_name}"
                )
                print(
                    f"     Resolution   : "
                    f"{frame.shape[1]}x{frame.shape[0]}"
                )

                return cap

            cap.release()

    print("\n[ERROR] No working webcam was found.")
    return None


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

def load_detector():

    if not os.path.exists(DETECTOR_MODEL):
        print(
            f"[ERROR] Face detector model not found:\n"
            f"{DETECTOR_MODEL}"
        )
        return None

    try:

        detector = cv2.FaceDetectorYN.create(
            DETECTOR_MODEL,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

        print("[OK] YuNet face detector loaded.")

        return detector

    except Exception as e:

        print(
            f"[ERROR] Could not load face detector:\n{e}"
        )

        return None


# ============================================================
# LOAD FACE RECOGNIZER
# ============================================================

def load_recognizer():

    if not os.path.exists(RECOGNITION_MODEL):
        print(
            f"[ERROR] Face recognition model not found:\n"
            f"{RECOGNITION_MODEL}"
        )
        return None

    try:

        recognizer = cv2.FaceRecognizerSF.create(
            RECOGNITION_MODEL,
            ""
        )

        print("[OK] SFace face recognizer loaded.")

        return recognizer

    except Exception as e:

        print(
            f"[ERROR] Could not load face recognizer:\n{e}"
        )

        return None


# ============================================================
# LOAD MEDIAPIPE LANDMARKER
# ============================================================

def load_landmarker():

    if not os.path.exists(LANDMARK_MODEL):
        print(
            f"[ERROR] Face landmark model not found:\n"
            f"{LANDMARK_MODEL}"
        )
        return None

    try:

        base_options = python.BaseOptions(
            model_asset_path=LANDMARK_MODEL
        )

        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        landmarker = vision.FaceLandmarker.create_from_options(
            options
        )

        print("[OK] MediaPipe face landmarker loaded.")

        return landmarker

    except Exception as e:

        print(
            f"[ERROR] Could not load MediaPipe landmarker:\n{e}"
        )

        return None


# ============================================================
# LOAD AUTHORIZED USERS
# ============================================================

def load_known_faces():

    known_faces = {}

    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

    files = [
        f
        for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith(".npy")
    ]

    if not files:

        print("\n[WARNING] No authorized users found.")
        print(
            "Please use option 1 from the menu "
            "to enroll a user first.\n"
        )

        return known_faces

    for filename in files:

        path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        try:

            feature = np.load(path)

            feature = feature.astype(
                np.float32
            )

            # Normalize feature
            norm = np.linalg.norm(feature)

            if norm > 0:
                feature = feature / norm

            username = os.path.splitext(
                filename
            )[0]

            known_faces[username] = feature

            print(
                f"[OK] Loaded authorized user: {username}"
            )

        except Exception as e:

            print(
                f"[WARNING] Could not load "
                f"{filename}: {e}"
            )

    print(
        f"\nTotal authorized users: "
        f"{len(known_faces)}"
    )

    return known_faces


# ============================================================
# LOGGING
# ============================================================

def log_access(username, status):

    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        with open(
            LOG_FILE,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"{timestamp} | "
                f"{username} | "
                f"{status}\n"
            )

    except Exception as e:

        print(
            f"[WARNING] Could not write log: {e}"
        )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_ear(landmarks, indices):

    try:

        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        a = distance(p2, p6)
        b = distance(p3, p5)
        c = distance(p1, p4)

        if c == 0:
            return 0

        ear = (a + b) / (2.0 * c)

        return ear

    except Exception:
        return 0


# MediaPipe Face Mesh-style eye landmark indices
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


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def get_landmarks(landmarker, frame):

    try:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Correct MediaPipe image creation
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(
            mp_image
        )

        if not result.face_landmarks:
            return None

        face_landmarks = result.face_landmarks[0]

        h, w = frame.shape[:2]

        points = []

        for landmark in face_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            points.append(
                (x, y)
            )

        return points

    except Exception as e:

        print(
            f"[WARNING] Landmark error: {e}"
        )

        return None


# ============================================================
# HEAD YAW ESTIMATION
# ============================================================

def calculate_head_yaw(landmarks):

    """
    Estimate left/right head movement using
    normalized nose position between the eyes.
    """

    try:

        left_eye = np.mean(
            np.array([
                landmarks[33],
                landmarks[133],
                landmarks[160],
                landmarks[159]
            ]),
            axis=0
        )

        right_eye = np.mean(
            np.array([
                landmarks[362],
                landmarks[263],
                landmarks[385],
                landmarks[386]
            ]),
            axis=0
        )

        nose = np.array(
            landmarks[1]
        )

        eye_center = (
            left_eye + right_eye
        ) / 2.0

        eye_distance = np.linalg.norm(
            right_eye - left_eye
        )

        if eye_distance == 0:
            return 0

        normalized_position = (
            nose[0] - eye_center[0]
        ) / eye_distance

        # Convert normalized position into
        # approximate degrees
        yaw = normalized_position * 60.0

        return yaw

    except Exception:

        return 0


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize_face(
    detector,
    recognizer,
    frame,
    known_faces
):

    if not known_faces:
        return "NO_USERS", 0.0

    h, w = frame.shape[:2]

    detector.setInputSize(
        (w, h)
    )

    try:

        _, faces = detector.detect(
            frame
        )

    except Exception:

        return "UNKNOWN", 0.0

    if faces is None or len(faces) == 0:

        return "NO_FACE", 0.0

    # Select largest face
    face = max(
        faces,
        key=lambda f: f[2] * f[3]
    )

    try:

        aligned_face = recognizer.alignCrop(
            frame,
            face
        )

        feature = recognizer.feature(
            aligned_face
        )

        feature = feature.astype(
            np.float32
        )

        norm = np.linalg.norm(feature)

        if norm == 0:

            return "UNKNOWN", 0.0

        feature = feature / norm

    except Exception as e:

        print(
            f"[WARNING] Recognition error: {e}"
        )

        return "UNKNOWN", 0.0

    scores = []

    for username, known_feature in known_faces.items():

        try:

            score = recognizer.match(
                feature,
                known_feature,
                cv2.FaceRecognizerSF_FR_COSINE
            )

            scores.append(
                (username, float(score))
            )

        except Exception:
            continue

    if not scores:

        return "UNKNOWN", 0.0

    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    best_name, best_score = scores[0]

    # Check second-best score
    second_score = (
        scores[1][1]
        if len(scores) > 1
        else 0.0
    )

    margin = (
        best_score - second_score
    )

    # Recognition decision
    if best_score >= RECOGNITION_THRESHOLD:

        if len(scores) == 1 or margin >= RECOGNITION_MARGIN:

            return best_name, best_score

    return "UNKNOWN", best_score


# ============================================================
# DRAW TEXT
# ============================================================

def draw_text(
    frame,
    text,
    position,
    scale=0.7,
    thickness=2
):

    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# MAIN DOOR LOCK SYSTEM
# ============================================================

def start_door_lock():

    print("\n")
    print("=" * 60)
    print("           AI FACE DOOR LOCK")
    print("=" * 60)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    detector = load_detector()

    if detector is None:
        return

    recognizer = load_recognizer()

    if recognizer is None:
        return

    landmarker = load_landmarker()

    if landmarker is None:
        return

    # --------------------------------------------------------
    # Load authorized users
    # --------------------------------------------------------

    known_faces = load_known_faces()

    if not known_faces:

        print(
            "\n[INFO] No authorized users are enrolled."
        )

        print(
            "\nPlease choose:"
        )
        print(
            "1. Enroll a new user"
        )
        print(
            "2. Return to menu"
        )

        choice = input(
            "\nEnter choice: "
        ).strip()

        return

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:
        return

    # --------------------------------------------------------
    # Liveness state
    # --------------------------------------------------------

    blink_closed_count = 0

    blink_detected = False

    open_frames = 0

    challenge_started = False

    challenge_start_time = 0

    required_direction = None

    first_turn_detected = False

    liveness_passed = False

    recognized_name = "UNKNOWN"

    recognized_score = 0.0

    access_granted = False

    unlock_time = 0

    # --------------------------------------------------------
    # Challenge direction
    # --------------------------------------------------------

    challenge_directions = [
        "LEFT",
        "RIGHT"
    ]

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DOOR LOCK ACTIVE")
    print("=" * 60)
    print("Look at the camera.")
    print("Press Q to exit.")
    print("=" * 60)

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "[WARNING] Could not read camera frame."
            )

            time.sleep(0.1)

            continue

        # ----------------------------------------------------
        # Resize if necessary
        # ----------------------------------------------------

        frame_height, frame_width = frame.shape[:2]

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        detector.setInputSize(
            (frame_width, frame_height)
        )

        try:

            _, faces = detector.detect(
                frame
            )

        except Exception:

            faces = None

        # ----------------------------------------------------
        # Default display
        # ----------------------------------------------------

        status = "LOOKING FOR FACE"

        status_color = (
            255,
            255,
            255
        )

        # ----------------------------------------------------
        # No face
        # ----------------------------------------------------

        if faces is None or len(faces) == 0:

            status = "NO FACE DETECTED"

            # Reset temporary liveness state
            blink_closed_count = 0
            open_frames = 0
            first_turn_detected = False
            liveness_passed = False

            recognized_name = "UNKNOWN"
            recognized_score = 0.0

        else:

            # ------------------------------------------------
            # Select largest face
            # ------------------------------------------------

            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, w, h = map(
                int,
                face[:4]
            )

            # Keep rectangle inside image
            x = max(0, x)
            y = max(0, y)

            w = min(
                w,
                frame_width - x
            )

            h = min(
                h,
                frame_height - y
            )

            # ------------------------------------------------
            # Draw face box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Get landmarks
            # ------------------------------------------------

            landmarks = get_landmarks(
                landmarker,
                frame
            )

            yaw = 0.0
            ear = 0.0

            if landmarks is not None:

                # --------------------------------------------
                # Calculate eye aspect ratio
                # --------------------------------------------

                left_ear = calculate_ear(
                    landmarks,
                    LEFT_EYE
                )

                right_ear = calculate_ear(
                    landmarks,
                    RIGHT_EYE
                )

                ear = (
                    left_ear +
                    right_ear
                ) / 2.0

                # --------------------------------------------
                # Blink detection
                # --------------------------------------------

                if ear < EAR_CLOSED_THRESHOLD:

                    blink_closed_count += 1

                else:

                    if (
                        blink_closed_count
                        >= BLINK_CLOSED_FRAMES
                    ):

                        blink_detected = True

                    blink_closed_count = 0

                # --------------------------------------------
                # Open eye baseline
                # --------------------------------------------

                if (
                    ear > EAR_OPEN_THRESHOLD
                    and not blink_detected
                ):

                    open_frames += 1

                else:

                    if blink_detected:

                        open_frames += 1

                # --------------------------------------------
                # Head yaw
                # --------------------------------------------

                yaw = calculate_head_yaw(
                    landmarks
                )

            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            recognized_name, recognized_score = recognize_face(
                detector,
                recognizer,
                frame,
                known_faces
            )

            # ------------------------------------------------
            # Start liveness challenge
            # ------------------------------------------------

            if (
                recognized_name != "UNKNOWN"
                and recognized_name != "NO_FACE"
                and recognized_name != "NO_USERS"
                and not challenge_started
                and not liveness_passed
            ):

                challenge_started = True

                challenge_start_time = time.time()

                required_direction = np.random.choice(
                    challenge_directions
                )

                first_turn_detected = False

                print(
                    f"\n[LIVENESS] Turn your head "
                    f"{required_direction}"
                )

            # ------------------------------------------------
            # Liveness challenge
            # ------------------------------------------------

            if challenge_started and not liveness_passed:

                elapsed = (
                    time.time()
                    - challenge_start_time
                )

                if elapsed > CHALLENGE_TIMEOUT:

                    print(
                        "\n[LIVENESS] Challenge timed out."
                    )

                    challenge_started = False

                    blink_detected = False
                    open_frames = 0
                    first_turn_detected = False

                else:

                    # ----------------------------------------
                    # Blink must happen
                    # ----------------------------------------

                    if not blink_detected:

                        status = "BLINK"

                    # ----------------------------------------
                    # After blink, require head turn
                    # ----------------------------------------

                    elif not first_turn_detected:

                        status = (
                            "TURN HEAD "
                            + required_direction
                        )

                        if required_direction == "LEFT":

                            if yaw < -YAW_REQUIRED:

                                first_turn_detected = True

                        elif required_direction == "RIGHT":

                            if yaw > YAW_REQUIRED:

                                first_turn_detected = True

                    # ----------------------------------------
                    # Liveness completed
                    # ----------------------------------------

                    if (
                        blink_detected
                        and first_turn_detected
                        and open_frames
                        >= OPEN_AFTER_BLINK_FRAMES
                    ):

                        liveness_passed = True

                        challenge_started = False

                        print(
                            "\n[LIVENESS] PASSED"
                        )

            # ------------------------------------------------
            # Access decision
            # ------------------------------------------------

            if liveness_passed:

                if recognized_name != "UNKNOWN":

                    status = "ACCESS GRANTED"

                    access_granted = True

                    unlock_time = time.time()

                    log_access(
                        recognized_name,
                        "GRANTED"
                    )

                    print(
                        f"\n[ACCESS GRANTED] "
                        f"{recognized_name}"
                    )

                    print(
                        "Door unlocked."
                    )

                    # ------------------------------------------------
                    # Display unlock for 5 seconds
                    # ------------------------------------------------

                    liveness_passed = False

                else:

                    status = "ACCESS DENIED"

                    access_granted = False

                    log_access(
                        "UNKNOWN",
                        "DENIED"
                    )

                    print(
                        "\n[ACCESS DENIED]"
                    )

                    # Reset
                    blink_detected = False
                    open_frames = 0
                    first_turn_detected = False
                    liveness_passed = False

            # ------------------------------------------------
            # Unlock timer
            # ------------------------------------------------

            if access_granted:

                elapsed_unlock = (
                    time.time()
                    - unlock_time
                )

                if elapsed_unlock >= 5:

                    access_granted = False

                    recognized_name = "UNKNOWN"

                    recognized_score = 0.0

                    blink_detected = False
                    open_frames = 0
                    first_turn_detected = False

                    print(
                        "\nDoor locked again."
                    )

        # ====================================================
        # DISPLAY INFORMATION
        # ====================================================

        # Status
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
            cv2.LINE_AA
        )

        # Recognition information
        if recognized_name not in [
            "UNKNOWN",
            "NO_FACE",
            "NO_USERS"
        ]:

            cv2.putText(
                frame,
                f"User: {recognized_name}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Similarity: {recognized_score:.3f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Liveness information
        blink_text = (
            "YES"
            if blink_detected
            else "NO"
        )

        cv2.putText(
            frame,
            f"Blink: {blink_text}",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Door state
        if access_granted:

            door_text = "DOOR: UNLOCKED"

        else:

            door_text = "DOOR: LOCKED"

        cv2.putText(
            frame,
            door_text,
            (20, frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Challenge instruction
        if challenge_started:

            if required_direction:

                challenge_text = (
                    f"TURN {required_direction}"
                )

                cv2.putText(
                    frame,
                    challenge_text,
                    (
                        20,
                        frame_height - 65
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

        # ----------------------------------------------------
        # Show camera
        # ----------------------------------------------------

        cv2.imshow(
            "AI Face Door Lock",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "\nStopping door lock..."
            )

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

    print(
        "\nDoor lock stopped."
    )


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        start_door_lock()

    except KeyboardInterrupt:

        print(
            "\n\nProgram interrupted by user."
        )

    except Exception as e:

        print(
            "\n[ERROR] Unexpected error:"
        )

        print(
            str(e)
        )

        import traceback

        traceback.print_exc()

    finally:

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print(
            "\nPress Enter to return..."
        )

        input()