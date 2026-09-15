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

# ------------------------------------------------------------
# Recognition
# ------------------------------------------------------------
# We will print the actual scores so this can be tuned later.
RECOGNITION_THRESHOLD = 0.45
RECOGNITION_MARGIN = 0.08

# ------------------------------------------------------------
# Liveness
# ------------------------------------------------------------
YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
OPEN_AFTER_BLINK_FRAMES = 5

CHALLENGE_TIMEOUT = 15


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Media Foundation", cv2.CAP_MSMF),
        ("Default", cv2.CAP_ANY)
    ]

    print("\nSearching for a working camera...")

    for camera_index in range(6):

        for backend_name, backend in backends:

            print(
                f"  Trying camera {camera_index} "
                f"using {backend_name}..."
            )

            cap = cv2.VideoCapture(
                camera_index,
                backend
            )

            if not cap.isOpened():

                cap.release()
                continue

            time.sleep(0.3)

            working = False
            frame = None

            for _ in range(5):

                ret, test_frame = cap.read()

                if not ret:
                    continue

                if test_frame is None:
                    continue

                if test_frame.size == 0:
                    continue

                min_value = int(test_frame.min())
                max_value = int(test_frame.max())
                mean_value = float(test_frame.mean())

                # Reject black camera
                if max_value <= 5:
                    continue

                if mean_value <= 1:
                    continue

                frame = test_frame
                working = True
                break

            if working:

                print(
                    "\n[OK] Working camera found!"
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

    print(
        "\n[ERROR] No working webcam was found."
    )

    return None


# ============================================================
# FACE DETECTOR
# ============================================================

def load_detector():

    if not os.path.exists(DETECTOR_MODEL):

        print(
            f"[ERROR] Detector model not found:\n"
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

        print(
            "[OK] YuNet face detector loaded."
        )

        return detector

    except Exception as e:

        print(
            f"[ERROR] Could not load face detector:\n{e}"
        )

        return None


# ============================================================
# FACE RECOGNIZER
# ============================================================

def load_recognizer():

    if not os.path.exists(RECOGNITION_MODEL):

        print(
            f"[ERROR] Recognition model not found:\n"
            f"{RECOGNITION_MODEL}"
        )

        return None

    try:

        recognizer = cv2.FaceRecognizerSF.create(
            RECOGNITION_MODEL,
            ""
        )

        print(
            "[OK] SFace face recognizer loaded."
        )

        return recognizer

    except Exception as e:

        print(
            f"[ERROR] Could not load face recognizer:\n{e}"
        )

        return None


# ============================================================
# MEDIAPIPE LANDMARKER
# ============================================================

def load_landmarker():

    if not os.path.exists(LANDMARK_MODEL):

        print(
            f"[ERROR] Landmark model not found:\n"
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

        print(
            "[OK] MediaPipe face landmarker loaded."
        )

        return landmarker

    except Exception as e:

        print(
            f"[ERROR] Could not load MediaPipe landmarker:\n{e}"
        )

        return None


# ============================================================
# LOAD KNOWN FACES
# ============================================================

def load_known_faces():

    known_faces = {}

    os.makedirs(
        KNOWN_FACES_DIR,
        exist_ok=True
    )

    files = [
        f
        for f in os.listdir(KNOWN_FACES_DIR)
        if f.lower().endswith(".npy")
    ]

    if not files:

        print(
            "\n[WARNING] No authorized users found."
        )

        return known_faces

    for filename in files:

        path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        try:

            feature = np.load(
                path
            )

            feature = feature.astype(
                np.float32
            )

            # Flatten feature if necessary
            feature = feature.reshape(
                1,
                -1
            )

            # Normalize
            norm = np.linalg.norm(
                feature
            )

            if norm > 0:

                feature = feature / norm

            username = os.path.splitext(
                filename
            )[0]

            known_faces[username] = feature

            print(
                f"[OK] Loaded authorized user: "
                f"{username}"
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
# DISTANCE
# ============================================================

def distance(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


# ============================================================
# EYE ASPECT RATIO
# ============================================================

def calculate_ear(
    landmarks,
    indices
):

    try:

        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]

        vertical_1 = distance(
            p2,
            p6
        )

        vertical_2 = distance(
            p3,
            p5
        )

        horizontal = distance(
            p1,
            p4
        )

        if horizontal == 0:

            return 0

        return (
            vertical_1 +
            vertical_2
        ) / (
            2.0 *
            horizontal
        )

    except Exception:

        return 0


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
# MEDIAPIPE LANDMARKS
# ============================================================

def get_landmarks(
    landmarker,
    frame
):

    try:

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Correct MediaPipe API
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(
            mp_image
        )

        if not result.face_landmarks:

            return None

        face_landmarks = (
            result.face_landmarks[0]
        )

        height, width = frame.shape[:2]

        points = []

        for landmark in face_landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

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
# HEAD YAW
# ============================================================

def calculate_head_yaw(
    landmarks
):

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
            left_eye +
            right_eye
        ) / 2.0

        eye_distance = np.linalg.norm(
            right_eye -
            left_eye
        )

        if eye_distance == 0:

            return 0

        normalized_position = (
            nose[0] -
            eye_center[0]
        ) / eye_distance

        yaw = (
            normalized_position *
            60.0
        )

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

    height, width = frame.shape[:2]

    detector.setInputSize(
        (width, height)
    )

    try:

        _, faces = detector.detect(
            frame
        )

    except Exception as e:

        print(
            f"[WARNING] Face detection error: {e}"
        )

        return "UNKNOWN", 0.0

    if faces is None or len(faces) == 0:

        return "NO_FACE", 0.0

    # Largest face
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

        feature = feature.reshape(
            1,
            -1
        )

        norm = np.linalg.norm(
            feature
        )

        if norm == 0:

            return "UNKNOWN", 0.0

        feature = feature / norm

    except Exception as e:

        print(
            f"[WARNING] Feature extraction error: {e}"
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
                (
                    username,
                    float(score)
                )
            )

        except Exception as e:

            print(
                f"[WARNING] Match error "
                f"for {username}: {e}"
            )

    if not scores:

        return "UNKNOWN", 0.0

    # Highest score first
    scores.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # --------------------------------------------------------
    # PRINT SCORES
    # --------------------------------------------------------

    print(
        "\n[RECOGNITION SCORES]"
    )

    for username, score in scores:

        print(
            f"  {username}: {score:.4f}"
        )

    # --------------------------------------------------------
    # Best match
    # --------------------------------------------------------

    best_name = scores[0][0]
    best_score = scores[0][1]

    if len(scores) > 1:

        second_score = scores[1][1]

    else:

        second_score = 0.0

    margin = (
        best_score -
        second_score
    )

    print(
        f"  Best: {best_name}"
    )

    print(
        f"  Best score: {best_score:.4f}"
    )

    print(
        f"  Margin: {margin:.4f}"
    )

    print(
        f"  Required threshold: "
        f"{RECOGNITION_THRESHOLD:.4f}"
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    if best_score >= RECOGNITION_THRESHOLD:

        # If only one authorized user exists
        if len(scores) == 1:

            return (
                best_name,
                best_score
            )

        # Multiple users require margin
        if margin >= RECOGNITION_MARGIN:

            return (
                best_name,
                best_score
            )

    return (
        "UNKNOWN",
        best_score
    )


# ============================================================
# MAIN
# ============================================================

def start_door_lock():

    print("\n")
    print("=" * 60)
    print("           AI FACE DOOR LOCK")
    print("=" * 60)

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector = load_detector()

    if detector is None:

        return

    # --------------------------------------------------------
    # Load recognizer
    # --------------------------------------------------------

    recognizer = load_recognizer()

    if recognizer is None:

        return

    # --------------------------------------------------------
    # Load landmark model
    # --------------------------------------------------------

    landmarker = load_landmarker()

    if landmarker is None:

        return

    # --------------------------------------------------------
    # Load users
    # --------------------------------------------------------

    known_faces = load_known_faces()

    if not known_faces:

        print(
            "\n[INFO] No authorized users are enrolled."
        )

        print(
            "Please enroll a user first."
        )

        try:
            landmarker.close()
        except Exception:
            pass

        return

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    cap = open_camera()

    if cap is None:

        try:
            landmarker.close()
        except Exception:
            pass

        return

    # --------------------------------------------------------
    # State
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

    last_recognition_time = 0

    # Prevent recognition from running
    # unnecessarily many times
    RECOGNITION_INTERVAL = 0.5

    print("\n")
    print("=" * 60)
    print("DOOR LOCK ACTIVE")
    print("=" * 60)
    print("Look at the camera.")
    print("Blink and turn your head when requested.")
    print("Press Q to exit.")
    print("=" * 60)

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:

            print(
                "[WARNING] Could not read camera frame."
            )

            continue

        height, width = frame.shape[:2]

        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        detector.setInputSize(
            (width, height)
        )

        try:

            _, faces = detector.detect(
                frame
            )

        except Exception:

            faces = None

        status = "LOOKING FOR FACE"

        # ----------------------------------------------------
        # NO FACE
        # ----------------------------------------------------

        if faces is None or len(faces) == 0:

            status = "NO FACE DETECTED"

            blink_closed_count = 0
            open_frames = 0

            recognized_name = "UNKNOWN"
            recognized_score = 0.0

            challenge_started = False
            first_turn_detected = False
            blink_detected = False
            liveness_passed = False

        else:

            # ------------------------------------------------
            # Largest face
            # ------------------------------------------------

            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, w, h = map(
                int,
                face[:4]
            )

            x = max(
                0,
                x
            )

            y = max(
                0,
                y
            )

            w = min(
                w,
                width - x
            )

            h = min(
                h,
                height - y
            )

            # ------------------------------------------------
            # Draw face
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 255, 255),
                2
            )

            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            current_time = time.time()

            if (
                current_time -
                last_recognition_time
                >= RECOGNITION_INTERVAL
            ):

                (
                    recognized_name,
                    recognized_score
                ) = recognize_face(
                    detector,
                    recognizer,
                    frame,
                    known_faces
                )

                last_recognition_time = (
                    current_time
                )

            # ------------------------------------------------
            # Landmarks
            # ------------------------------------------------

            landmarks = get_landmarks(
                landmarker,
                frame
            )

            yaw = 0.0
            ear = 0.0

            if landmarks is not None:

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
                # Blink
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
                # Open eyes after blink
                # --------------------------------------------

                if (
                    blink_detected
                    and ear > EAR_OPEN_THRESHOLD
                ):

                    open_frames += 1

                # --------------------------------------------
                # Head yaw
                # --------------------------------------------

                yaw = calculate_head_yaw(
                    landmarks
                )

            # ------------------------------------------------
            # Start challenge
            # ------------------------------------------------

            if (
                recognized_name != "UNKNOWN"
                and recognized_name != "NO_FACE"
                and recognized_name != "NO_USERS"
                and not challenge_started
                and not liveness_passed
                and not access_granted
            ):

                challenge_started = True

                challenge_start_time = (
                    time.time()
                )

                required_direction = (
                    np.random.choice(
                        ["LEFT", "RIGHT"]
                    )
                )

                blink_detected = False
                blink_closed_count = 0
                open_frames = 0
                first_turn_detected = False

                print(
                    f"\n[LIVENESS] Turn your head "
                    f"{required_direction}"
                )

            # ------------------------------------------------
            # LIVENESS
            # ------------------------------------------------

            if (
                challenge_started
                and not liveness_passed
            ):

                elapsed = (
                    time.time() -
                    challenge_start_time
                )

                if elapsed > CHALLENGE_TIMEOUT:

                    print(
                        "\n[LIVENESS] "
                        "Challenge timed out."
                    )

                    challenge_started = False

                    blink_detected = False
                    blink_closed_count = 0
                    open_frames = 0
                    first_turn_detected = False

                else:

                    # ----------------------------------------
                    # Blink first
                    # ----------------------------------------

                    if not blink_detected:

                        status = "BLINK"

                    # ----------------------------------------
                    # Head movement
                    # ----------------------------------------

                    elif not first_turn_detected:

                        status = (
                            "TURN HEAD "
                            + required_direction
                        )

                        if (
                            required_direction
                            == "LEFT"
                        ):

                            if yaw < -YAW_REQUIRED:

                                first_turn_detected = True

                        else:

                            if yaw > YAW_REQUIRED:

                                first_turn_detected = True

                    # ----------------------------------------
                    # Liveness success
                    # ----------------------------------------

                    if (
                        blink_detected
                        and first_turn_detected
                        and open_frames >=
                        OPEN_AFTER_BLINK_FRAMES
                    ):

                        liveness_passed = True

                        challenge_started = False

                        print(
                            "\n[LIVENESS] PASSED"
                        )

            # ------------------------------------------------
            # ACCESS DECISION
            # ------------------------------------------------

            if liveness_passed:

                if (
                    recognized_name != "UNKNOWN"
                    and recognized_name != "NO_FACE"
                ):

                    access_granted = True

                    unlock_time = (
                        time.time()
                    )

                    print(
                        f"\n[ACCESS GRANTED] "
                        f"{recognized_name}"
                    )

                    print(
                        f"[MATCH SCORE] "
                        f"{recognized_score:.4f}"
                    )

                    print(
                        "[DOOR] UNLOCKED"
                    )

                    log_access(
                        recognized_name,
                        "GRANTED"
                    )

                else:

                    print(
                        "\n[ACCESS DENIED]"
                    )

                    print(
                        f"[BEST SCORE] "
                        f"{recognized_score:.4f}"
                    )

                    log_access(
                        "UNKNOWN",
                        "DENIED"
                    )

                # Reset challenge
                liveness_passed = False
                blink_detected = False
                blink_closed_count = 0
                open_frames = 0
                first_turn_detected = False

        # ====================================================
        # DOOR TIMER
        # ====================================================

        if access_granted:

            elapsed_unlock = (
                time.time() -
                unlock_time
            )

            if elapsed_unlock >= 5:

                access_granted = False

                recognized_name = "UNKNOWN"
                recognized_score = 0.0

                blink_detected = False
                blink_closed_count = 0
                open_frames = 0
                first_turn_detected = False

                print(
                    "\n[DOOR] Locked again."
                )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # User
        # ----------------------------------------------------

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
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Score: {recognized_score:.3f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Blink
        # ----------------------------------------------------

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
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Door
        # ----------------------------------------------------

        if access_granted:

            door_text = "DOOR: UNLOCKED"

        else:

            door_text = "DOOR: LOCKED"

        cv2.putText(
            frame,
            door_text,
            (20, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Challenge
        # ----------------------------------------------------

        if challenge_started:

            if not blink_detected:

                challenge_text = "BLINK"

            elif not first_turn_detected:

                challenge_text = (
                    "TURN "
                    + required_direction
                )

            else:

                challenge_text = "PROCESSING..."

            cv2.putText(
                frame,
                challenge_text,
                (20, height - 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Camera window
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
# ENTRY POINT
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