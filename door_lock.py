import cv2
import numpy as np
import os
import time
import math
import random
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

CAMERA_INDEX = 1

# IMPORTANT:
# Start with 0.50 instead of the old 0.363.
# We can calibrate this later using recognition_test.py.
RECOGNITION_THRESHOLD = 0.50

# Liveness
YAW_REQUIRED = 15.0

EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
OPEN_BASELINE_FRAMES = 15
OPEN_AFTER_BLINK_FRAMES = 5

CHALLENGE_TIMEOUT = 15


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    DETECTOR_MODEL,
    RECOGNITION_MODEL,
    LANDMARK_MODEL
]

for file in required_files:

    if not os.path.exists(file):

        print(f"❌ Missing file: {file}")
        exit()


if not os.path.exists(KNOWN_FACES_DIR):

    print("❌ known_faces folder not found")
    exit()


# ============================================================
# LOAD YUNET
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
# LOAD SFACE
# ============================================================

print("Loading SFace...")

recognizer = cv2.FaceRecognizerSF.create(
    RECOGNITION_MODEL,
    ""
)

print("✅ SFace loaded")


# ============================================================
# LOAD MEDIAPIPE FACE LANDMARKER
# ============================================================

print("Loading MediaPipe Face Landmarker...")

base_options = python.BaseOptions(
    model_asset_path=LANDMARK_MODEL
)

landmarker_options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(
    landmarker_options
)

print("✅ Face Landmarker loaded")


# ============================================================
# LOAD AUTHORIZED USERS
# ============================================================

authorized_faces = {}

print()
print("Loading authorized users...")
print()

for file in os.listdir(KNOWN_FACES_DIR):

    if not file.lower().endswith(".npy"):
        continue

    path = os.path.join(
        KNOWN_FACES_DIR,
        file
    )

    name = os.path.splitext(file)[0]

    try:

        feature = np.load(path)

        authorized_faces[name] = feature

        print(f"✅ Loaded: {name}")

    except Exception as e:

        print(
            f"❌ Failed to load {file}: {e}"
        )


if len(authorized_faces) == 0:

    print()
    print("❌ No authorized faces found.")
    print("Run enroll_user.py first.")
    exit()


print()
print("======================================")
print("AUTHORIZED USERS")
print("======================================")

for name in authorized_faces:

    print("👤", name)

print("======================================")
print()


# ============================================================
# LOGGING
# ============================================================

os.makedirs(
    "logs",
    exist_ok=True
)


def log_access(
    name,
    status,
    similarity
):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
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
            f"Similarity: {similarity:.4f}\n"
        )


# ============================================================
# EAR FUNCTIONS
# ============================================================

LEFT_EYE = [
    362,
    385,
    387,
    263,
    373,
    380
]

RIGHT_EYE = [
    33,
    160,
    158,
    133,
    153,
    144
]


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(
    landmarks,
    eye
):

    vertical1 = distance(
        landmarks[eye[1]],
        landmarks[eye[5]]
    )

    vertical2 = distance(
        landmarks[eye[2]],
        landmarks[eye[4]]
    )

    horizontal = distance(
        landmarks[eye[0]],
        landmarks[eye[3]]
    )

    if horizontal == 0:

        return 0

    return (
        vertical1 + vertical2
    ) / (
        2 * horizontal
    )


# ============================================================
# HEAD POSE
# ============================================================

MODEL_POINTS = np.array([

    (0.0, 0.0, 0.0),           # Nose
    (0.0, -63.6, -12.5),       # Chin
    (-43.3, 32.7, -26.0),      # Right eye
    (43.3, 32.7, -26.0),       # Left eye
    (-28.9, -28.9, -24.1),     # Mouth left
    (28.9, -28.9, -24.1)       # Mouth right

], dtype=np.float64)


def get_head_pose(
    landmarks,
    width,
    height
):

    image_points = np.array([

        (
            landmarks[1].x * width,
            landmarks[1].y * height
        ),

        (
            landmarks[152].x * width,
            landmarks[152].y * height
        ),

        (
            landmarks[33].x * width,
            landmarks[33].y * height
        ),

        (
            landmarks[263].x * width,
            landmarks[263].y * height
        ),

        (
            landmarks[61].x * width,
            landmarks[61].y * height
        ),

        (
            landmarks[291].x * width,
            landmarks[291].y * height
        )

    ], dtype=np.float64)

    focal_length = width

    camera_matrix = np.array([

        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]

    ], dtype=np.float64)

    distortion = np.zeros(
        (4, 1)
    )

    success, rotation_vector, translation_vector = cv2.solvePnP(

        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE

    )

    if not success:

        return None

    rotation_matrix, _ = cv2.Rodrigues(
        rotation_vector
    )

    sy = math.sqrt(

        rotation_matrix[0, 0] ** 2 +
        rotation_matrix[1, 0] ** 2

    )

    if sy > 1e-6:

        pitch = math.degrees(
            math.atan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )
        )

        yaw = math.degrees(
            math.atan2(
                -rotation_matrix[2, 0],
                sy
            )
        )

        roll = math.degrees(
            math.atan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )
        )

    else:

        pitch = 0
        yaw = 0
        roll = 0

    return pitch, yaw, roll


# ============================================================
# LIVENESS STATE RESET
# ============================================================

def reset_liveness():

    challenge = random.choice([
        "LEFT",
        "RIGHT"
    ])

    return {

        "state": "BASELINE",

        "challenge": challenge,

        "baseline_yaws": [],

        "baseline_yaw": None,

        "blink_closed": 0,

        "blink_done": False,

        "head_turn_done": False,

        "start_time": time.time(),

        "passed": False

    }


liveness = reset_liveness()


# ============================================================
# CAMERA
# ============================================================

print("Opening camera...")

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("❌ Camera could not be opened")
    exit()


print("✅ Camera started")
print()
print("Press Q to quit.")
print("Press R to restart authentication.")
print()

# ============================================================
# SYSTEM STATE
# ============================================================

door_unlocked = False

recognized_name = "Unknown"

best_similarity = 0.0

last_log_time = 0

last_status = ""

authentication_done = False

start_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("❌ Failed to read camera")
        break

    h, w = frame.shape[:2]

    # --------------------------------------------------------
    # DISPLAY MIRRORED
    # --------------------------------------------------------

    display = cv2.flip(
        frame,
        1
    )

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    detector.setInputSize(
        (w, h)
    )

    _, faces = detector.detect(
        frame
    )

    # Default state
    door_unlocked = False

    recognized_name = "Unknown"

    best_similarity = 0.0

    # ========================================================
    # NO FACE
    # ========================================================

    if faces is None or len(faces) == 0:

        cv2.putText(
            display,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Restart liveness
        liveness = reset_liveness()

        authentication_done = False

    # ========================================================
    # MULTIPLE FACES
    # ========================================================

    elif len(faces) > 1:

        cv2.putText(
            display,
            "MULTIPLE FACES - ACCESS DENIED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        for face in faces:

            x, y, fw, fh = (
                face[:4].astype(int)
            )

            cv2.rectangle(
                display,
                (x, y),
                (x + fw, y + fh),
                (0, 0, 255),
                2
            )

        door_unlocked = False

        liveness = reset_liveness()

        authentication_done = False

    # ========================================================
    # ONE FACE
    # ========================================================

    else:

        face = faces[0]

        x, y, fw, fh = (
            face[:4].astype(int)
        )

        # ====================================================
        # LIVENESS
        # ====================================================

        if not liveness["passed"]:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(
                (time.time() - start_time) * 1000
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            liveness_status = "CHECKING LIVENESS"

            liveness_color = (
                0,
                255,
                255
            )

            if result.face_landmarks:

                landmarks = result.face_landmarks[0]

                pose = get_head_pose(
                    landmarks,
                    w,
                    h
                )

                if pose is not None:

                    pitch, yaw, roll = pose

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
                    ) / 2

                    state = liveness["state"]

                    # ========================================
                    # BASELINE
                    # ========================================

                    if state == "BASELINE":

                        liveness_status = (
                            "KEEP HEAD STRAIGHT"
                        )

                        if abs(yaw) < 15:

                            liveness[
                                "baseline_yaws"
                            ].append(yaw)

                        if len(
                            liveness["baseline_yaws"]
                        ) >= OPEN_BASELINE_FRAMES:

                            liveness[
                                "baseline_yaw"
                            ] = float(
                                np.median(
                                    liveness[
                                        "baseline_yaws"
                                    ]
                                )
                            )

                            liveness[
                                "state"
                            ] = "BLINK"

                    # ========================================
                    # BLINK
                    # ========================================

                    elif state == "BLINK":

                        liveness_status = (
                            "BLINK ONCE"
                        )

                        if ear < EAR_CLOSED_THRESHOLD:

                            liveness[
                                "blink_closed"
                            ] += 1

                        else:

                            if (
                                liveness[
                                    "blink_closed"
                                ] >= BLINK_CLOSED_FRAMES
                                and
                                ear > EAR_OPEN_THRESHOLD
                            ):

                                liveness[
                                    "blink_done"
                                ] = True

                                liveness[
                                    "state"
                                ] = "TURN"

                                print(
                                    "✅ Blink detected"
                                )

                                print(
                                    "👉 Turn head "
                                    +
                                    liveness[
                                        "challenge"
                                    ]
                                )

                            liveness[
                                "blink_closed"
                            ] = 0

                    # ========================================
                    # HEAD TURN
                    # ========================================

                    elif state == "TURN":

                        challenge = (
                            liveness[
                                "challenge"
                            ]
                        )

                        liveness_status = (
                            f"TURN HEAD {challenge}"
                        )

                        baseline = (
                            liveness[
                                "baseline_yaw"
                            ]
                        )

                        yaw_change = (
                            yaw - baseline
                        )

                        # LEFT / RIGHT
                        if challenge == "LEFT":

                            if (
                                yaw_change
                                >
                                YAW_REQUIRED
                            ):

                                liveness[
                                    "head_turn_done"
                                ] = True

                        else:

                            if (
                                yaw_change
                                <
                                -YAW_REQUIRED
                            ):

                                liveness[
                                    "head_turn_done"
                                ] = True

                        # ====================================
                        # PASSED
                        # ====================================

                        if (
                            liveness[
                                "blink_done"
                            ]
                            and
                            liveness[
                                "head_turn_done"
                            ]
                        ):

                            liveness[
                                "passed"
                            ] = True

                            liveness[
                                "state"
                            ] = "PASSED"

                            print()
                            print(
                                "================================"
                            )
                            print(
                                "✅ LIVENESS PASSED"
                            )
                            print(
                                "================================"
                            )
                            print()

                    # ========================================
                    # PASSED
                    # ========================================

                    elif state == "PASSED":

                        liveness_status = (
                            "LIVENESS PASSED"
                        )

                        liveness_color = (
                            0,
                            255,
                            0
                        )

                    # Display values
                    cv2.putText(
                        display,
                        f"Yaw: {yaw:.1f}",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"EAR: {ear:.3f}",
                        (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            # ------------------------------------------------
            # DRAW LIVENESS STATUS
            # ------------------------------------------------

            cv2.putText(
                display,
                liveness_status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                liveness_color,
                2
            )

            cv2.rectangle(
                display,
                (x, y),
                (x + fw, y + fh),
                (0, 255, 255),
                2
            )

        # ====================================================
        # RECOGNITION
        # ====================================================

        else:

            # ------------------------------------------------
            # ALIGN
            # ------------------------------------------------

            aligned_face = recognizer.alignCrop(
                frame,
                face
            )

            # ------------------------------------------------
            # FEATURE
            # ------------------------------------------------

            feature = recognizer.feature(
                aligned_face
            )

            # ------------------------------------------------
            # COMPARE AGAINST EVERY USER
            # ------------------------------------------------

            best_name = "Unknown"

            best_score = -1.0

            for name, known_feature in authorized_faces.items():

                score = recognizer.match(
                    known_feature,
                    feature,
                    cv2.FaceRecognizerSF_FR_COSINE
                )

                print(
                    f"{name}: {score:.4f}"
                )

                if score > best_score:

                    best_score = score

                    best_name = name

            best_similarity = best_score

            # =================================================
            # AUTHORIZATION
            # =================================================

            if (
                best_score >=
                RECOGNITION_THRESHOLD
            ):

                recognized_name = best_name

                door_unlocked = True

                box_color = (
                    0,
                    255,
                    0
                )

                status = "ACCESS GRANTED"

                cv2.putText(
                    display,
                    "ACCESS GRANTED",
                    (
                        x,
                        max(y - 35, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    display,
                    f"USER: {best_name}",
                    (
                        x,
                        y + fh + 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:

                recognized_name = "Unknown"

                door_unlocked = False

                box_color = (
                    0,
                    0,
                    255
                )

                status = "ACCESS DENIED"

                cv2.putText(
                    display,
                    "ACCESS DENIED",
                    (
                        x,
                        max(y - 35, 25)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    display,
                    "UNKNOWN USER",
                    (
                        x,
                        y + fh + 25
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            # ------------------------------------------------
            # FACE BOX
            # ------------------------------------------------

            cv2.rectangle(
                display,
                (x, y),
                (x + fw, y + fh),
                box_color,
                3
            )

            # ------------------------------------------------
            # LOG
            # ------------------------------------------------

            current_time = time.time()

            if (
                current_time -
                last_log_time
                >= 2
            ):

                log_access(
                    recognized_name,
                    status,
                    best_similarity
                )

                last_log_time = (
                    current_time
                )


    # ========================================================
    # DOOR STATUS
    # ========================================================

    if door_unlocked:

        door_text = "DOOR: UNLOCKED"

        door_color = (
            0,
            255,
            0
        )

    else:

        door_text = "DOOR: LOCKED"

        door_color = (
            0,
            0,
            255
        )

    cv2.putText(
        display,
        door_text,
        (
            20,
            h - 50
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        door_color,
        3
    )

    # ========================================================
    # SIMILARITY
    # ========================================================

    cv2.putText(
        display,
        f"Similarity: {best_similarity:.3f}",
        (
            20,
            h - 15
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "AI Digital Door Lock",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    # ========================================================
    # QUIT
    # ========================================================

    if key == ord("q"):

        break

    # ========================================================
    # RESTART
    # ========================================================

    if key == ord("r"):

        print()
        print("🔄 Authentication restarted")

        liveness = reset_liveness()

        authentication_done = False

        door_unlocked = False

        recognized_name = "Unknown"

        best_similarity = 0.0


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("======================================")
print("AI Door Lock stopped.")
print("======================================")