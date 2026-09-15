import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 1

CHALLENGE_TIME = 12

# Minimum head rotation required
YAW_CHANGE_REQUIRED = 15.0

# Blink settings
EAR_OPEN_THRESHOLD = 0.23
EAR_CLOSED_THRESHOLD = 0.19

BLINK_CLOSED_FRAMES = 3
BLINK_OPEN_FRAMES = 5

# ============================================================
# MEDIAPIPE
# ============================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/face_landmarker.task"

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("======================================")
print("   AI DOOR LOCK - LIVENESS TEST")
print("======================================")
print("Camera started")
print()

# ============================================================
# HEAD POSE MODEL
# ============================================================

# MediaPipe landmark IDs
# 1   = nose
# 152 = chin
# 33  = right eye outer
# 263 = left eye outer
# 61  = mouth left
# 291 = mouth right

MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -63.6, -12.5),      # Chin
    (-43.3, 32.7, -26.0),     # Right eye
    (43.3, 32.7, -26.0),      # Left eye
    (-28.9, -28.9, -24.1),    # Mouth left
    (28.9, -28.9, -24.1)      # Mouth right
], dtype=np.float64)


def get_head_pose(landmarks, width, height):

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

    distortion = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate Euler angles
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

        pitch = math.degrees(
            math.atan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )
        )

        yaw = math.degrees(
            math.atan2(
                -rotation_matrix[2, 0],
                sy
            )
        )

        roll = 0

    return pitch, yaw, roll


# ============================================================
# EAR
# ============================================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def distance(p1, p2):

    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )


def calculate_ear(landmarks, eye):

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
    ) / (2 * horizontal)


# ============================================================
# RANDOM CHALLENGE
# ============================================================

challenge = random.choice([
    "LEFT",
    "RIGHT"
])

print(f"🎯 Challenge: TURN HEAD {challenge}")
print("First keep your face straight.")
print("Then turn your head when instructed.")
print()

# ============================================================
# STATES
# ============================================================

state = "GET_BASELINE"

baseline_yaws = []

baseline_yaw = None

blink_closed = 0
blink_open = 0

blink_done = False
head_turn_done = False

challenge_start = time.time()

start_time = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # Mirror only the displayed image
    display = cv2.flip(frame, 1)

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

    status = "NO FACE"
    status_color = (0, 0, 255)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        height, width = frame.shape[:2]

        pose = get_head_pose(
            landmarks,
            width,
            height
        )

        if pose is not None:

            pitch, yaw, roll = pose

            ear_left = calculate_ear(
                landmarks,
                LEFT_EYE
            )

            ear_right = calculate_ear(
                landmarks,
                RIGHT_EYE
            )

            ear = (
                ear_left + ear_right
            ) / 2

            # ==================================================
            # STEP 1 - GET STABLE BASELINE
            # ==================================================

            if state == "GET_BASELINE":

                status = "KEEP HEAD STRAIGHT"
                status_color = (0, 255, 255)

                # Only accept reasonable straight-head values
                if abs(yaw) < 15:

                    baseline_yaws.append(yaw)

                if len(baseline_yaws) >= 20:

                    baseline_yaw = float(
                        np.median(
                            baseline_yaws
                        )
                    )

                    state = "BLINK"

                    print(
                        f"✅ Baseline head position: "
                        f"{baseline_yaw:.1f}°"
                    )

                    print(
                        "👉 BLINK ONCE"
                    )

            # ==================================================
            # STEP 2 - BLINK
            # ==================================================

            elif state == "BLINK":

                status = "BLINK ONCE"
                status_color = (0, 255, 255)

                if ear < EAR_CLOSED_THRESHOLD:

                    blink_closed += 1

                else:

                    if blink_closed >= BLINK_CLOSED_FRAMES:

                        if ear > EAR_OPEN_THRESHOLD:

                            blink_open += 1

                    else:

                        blink_closed = 0

                if (
                    blink_closed >= BLINK_CLOSED_FRAMES
                    and
                    ear > EAR_OPEN_THRESHOLD
                ):

                    blink_done = True

                    state = "TURN"

                    print("✅ Blink completed")
                    print(
                        f"👉 TURN HEAD {challenge}"
                    )

            # ==================================================
            # STEP 3 - HEAD TURN
            # ==================================================

            elif state == "TURN":

                status = f"TURN HEAD {challenge}"
                status_color = (0, 255, 255)

                yaw_change = yaw - baseline_yaw

                # Determine direction.
                #
                # LEFT and RIGHT are based on actual
                # head pose, not image movement.

                if challenge == "LEFT":

                    if yaw_change > YAW_CHANGE_REQUIRED:

                        head_turn_done = True

                else:

                    if yaw_change < -YAW_CHANGE_REQUIRED:

                        head_turn_done = True

                # ==================================================
                # LIVENESS PASSED
                # ==================================================

                if head_turn_done and blink_done:

                    state = "PASSED"

                    print()
                    print(
                        "======================================"
                    )
                    print(
                        "✅ REAL PERSON - LIVENESS PASSED"
                    )
                    print(
                        "======================================"
                    )
                    print()

            # ==================================================
            # PASSED
            # ==================================================

            elif state == "PASSED":

                status = "LIVENESS PASSED"
                status_color = (0, 255, 0)

            # ==================================================
            # DISPLAY HEAD POSE
            # ==================================================

            cv2.putText(
                display,
                f"Yaw: {yaw:.1f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"EAR: {ear:.3f}",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            # Draw face landmarks
            for point in landmarks:

                x = int(
                    point.x * display.shape[1]
                )

                y = int(
                    point.y * display.shape[0]
                )

                cv2.circle(
                    display,
                    (display.shape[1] - x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

    # ============================================================
    # TIMER
    # ============================================================

    elapsed = time.time() - challenge_start

    if state != "PASSED" and elapsed > CHALLENGE_TIME:

        status = "CHALLENGE FAILED"
        status_color = (0, 0, 255)

        cv2.putText(
            display,
            "PRESS R TO RETRY",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # ============================================================
    # STATUS
    # ============================================================

    cv2.putText(
        display,
        status,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )

    cv2.putText(
        display,
        "Q = Quit | R = Restart",
        (20, display.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "AI Door Lock - Liveness Test",
        display
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # ============================================================
    # RESTART
    # ============================================================

    if key == ord("r"):

        challenge = random.choice([
            "LEFT",
            "RIGHT"
        ])

        state = "GET_BASELINE"

        baseline_yaws = []
        baseline_yaw = None

        blink_closed = 0
        blink_open = 0

        blink_done = False
        head_turn_done = False

        challenge_start = time.time()

        print()
        print(
            f"🔄 New challenge: TURN HEAD {challenge}"
        )


cap.release()
cv2.destroyAllWindows()