import mediapipe as mp

print("MediaPipe version:", mp.__version__)
print("Has tasks:", hasattr(mp, "tasks"))
print("Has Image:", hasattr(mp, "Image"))
print("MediaPipe Tasks API is available!")
