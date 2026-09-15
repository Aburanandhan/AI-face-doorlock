import cv2

backends = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY)
]

for name, backend in backends:

    print("\n==============================")
    print("Testing:", name)
    print("==============================")

    cap = cv2.VideoCapture(0, backend)

    print("Opened:", cap.isOpened())

    if not cap.isOpened():
        print("Could not open")
        continue

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    for i in range(10):
        ret, frame = cap.read()

        if ret and frame is not None:
            print(
                "Frame", i,
                "| mean =", frame.mean(),
                "| min =", frame.min(),
                "| max =", frame.max()
            )
        else:
            print("Frame", i, "FAILED")

    cap.release()

print("\nDiagnostic finished.")
