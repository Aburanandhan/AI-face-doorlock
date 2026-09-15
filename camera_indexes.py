import cv2

print("Testing camera indexes...\n")

for i in range(6):
    print(f"===== Camera Index {i} =====")

    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    print("Opened:", cap.isOpened())

    if cap.isOpened():
        ret, frame = cap.read()

        print("Frame:", ret)

        if frame is not None:
            print("Shape:", frame.shape)
            print("Min:", frame.min())
            print("Max:", frame.max())
            print("Mean:", frame.mean())

    cap.release()
    print()