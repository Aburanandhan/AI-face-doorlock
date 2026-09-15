import cv2

print("Testing camera with DirectShow...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    print("Frame mean:", frame.mean(), end="\r")

    cv2.imshow("DirectShow Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nCamera test finished.")
