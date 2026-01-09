import cv2

# OpenCV >= 4.5 推荐方式
if __name__ == '__main__':
    tracker = cv2.TrackerKCF_create()

    cap = cv2.VideoCapture("./video/basket.mp4")
    # cap = cv2.VideoCapture(0)  # 摄像头

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("无法读取视频")

    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    tracker.init(frame, bbox)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, bbox = tracker.update(frame)

        if ok:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "KCF Tracking", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Tracking failure", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

