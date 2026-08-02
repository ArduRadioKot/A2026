import time
import threading
import cv2

from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


# ==========================
# НАСТРОЙКИ
# ==========================

STREAM_NAME = "pioneer"

TARGET_MARKER_ID = 5

CAMERA_ANGLE = -80
VIDEO_FPS = 20


# ==========================
# СОСТОЯНИЕ
# ==========================

running = True

camera = None
viewer = None


marker_visible = False
marker_center = None
marker_id_found = None


marker_lock = threading.Lock()


# ==========================
# ARUCO
# ==========================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# ==========================
# КАМЕРА + СТРИМ + ARUCO
# ==========================

def video_worker():

    global camera
    global viewer

    global marker_visible
    global marker_center
    global marker_id_found


    try:

        print("[VIDEO] Подключение камеры...")

        camera = Camera()
        viewer = ImageViewer()


        print()
        print("=" * 50)
        print("Стрим запущен")
        print(
            f"http://10.42.0.1:8889/{STREAM_NAME}"
        )
        print("=" * 50)
        print()


        while running:


            frame = camera.get_cv_frame(
                timeout=2.0
            )


            if frame is None:
                continue


            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )


            corners, ids, _ = (
                aruco_detector.detectMarkers(gray)
            )


            current_visible = False
            current_center = None
            current_id = None


            if ids is not None:


                for marker_corners, marker_id in zip(
                    corners,
                    ids.flatten()
                ):


                    marker_id = int(marker_id)


                    points = (
                        marker_corners
                        .reshape((4,2))
                        .astype(int)
                    )


                    cv2.polylines(
                        frame,
                        [points],
                        True,
                        (0,255,0),
                        2
                    )


                    cx = int(
                        points[:,0].mean()
                    )

                    cy = int(
                        points[:,1].mean()
                    )


                    cv2.circle(
                        frame,
                        (cx,cy),
                        5,
                        (0,0,255),
                        -1
                    )


                    cv2.putText(
                        frame,
                        f"ARUCO ID: {marker_id}",
                        (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2
                    )


                    if marker_id == TARGET_MARKER_ID:

                        current_visible = True
                        current_center = (
                            cx,
                            cy
                        )

                        current_id = marker_id


                        cv2.putText(
                            frame,
                            "TARGET FOUND",
                            (cx+10,cy+30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2
                        )



            with marker_lock:

                marker_visible = current_visible
                marker_center = current_center
                marker_id_found = current_id



            cv2.putText(
                frame,
                f"SEARCH ID: {TARGET_MARKER_ID}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )


            status = (
                "FOUND"
                if current_visible
                else "NOT FOUND"
            )


            cv2.putText(
                frame,
                status,
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )


            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=VIDEO_FPS
            )


    except Exception as e:

        print(
            "[VIDEO ERROR]",
            e
        )


    finally:


        print("[VIDEO] Остановка")


        if camera:

            try:
                camera.stop()

            except:
                pass


        if viewer:

            try:
                viewer.close()

            except:
                pass



# ==========================
# ЗАПУСК
# ==========================

servo_camera = ServoCamera()

if servo_camera.set_angle(CAMERA_ANGLE):
    print(
        "[CAMERA] Угол установлен"
    )


thread = threading.Thread(
    target=video_worker,
    daemon=True
)


thread.start()



try:

    while True:

        with marker_lock:

            if marker_visible:

                print(
                    "[ARUCO] Найден ID:",
                    marker_id_found,
                    "центр:",
                    marker_center
                )


        time.sleep(0.2)



except KeyboardInterrupt:


    print(
        "\nОстановка"
    )


finally:

    running = False

    thread.join(
        timeout=3
    )

    print(
        "[END]"
    )