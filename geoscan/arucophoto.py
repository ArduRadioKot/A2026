import time
import threading
import cv2
import os
from datetime import datetime

from pioneer_sdk2 import Pioneer
from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"

PHOTO_DIR = "aruco_photos"

CAMERA_ANGLE = -80

running = True


# Создаём папку для фотографий
os.makedirs(
    PHOTO_DIR,
    exist_ok=True
)


# ============================================================
# ARUCO
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# Уже найденные ID
aruco_ids_set = set()

# ID в порядке обнаружения
aruco_ids = []

aruco_lock = threading.Lock()


# ============================================================
# СОХРАНЕНИЕ ПОЛНОГО КАДРА
# ============================================================

def save_aruco_photo(frame, marker_id):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]

    filename = (
        f"aruco_{marker_id}_{timestamp}.jpg"
    )

    filepath = os.path.join(
        PHOTO_DIR,
        filename
    )

    success = cv2.imwrite(
        filepath,
        frame
    )

    if success:

        print()
        print("=" * 60)
        print("[PHOTO] НОВЫЙ ARUCO")
        print(f"ArUco ID: {marker_id}")
        print("[PHOTO] Полный кадр сохранён")
        print(f"[PHOTO] {filepath}")
        print("=" * 60)
        print()

        return filepath

    print(
        f"[PHOTO ERROR] Не удалось сохранить ArUco {marker_id}"
    )

    return None


# ============================================================
# ВИДЕО
# ============================================================

def video_worker():

    global running

    camera = None
    viewer = None

    try:

        print("[VIDEO] Подключение камеры...")

        camera = Camera()
        viewer = ImageViewer()

        print()
        print("=" * 60)
        print("Видеопоток запущен")
        print()
        print("Открой в браузере:")
        print(
            f"http://10.42.0.1:8889/{STREAM_NAME}"
        )
        print()
        print(
            f"Фотографии сохраняются в: {PHOTO_DIR}"
        )
        print("=" * 60)
        print()


        while running:

            # ====================================================
            # ПОЛУЧАЕМ КАДР
            # ====================================================

            frame = camera.get_cv_frame(
                timeout=2.0
            )

            if frame is None:

                print("[VIDEO] Нет кадра")

                continue


            # Сохраняем чистую копию полного кадра.
            # Именно она будет записываться на диск.
            original_frame = frame.copy()


            # ====================================================
            # ПОИСК ARUCO
            # ====================================================

            gray = cv2.cvtColor(
                original_frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, rejected = (
                aruco_detector.detectMarkers(
                    gray
                )
            )


            # ====================================================
            # ОБРАБОТКА МЕТОК
            # ====================================================

            if ids is not None:

                for marker_corners, marker_id in zip(
                    corners,
                    ids.flatten()
                ):

                    marker_id = int(
                        marker_id
                    )


                    # ================================================
                    # КООРДИНАТЫ МЕТКИ
                    # ================================================

                    points = (
                        marker_corners
                        .reshape((4, 2))
                        .astype(int)
                    )

                    top_left = points[0]
                    top_right = points[1]
                    bottom_right = points[2]
                    bottom_left = points[3]


                    # ================================================
                    # ПРОВЕРЯЕМ, НОВАЯ ЛИ МЕТКА
                    # ================================================

                    new_marker = False

                    with aruco_lock:

                        if marker_id not in aruco_ids_set:

                            aruco_ids_set.add(
                                marker_id
                            )

                            aruco_ids.append(
                                marker_id
                            )

                            new_marker = True


                    # ================================================
                    # НОВЫЙ ID -> СОХРАНЯЕМ ПОЛНЫЙ КАДР
                    # ================================================

                    if new_marker:

                        print(
                            f"[ARUCO] Новый уникальный ID: {marker_id}"
                        )

                        print(
                            f"[ARUCO] Все найденные ID: {aruco_ids}"
                        )

                        save_aruco_photo(
                            original_frame,
                            marker_id
                        )


                    # ================================================
                    # РИСУЕМ РАМКУ ARUCO
                    # ================================================

                    cv2.line(
                        frame,
                        tuple(top_left),
                        tuple(top_right),
                        (0, 255, 0),
                        3
                    )

                    cv2.line(
                        frame,
                        tuple(top_right),
                        tuple(bottom_right),
                        (0, 255, 0),
                        3
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_right),
                        tuple(bottom_left),
                        (0, 255, 0),
                        3
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_left),
                        tuple(top_left),
                        (0, 255, 0),
                        3
                    )


                    # ================================================
                    # ЦЕНТР МЕТКИ
                    # ================================================

                    center_x = int(
                        (
                            top_left[0]
                            + bottom_right[0]
                        ) / 2
                    )

                    center_y = int(
                        (
                            top_left[1]
                            + bottom_right[1]
                        ) / 2
                    )

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        5,
                        (0, 0, 255),
                        -1
                    )


                    # ================================================
                    # ID НА ВИДЕО
                    # ================================================

                    label = (
                        f"ARUCO ID: {marker_id}"
                    )

                    cv2.putText(
                        frame,
                        label,
                        (
                            int(top_left[0]),
                            max(
                                25,
                                int(top_left[1]) - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )


            # ====================================================
            # СТАТУС
            # ====================================================

            with aruco_lock:

                unique_count = len(
                    aruco_ids
                )

                ids_text = ",".join(
                    map(str, aruco_ids)
                )


            cv2.rectangle(
                frame,
                (10, 10),
                (500, 110),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                frame,
                "ARUCO SCANNER",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"UNIQUE: {unique_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            if ids_text:

                cv2.putText(
                    frame,
                    f"IDs: {ids_text}",
                    (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )


            # ====================================================
            # WEB STREAM
            # ====================================================

            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=20
            )


    except Exception as e:

        print(
            "[VIDEO ERROR]",
            e
        )


    finally:

        print(
            "[VIDEO] Остановка камеры..."
        )

        if camera is not None:

            try:
                camera.stop()
            except Exception:
                pass

        if viewer is not None:

            try:
                viewer.close()
            except Exception:
                pass


# ============================================================
# MAIN
# ============================================================

pioneer = None


try:

    # ========================================================
    # ПОДКЛЮЧЕНИЕ
    # ========================================================

    print(
        "Подключение к Pioneer..."
    )

    pioneer = Pioneer()

    print(
        "Pioneer подключен"
    )


    # ========================================================
    # КАМЕРА ВНИЗ
    # ========================================================

    print(
        f"Поворот камеры на {CAMERA_ANGLE} градусов..."
    )

    servo_camera = ServoCamera()

    if servo_camera.set_angle(
        CAMERA_ANGLE
    ):

        print(
            f"Камера установлена на {CAMERA_ANGLE} градусов"
        )

    else:

        print(
            "Не удалось повернуть камеру"
        )


    time.sleep(1)


    # ========================================================
    # ЗАПУСК ВИДЕО
    # ========================================================

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )

    video_thread.start()


    print()
    print(
        "ArUco сканирование запущено"
    )

    print(
        "Новый ID = сохранение полного кадра"
    )

    print(
        "Для выхода нажмите CTRL+C"
    )


    # ========================================================
    # ОЖИДАНИЕ
    # ========================================================

    while True:

        time.sleep(1)


# ============================================================
# CTRL+C
# ============================================================

except KeyboardInterrupt:

    print(
        "\nОстановка оператором"
    )


# ============================================================
# ЗАВЕРШЕНИЕ
# ============================================================

finally:

    running = False

    time.sleep(1)

    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)

    with aruco_lock:

        print(
            f"Найдено уникальных ArUco: {len(aruco_ids)}"
        )

        print(
            f"ID: {aruco_ids}"
        )

    print(
        f"Фотографии: {PHOTO_DIR}"
    )

    print("=" * 60)


    if pioneer is not None:

        try:

            pioneer.close_connection()

        except Exception:

            pass


    print(
        "Программа завершена"
    )
