import time
import threading
import cv2

from pioneer_sdk2 import Pioneer
from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"

running = True


# ============================================================
# ARUCO
# ============================================================

# Храним уникальные ID
aruco_ids_set = set()

# Храним порядок первого обнаружения
aruco_ids = []

aruco_lock = threading.Lock()


# ВАЖНО:
# словарь должен соответствовать вашим физическим ArUco-меткам
aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# ============================================================
# МАРШРУТ
# ============================================================

route = [
    (-1.4, 0.5, 1.5, 0.0),
    (-1.4, 1.0, 1.5, 0.0),
    (-1.4, 2.5, 1.5, 0.0),
    (-1.4, 4.0, 1.5, 0.0),

    (-1.8, 4.0, 1.5, 0.0),
    (-1.8, 2.5, 1.5, 0.0),
    (-1.8, 0.5, 1.5, 0.0),

    (-2.2, 0.5, 1.5, 0.0),
    (-2.2, 2.5, 1.5, 0.0),
    (-2.2, 4.0, 1.5, 0.0),


    (-2.8, 4.0, 1.7, 0.0),
    (-2.8, 2.5, 1.7, 0.0),
    (-2.8, 0.5, 1.7, 0.0),

    (-3.1, 0.5, 1.7, 0.0),
    (-3.1, 2.5, 1.7, 0.0),
    (-3.1, 4.0, 1.7, 0.0),


    (-0.5, 1.0, 1.7, 0.0)
]


# ============================================================
# ВИДЕО + ARUCO
# ============================================================

def video_worker():

    global running

    camera = None
    viewer = None

    try:

        print("[VIDEO] Запуск камеры...")

        camera = Camera()

        viewer = ImageViewer()

        print()
        print("=" * 60)
        print("[VIDEO] Веб-поток:")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}")
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
                continue


            # ====================================================
            # ARUCO
            # ====================================================

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, rejected = (
                aruco_detector.detectMarkers(gray)
            )


            # ====================================================
            # ЕСЛИ НАЙДЕНЫ МЕТКИ
            # ====================================================

            if ids is not None:

                for marker_corners, marker_id in zip(
                    corners,
                    ids.flatten()
                ):

                    marker_id = int(marker_id)


                    # Получаем четыре угла
                    points = marker_corners.reshape(
                        (4, 2)
                    ).astype(int)

                    top_left = points[0]
                    top_right = points[1]
                    bottom_right = points[2]
                    bottom_left = points[3]


                    # ====================================================
                    # ОБВОДКА ARUCO
                    # ====================================================

                    cv2.line(
                        frame,
                        tuple(top_left),
                        tuple(top_right),
                        (0, 255, 0),
                        4
                    )

                    cv2.line(
                        frame,
                        tuple(top_right),
                        tuple(bottom_right),
                        (0, 255, 0),
                        4
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_right),
                        tuple(bottom_left),
                        (0, 255, 0),
                        4
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_left),
                        tuple(top_left),
                        (0, 255, 0),
                        4
                    )


                    # ====================================================
                    # ЦЕНТР МЕТКИ
                    # ====================================================

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
                        6,
                        (0, 0, 255),
                        -1
                    )


                    # ====================================================
                    # ID НА ВИДЕО
                    # ====================================================

                    label = f"ARUCO ID: {marker_id}"


                    (text_width, text_height), baseline = (
                        cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            2
                        )
                    )


                    text_x = int(top_left[0])

                    text_y = int(
                        top_left[1] - 10
                    )


                    # Если текст выходит за верх экрана
                    if text_y < text_height + 10:

                        text_y = int(
                            bottom_left[1]
                            + text_height
                            + 15
                        )


                    # Чёрный фон
                    cv2.rectangle(
                        frame,
                        (
                            text_x,
                            text_y - text_height - 7
                        ),
                        (
                            text_x + text_width + 10,
                            text_y + 5
                        ),
                        (0, 0, 0),
                        -1
                    )


                    # Текст
                    cv2.putText(
                        frame,
                        label,
                        (
                            text_x + 5,
                            text_y
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    with aruco_lock:

                        if marker_id not in aruco_ids_set:

                            aruco_ids_set.add(
                                marker_id
                            )

                            aruco_ids.append(
                                marker_id
                            )

                            print()
                            print(
                                "=" * 40
                            )

                            print(
                                f"[ARUCO] НОВАЯ МЕТКА: {marker_id}"
                            )

                            print(
                                f"[ARUCO] Все найденные: {aruco_ids}"
                            )

                            print(
                                "=" * 40
                            )

                            print()


            # ====================================================
            # ИНФОРМАЦИЯ В ЛЕВОМ ВЕРХНЕМ УГЛУ
            # ====================================================

            cv2.rectangle(
                frame,
                (10, 10),
                (420, 110),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                "PIONEER STREAM",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            with aruco_lock:

                aruco_count = len(
                    aruco_ids
                )

                ids_text = ",".join(
                    map(str, aruco_ids)
                )


            cv2.putText(
                frame,
                f"ARUCO FOUND: {aruco_count}",
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
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )


            # ====================================================
            # ПУБЛИКАЦИЯ В ВЕБ
            # ====================================================

            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=20
            )


    except Exception as e:

        print(
            "[VIDEO ERROR]:",
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
# ПОЛЁТ
# ============================================================

pioneer = None


try:

    # ============================================================
    # ПОДКЛЮЧЕНИЕ
    # ============================================================

    print(
        "Подключение к Pioneer..."
    )

    pioneer = Pioneer()

    print(
        "Pioneer подключен"
    )


    # ============================================================
    # ПОВОРОТ КАМЕРЫ ВНИЗ
    # ============================================================

    print(
        "Поворот камеры вниз..."
    )


    servo_camera = ServoCamera()


    if servo_camera.set_angle(-80):

        print(
            "Камера установлена на -80 градусов"
        )

    else:

        print(
            "Не удалось повернуть камеру"
        )


    time.sleep(1)


    # ============================================================
    # ЗАПУСК ВИДЕО
    # ============================================================

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )

    video_thread.start()


    time.sleep(2)


    # ============================================================
    # ARM
    # ============================================================

    print(
        "ARM..."
    )

    pioneer.arm()

    print(
        "ARM: OK"
    )


    # ============================================================
    # ВЗЛЁТ
    # ============================================================

    print(
        "Взлёт..."
    )

    pioneer.takeoff()


    while not pioneer.point_reached():

        time.sleep(0.1)


    print(
        "Взлёт завершён"
    )


    # ============================================================
    # МАРШРУТ
    # ============================================================

    for x, y, z, yaw in route:

        print()
        print(
            f"Летим к точке: "
            f"x={x}, y={y}, z={z}, yaw={yaw}"
        )


        pioneer.go_to_local_point(
            x,
            y,
            z,
            yaw
        )


        while not pioneer.point_reached():

            pos = pioneer.get_local_position_lps()


            if pos:

                print(
                    "Позиция:",
                    [
                        round(value, 2)
                        for value in pos
                    ],
                    end="\r"
                )


            time.sleep(0.12)


        print()

        print(
            f"Точка достигнута: "
            f"x={x}, y={y}"
        )


        # ========================================================
        # ARUCO ПОСЛЕ КАЖДОЙ ТОЧКИ
        # ========================================================

        with aruco_lock:

            print(
                "Найденные ArUco:",
                aruco_ids
            )


    # ============================================================
    # МАРШРУТ ЗАВЕРШЁН
    # ============================================================

    print()
    print(
        "=" * 60
    )

    print(
        "МАРШРУТ ЗАВЕРШЁН"
    )


    with aruco_lock:

        print(
            "Уникальные ArUco ID:"
        )

        print(
            aruco_ids
        )

        print(
            "Количество:",
            len(aruco_ids)
        )


    print(
        "=" * 60
    )


    # ============================================================
    # ПОСАДКА
    # ============================================================

    print(
        "Посадка..."
    )


    pioneer.land()


    while not pioneer.point_reached():

        time.sleep(0.1)


    pioneer.disarm()


    print(
        "Миссия завершена"
    )


# ============================================================
# CTRL + C
# ============================================================

except KeyboardInterrupt:

    print(
        "\nОстановка оператором"
    )


    if pioneer is not None:

        try:

            print(
                "Аварийная посадка..."
            )

            pioneer.land()

            time.sleep(3)

            pioneer.disarm()

        except Exception as e:

            print(
                "Ошибка посадки:",
                e
            )


# ============================================================
# КРИТИЧЕСКАЯ ОШИБКА
# ============================================================

except Exception as e:

    print()
    print(
        "КРИТИЧЕСКАЯ ОШИБКА:",
        e
    )


    if pioneer is not None:

        try:

            print(
                "Аварийная посадка..."
            )

            pioneer.land()

            time.sleep(3)

            pioneer.disarm()

        except Exception as e2:

            print(
                "Ошибка аварийной посадки:",
                e2
            )


# ============================================================
# ЗАВЕРШЕНИЕ
# ============================================================

finally:

    running = False


    print()
    print(
        "=" * 60
    )

    print(
        "ИТОГОВЫЕ ARUCO ID"
    )


    with aruco_lock:

        print(
            aruco_ids
        )

        print(
            f"Всего уникальных: {len(aruco_ids)}"
        )


    print(
        "=" * 60
    )


    if pioneer is not None:

        try:

            print(
                "Закрытие соединения"
            )

            pioneer.close_connection()

        except Exception:
            pass


    print(
        "Программа завершена"
    )