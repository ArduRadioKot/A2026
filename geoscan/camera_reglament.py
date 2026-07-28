import time
import threading
import cv2
import os
from datetime import datetime

from ultralytics import YOLO

from pioneer_sdk2 import Pioneer
from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"
VIDEO_OUTPUT = "flight.mp4"
VIDEO_FPS = 30

YOLO_MODEL_PATH = "yolo11n_two_boats_best.pt"
YOLO_CONF = 0.45
YOLO_IMGSZ = 640

# Папка для фотографий найденных кораблей
PHOTO_DIR = "captured_boats"

# Дополнительный отступ вокруг корабля на фотографии
PHOTO_PADDING = 20

os.makedirs(
    PHOTO_DIR,
    exist_ok=True
)

running = True


# ============================================================
# YOLO
# ============================================================

print("[YOLO] Загрузка модели...")

yolo_model = YOLO(
    YOLO_MODEL_PATH
)

print("[YOLO] Модель загружена")


# ============================================================
# ARUCO
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_1000
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)

# Уже найденные уникальные метки
aruco_ids_set = set()

# Порядок обнаружения
aruco_ids = []

aruco_lock = threading.Lock()


# ============================================================
# КАМЕРА ВНИЗ
# ============================================================

servo_camera = ServoCamera()

if servo_camera.set_angle(-80):

    print(
        "[CAMERA] Камера установлена на -80 градусов"
    )

else:

    print(
        "[CAMERA] Не удалось установить угол камеры"
    )


# ============================================================
# ПРОВЕРКА: ТОЧКА ВНУТРИ КОРАБЛЯ
# ============================================================

def point_inside_box(
    point_x,
    point_y,
    box
):

    x1, y1, x2, y2 = box

    return (
        x1 <= point_x <= x2
        and
        y1 <= point_y <= y2
    )


# ============================================================
# СОХРАНЕНИЕ ФОТО КОРАБЛЯ
# ============================================================

def save_boat_photo(
    original_frame,
    boat_box,
    boat_class,
    marker_id,
    confidence
):

    x1, y1, x2, y2 = boat_box

    height, width = original_frame.shape[:2]


    # Добавляем небольшой запас вокруг корабля
    x1 = max(
        0,
        x1 - PHOTO_PADDING
    )

    y1 = max(
        0,
        y1 - PHOTO_PADDING
    )

    x2 = min(
        width,
        x2 + PHOTO_PADDING
    )

    y2 = min(
        height,
        y2 + PHOTO_PADDING
    )


    # Проверяем корректность координат
    if x2 <= x1 or y2 <= y1:

        print(
            "[PHOTO] Некорректная рамка корабля"
        )

        return None


    # Вырезаем корабль из исходного,
    # ещё не размеченного кадра
    boat_image = original_frame[
        y1:y2,
        x1:x2
    ].copy()


    if boat_image.size == 0:

        print(
            "[PHOTO] Пустое изображение корабля"
        )

        return None


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )[:-3]


    safe_class_name = str(
        boat_class
    ).replace(
        " ",
        "_"
    )


    filename = (
        f"{safe_class_name}"
        f"_aruco_{marker_id}"
        f"_{timestamp}.jpg"
    )


    filepath = os.path.join(
        PHOTO_DIR,
        filename
    )


    success = cv2.imwrite(
        filepath,
        boat_image
    )


    if success:

        print()
        print("=" * 60)

        print(
            "[PHOTO] КОРАБЛЬ СОХРАНЁН"
        )

        print(
            f"Класс: {boat_class}"
        )

        print(
            f"YOLO confidence: {confidence:.2f}"
        )

        print(
            f"ArUco ID: {marker_id}"
        )

        print(
            f"Файл: {filepath}"
        )

        print("=" * 60)
        print()

        return filepath


    print(
        "[PHOTO] Ошибка сохранения фотографии"
    )

    return None


# ============================================================
# ВИДЕО ПОТОК
# ============================================================

def video_worker():

    global running

    camera = None
    viewer = None
    writer = None

    try:

        print(
            "[VIDEO] Подключение камеры..."
        )

        camera = Camera()
        viewer = ImageViewer()

        print()
        print("=" * 60)

        print(
            "Видеопоток запущен"
        )

        print()

        print(
            "Открой в браузере:"
        )

        print(
            f"http://10.42.0.1:8889/{STREAM_NAME}"
        )

        print("=" * 60)
        print()


        while running:

            frame = camera.get_cv_frame(
                timeout=2.0
            )


            if frame is None:

                print(
                    "[VIDEO] Нет кадра"
                )

                continue

            # ====================================================
            # ЗАПИСЬ ВИДЕО
            # ====================================================

            if writer is None:
                height, width = frame.shape[:2]

                writer = cv2.VideoWriter(
                    VIDEO_OUTPUT,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    (width, height)
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Не удалось открыть файл записи: {VIDEO_OUTPUT}"
                    )

                print(
                    f"[VIDEO] Запись полёта: {VIDEO_OUTPUT}"
                )


            # ====================================================
            # СОХРАНЯЕМ ЧИСТЫЙ КАДР
            # ====================================================

            # Именно из него потом вырезается фотография корабля,
            # поэтому на сохранённом фото не будет рамок и текста.

            original_frame = frame.copy()


            # ====================================================
            # YOLO
            # ====================================================

            boat_detections = []

            try:

                results = yolo_model.predict(
                    source=frame,
                    conf=YOLO_CONF,
                    imgsz=YOLO_IMGSZ,
                    verbose=False
                )


                if results:

                    result = results[0]


                    if result.boxes is not None:

                        for box in result.boxes:

                            confidence = float(
                                box.conf[0]
                            )

                            class_id = int(
                                box.cls[0]
                            )

                            class_name = result.names[
                                class_id
                            ]


                            x1, y1, x2, y2 = (
                                box.xyxy[0]
                                .cpu()
                                .numpy()
                                .astype(int)
                            )


                            # Сохраняем информацию о корабле,
                            # чтобы потом проверить,
                            # находится ли ArUco внутри него.

                            boat_detections.append(
                                {
                                    "box": (
                                        x1,
                                        y1,
                                        x2,
                                        y2
                                    ),
                                    "class": class_name,
                                    "confidence": confidence
                                }
                            )


                            # ========================================
                            # РАМКА КОРАБЛЯ
                            # ========================================

                            cv2.rectangle(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (255, 0, 0),
                                3
                            )


                            label = (
                                f"{class_name} "
                                f"{confidence:.2f}"
                            )


                            (
                                text_width,
                                text_height
                            ), baseline = cv2.getTextSize(
                                label,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                2
                            )


                            text_y = max(
                                y1,
                                text_height + 10
                            )


                            cv2.rectangle(
                                frame,
                                (
                                    x1,
                                    text_y
                                    - text_height
                                    - 8
                                ),
                                (
                                    x1
                                    + text_width
                                    + 10,
                                    text_y + 4
                                ),
                                (0, 0, 0),
                                -1
                            )


                            cv2.putText(
                                frame,
                                label,
                                (
                                    x1 + 5,
                                    text_y
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (255, 255, 255),
                                2
                            )


            except Exception as e:

                print(
                    "[YOLO ERROR]",
                    e
                )


            # ====================================================
            # ARUCO
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


            if ids is not None:

                for marker_corners, marker_id in zip(
                    corners,
                    ids.flatten()
                ):

                    marker_id = int(
                        marker_id
                    )


                    points = (
                        marker_corners
                        .reshape((4, 2))
                        .astype(int)
                    )


                    top_left = points[0]
                    top_right = points[1]
                    bottom_right = points[2]
                    bottom_left = points[3]


                    # ====================================================
                    # ЦЕНТР ARUCO
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


                    # ====================================================
                    # ИЩЕМ КОРАБЛЬ, НА КОТОРОМ НАХОДИТСЯ ARUCO
                    # ====================================================

                    matched_boat = None


                    for boat in boat_detections:

                        if point_inside_box(
                            center_x,
                            center_y,
                            boat["box"]
                        ):

                            matched_boat = boat
                            break


                    # ====================================================
                    # НОВЫЙ ARUCO НА КОРАБЛЕ
                    # ====================================================

                    if matched_boat is not None:

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
                        # ФОТО ТОЛЬКО ПРИ ПЕРВОМ ОБНАРУЖЕНИИ
                        # ================================================

                        if new_marker:

                            print()
                            print(
                                f"[ARUCO] НОВАЯ МЕТКА: "
                                f"{marker_id}"
                            )

                            print(
                                f"[ARUCO] На корабле: "
                                f"{matched_boat['class']}"
                            )

                            print(
                                f"[ARUCO] Все ID: "
                                f"{aruco_ids}"
                            )


                            save_boat_photo(
                                original_frame=original_frame,
                                boat_box=matched_boat[
                                    "box"
                                ],
                                boat_class=matched_boat[
                                    "class"
                                ],
                                marker_id=marker_id,
                                confidence=matched_boat[
                                    "confidence"
                                ]
                            )


                    # ====================================================
                    # РАМКА ARUCO
                    # ====================================================

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


                    # ====================================================
                    # ЦЕНТР МЕТКИ
                    # ====================================================

                    cv2.circle(
                        frame,
                        (
                            center_x,
                            center_y
                        ),
                        5,
                        (0, 0, 255),
                        -1
                    )


                    # ====================================================
                    # ПОДПИСЬ ARUCO
                    # ====================================================

                    aruco_label = (
                        f"ARUCO ID: {marker_id}"
                    )


                    if matched_boat is not None:

                        aruco_label += (
                            f" [{matched_boat['class']}]"
                        )


                    cv2.putText(
                        frame,
                        aruco_label,
                        (
                            int(
                                top_left[0]
                            ),
                            max(
                                20,
                                int(
                                    top_left[1]
                                ) - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )


            # ====================================================
            # ИНФОРМАЦИЯ
            # ====================================================

            with aruco_lock:

                aruco_count = len(
                    aruco_ids
                )

                ids_text = ",".join(
                    map(
                        str,
                        aruco_ids
                    )
                )


            cv2.rectangle(
                frame,
                (10, 10),
                (520, 135),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                "PIONEER CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"BOATS: {len(boat_detections)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"UNIQUE BOATS / ARUCO: {aruco_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            if ids_text:

                cv2.putText(
                    frame,
                    f"IDs: {ids_text}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )


            # ====================================================
            # ЗАПИСЬ ОБРАБОТАННОГО КАДРА
            # ====================================================

            if writer is not None:
                writer.write(frame)


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
            "[VIDEO] Остановка"
        )


        if camera is not None:

            try:

                camera.stop()

            except Exception:

                pass


        if writer is not None:

            try:
                writer.release()
                print(
                    f"[VIDEO] Запись сохранена: {VIDEO_OUTPUT}"
                )
            except Exception:
                pass


        if viewer is not None:

            try:

                viewer.close()

            except Exception:

                pass



# ============================================================
# МАРШРУТ ПОЛЁТА
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
# MAIN + АВТОНОМНЫЙ ПОЛЁТ
# ============================================================

pioneer = None

try:
    print("Подключение к Pioneer...")
    pioneer = Pioneer()
    print("Pioneer подключен")

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )
    video_thread.start()

    print("Камера и распознавание запущены")
    time.sleep(2)

    print("ARM...")
    pioneer.arm()
    print("ARM: OK")

    print("Взлёт...")
    pioneer.takeoff()

    while not pioneer.point_reached():
        time.sleep(0.10)

    print("Взлёт завершён")

    for x, y, z, yaw in route:
        print(f"Точка: x={x}, y={y}, z={z}, yaw={yaw}")

        pioneer.go_to_local_point(
            x,
            y,
            z,
            yaw
        )

        while not pioneer.point_reached():
            time.sleep(0.12)

        print(f"Точка достигнута: x={x}, y={y}, z={z}")

    print("Посадка...")
    pioneer.land()

    while not pioneer.point_reached():
        time.sleep(0.05)

    pioneer.disarm()
    print("Миссия завершена.")


except KeyboardInterrupt:
    print("\nОстановка оператором")

    if pioneer is not None:
        try:
            print("Аварийная посадка...")
            pioneer.land()
            time.sleep(3)
            pioneer.disarm()
        except Exception as e:
            print("Ошибка при аварийной посадке:", e)


except Exception as e:
    print("\nКРИТИЧЕСКАЯ ОШИБКА:", e)

    if pioneer is not None:
        try:
            print("Выполняется аварийная посадка...")
            pioneer.land()
            time.sleep(3)
            pioneer.disarm()
            print("Аппарат посажен.")
        except Exception as land_error:
            print("Ошибка аварийной посадки:", land_error)


finally:
    running = False
    time.sleep(1)

    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)

    with aruco_lock:
        print(f"Найдено уникальных ArUco: {len(aruco_ids)}")
        print(f"ID: {aruco_ids}")

    print(f"Фотографии: {PHOTO_DIR}")
    print(f"Видеозапись: {VIDEO_OUTPUT}")

    print("=" * 60)

    if pioneer is not None:
        try:
            pioneer.close_connection()
        except Exception:
            pass

    print("Программа завершена.")
