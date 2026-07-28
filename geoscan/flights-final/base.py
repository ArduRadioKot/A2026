import time
import threading

import cv2
import numpy as np
from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera
from rknnlite.api import RKNNLite


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"
VIDEO_OUTPUT = "flight.mp4"
VIDEO_FPS = 30

RKNN_MODEL = "best-rk3576.rknn"
RKNN_CONF = 0.60
RKNN_SIZE = 640
NMS_IOU = 0.45

# ВАЖНО: порядок должен совпадать с индексами классов вашей RKNN-модели.
# Если при обучении 0=unregistered, 1=registered — поменяйте строки местами.
CLASS_NAMES = [
    "unregistered",
    "registered",
]

# Цвета по регламенту (OpenCV использует BGR):
# зарегистрированное — зелёный, незарегистрированное — оранжевый.
COLOR_REGISTERED = (0, 255, 0)
COLOR_UNREGISTERED = (0, 165, 255)
COLOR_UNKNOWN = (255, 0, 0)
COLOR_ARUCO = (255, 255, 255)

running = True

aruco_ids_set = set()
aruco_ids = []
registered_boat_ids = set()
unregistered_boat_ids = set()
aruco_lock = threading.Lock()


# ============================================================
# RKNN / YOLO11
# ============================================================

print("[RKNN] Загрузка модели...")
rknn = RKNNLite()

ret = rknn.load_rknn(RKNN_MODEL)
if ret != 0:
    raise RuntimeError(f"Не удалось загрузить RKNN модель: {RKNN_MODEL}")

ret = rknn.init_runtime()
if ret != 0:
    raise RuntimeError("Не удалось запустить RKNN runtime")

print("[RKNN] Модель загружена")


def preprocess(frame):
    """Подготовка BGR-кадра для RKNN YOLO11."""
    image = cv2.resize(
        frame,
        (RKNN_SIZE, RKNN_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.ascontiguousarray(image)
    return np.expand_dims(image, axis=0)


def xywh_to_xyxy(boxes):
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def box_iou_one_to_many(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )
    union = area1 + area2 - intersection
    return intersection / np.maximum(union, 1e-6)


def numpy_nms(boxes, scores, iou_threshold=NMS_IOU):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int32)

    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)

        if order.size == 1:
            break

        rest = order[1:]
        ious = box_iou_one_to_many(boxes[i], boxes[rest])
        order = rest[ious <= iou_threshold]

    return np.asarray(keep, dtype=np.int32)


def flatten_yolo_output(outputs):
    """Приводит типичный декодированный Ultralytics YOLO11 output к [N, 4+C]."""
    candidates = []

    for output in outputs:
        arr = np.asarray(output)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            continue

        # Типичный вариант [C, N] -> [N, C].
        if arr.shape[0] < arr.shape[1] and 5 <= arr.shape[0] <= 512:
            arr = arr.T

        if 5 <= arr.shape[1] <= 512:
            candidates.append(arr.astype(np.float32, copy=False))

    if not candidates:
        shapes = [np.asarray(x).shape for x in outputs]
        raise RuntimeError(
            "Не удалось распознать формат выхода RKNN. "
            f"Получены shapes={shapes}. Для multi-output DFL нужен отдельный декодер."
        )

    return max(candidates, key=lambda x: x.shape[0])


def postprocess(outputs, original_shape):
    """Возвращает [x1, y1, x2, y2, class_id, confidence]."""
    pred = flatten_yolo_output(outputs)

    if pred.size == 0:
        return []

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]

    if class_scores.shape[1] == 0:
        return []

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= RKNN_CONF
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    boxes = xywh_to_xyxy(boxes_xywh)

    # На случай нормализованных координат 0..1.
    if boxes.size and np.nanmax(np.abs(boxes)) <= 2.0:
        boxes[:, [0, 2]] *= RKNN_SIZE
        boxes[:, [1, 3]] *= RKNN_SIZE

    frame_h, frame_w = original_shape[:2]
    scale_x = frame_w / float(RKNN_SIZE)
    scale_y = frame_h / float(RKNN_SIZE)

    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_h - 1)

    detections = []

    # NMS отдельно для каждого класса.
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        keep_local = numpy_nms(boxes[idx], scores[idx], NMS_IOU)

        for local_i in keep_local:
            i = idx[int(local_i)]
            x1, y1, x2, y2 = boxes[i]
            detections.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    int(class_ids[i]),
                    float(scores[i]),
                ]
            )

    detections.sort(key=lambda item: item[5], reverse=True)
    return detections


# ============================================================
# ARUCO 4x4_1000 ПО РЕГЛАМЕНТУ
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_1000
)
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    cv2.aruco.DetectorParameters(),
)


# ============================================================
# КАМЕРА ВНИЗ
# ============================================================

servo_camera = ServoCamera()
if servo_camera.set_angle(-80):
    print("[CAMERA] Камера установлена на -80 градусов")
else:
    print("[CAMERA] Не удалось установить угол камеры")


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def point_inside_box(point_x, point_y, box):
    x1, y1, x2, y2 = box
    return x1 <= point_x <= x2 and y1 <= point_y <= y2


def class_name_from_id(class_id):
    if 0 <= class_id < len(CLASS_NAMES):
        return CLASS_NAMES[class_id]
    return f"class_{class_id}"


def class_kind(class_name):
    """registered / unregistered / unknown с поддержкой RU/EN имён классов."""
    key = str(class_name).lower()

    # Сначала unregistered: это слово содержит registered.
    if "unregistered" in key or "незарегистр" in key:
        return "unregistered"

    if "registered" in key or "зарегистр" in key:
        return "registered"

    return "unknown"


def class_color(class_name):
    kind = class_kind(class_name)
    if kind == "registered":
        return COLOR_REGISTERED
    if kind == "unregistered":
        return COLOR_UNREGISTERED
    return COLOR_UNKNOWN


def draw_box_label(frame, box, class_name, confidence):
    x1, y1, x2, y2 = box
    color = class_color(class_name)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    label = f"{class_name} {confidence:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        2,
    )
    text_y = max(y1, text_h + 10)

    cv2.rectangle(
        frame,
        (x1, text_y - text_h - 8),
        (x1 + text_w + 10, text_y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 5, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
    )


# ============================================================
# ВИДЕО + ДЕТЕКТОР + ARUCO
# ============================================================


def video_worker():
    global running

    camera = None
    viewer = None
    writer = None

    try:
        print("[VIDEO] Подключение камеры...")
        camera = Camera()
        viewer = ImageViewer()

        print("=" * 60)
        print("Видеопоток запущен")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}")
        print("=" * 60)

        while running:
            frame = camera.get_cv_frame(timeout=2.0)
            if frame is None:
                continue

            # Чистый кадр используем для нейросети и ArUco.
            original_frame = frame.copy()

            # Writer создаём по фактическому размеру камеры.
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    VIDEO_OUTPUT,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(
                        f"Не удалось открыть файл записи: {VIDEO_OUTPUT}"
                    )
                print(f"[VIDEO] Запись полёта: {VIDEO_OUTPUT}")

            # ----------------------------------------------------
            # RKNN AI DETECTOR
            # ----------------------------------------------------
            boat_detections = []

            try:
                input_data = preprocess(original_frame)
                outputs = rknn.inference(inputs=[input_data])

                if not hasattr(video_worker, "printed_output_shapes"):
                    print(
                        "[RKNN] Output shapes:",
                        [np.asarray(x).shape for x in outputs],
                    )
                    video_worker.printed_output_shapes = True

                detections = postprocess(outputs, original_frame.shape)

                for x1, y1, x2, y2, cls, conf in detections:
                    x1, y1, x2, y2 = map(
                        int,
                        (x1, y1, x2, y2),
                    )
                    class_name = class_name_from_id(int(cls))

                    boat_detections.append(
                        {
                            "box": (x1, y1, x2, y2),
                            "class": class_name,
                            "confidence": float(conf),
                        }
                    )

                    # Рамка сразу рисуется на frame — именно этот frame
                    # далее попадёт и в web-stream, и в flight.mp4.
                    draw_box_label(
                        frame,
                        (x1, y1, x2, y2),
                        class_name,
                        float(conf),
                    )

            except Exception as error:
                print("[RKNN ERROR]", error)

            # ----------------------------------------------------
            # ARUCO
            # ----------------------------------------------------
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None:
                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    points = marker_corners.reshape((4, 2)).astype(int)

                    top_left = points[0]
                    top_right = points[1]
                    bottom_right = points[2]
                    bottom_left = points[3]

                    center_x = int((top_left[0] + bottom_right[0]) / 2)
                    center_y = int((top_left[1] + bottom_right[1]) / 2)

                    matched_boat = None
                    for boat in boat_detections:
                        if point_inside_box(
                            center_x,
                            center_y,
                            boat["box"],
                        ):
                            matched_boat = boat
                            break

                    # Уникальным объект считаем только тогда, когда ArUco
                    # действительно находится внутри AI-детекции судна.
                    if matched_boat is not None:
                        new_marker = False

                        with aruco_lock:
                            if marker_id not in aruco_ids_set:
                                aruco_ids_set.add(marker_id)
                                aruco_ids.append(marker_id)
                                new_marker = True

                        if new_marker:
                            detected_class = str(matched_boat["class"])
                            kind = class_kind(detected_class)

                            # Формат терминала из регламента.
                            print(f"Обнаружено судно с id {marker_id}")

                            with aruco_lock:
                                if kind == "unregistered":
                                    unregistered_boat_ids.add(marker_id)
                                    print("Обнаружено незарегистрированное судно")
                                elif kind == "registered":
                                    registered_boat_ids.add(marker_id)
                                    print("Обнаружено зарегистрированное судно")
                                else:
                                    print(
                                        "Обнаружено судно "
                                        f"класса «{detected_class}»"
                                    )

                    # Рамка ArUco + ID тоже попадёт в записываемое видео.
                    cv2.polylines(
                        frame,
                        [points.reshape((-1, 1, 2))],
                        True,
                        COLOR_ARUCO,
                        2,
                    )
                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        4,
                        (0, 0, 255),
                        -1,
                    )

                    aruco_label = f"ARUCO ID: {marker_id}"
                    if matched_boat is not None:
                        aruco_label += f" [{matched_boat['class']}]"

                    cv2.putText(
                        frame,
                        aruco_label,
                        (int(top_left[0]), max(20, int(top_left[1]) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        COLOR_ARUCO,
                        2,
                    )

            # ----------------------------------------------------
            # ИНФОРМАЦИОННАЯ ПАНЕЛЬ
            # ----------------------------------------------------
            with aruco_lock:
                aruco_count = len(aruco_ids)
                ids_text = ",".join(map(str, aruco_ids))

            cv2.rectangle(frame, (10, 10), (540, 135), (0, 0, 0), -1)
            cv2.putText(
                frame,
                "PIONEER CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"BOATS: {len(boat_detections)}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"UNIQUE BOATS / ARUCO: {aruco_count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            if ids_text:
                cv2.putText(
                    frame,
                    f"IDs: {ids_text}",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            # КЛЮЧЕВО: записываем ПОСЛЕ всей отрисовки.
            # Поэтому AI-рамки, подписи, ArUco и информационная панель
            # присутствуют в flight.mp4.
            writer.write(frame)

            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=20,
            )

    except Exception as error:
        print("[VIDEO ERROR]", error)

    finally:
        print("[VIDEO] Остановка")

        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass

        if writer is not None:
            try:
                writer.release()
                print(f"[VIDEO] Запись сохранена: {VIDEO_OUTPUT}")
            except Exception:
                pass

        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass


# ============================================================
# МАРШРУТ ПОЛЁТА — СОХРАНЁН ИЗ ИСХОДНОГО КОДА
# ============================================================

route = [
    (-1.0, 1.0, 1.5, 0.0),
    (-1.0, 1.0, 1.5, 0.0),
    (-1.0, 2.5, 1.5, 0.0),
    (-1.0, 4.0, 1.5, 0.0),

    (-1.6, 4.0, 1.5, 0.0),
    (-1.6, 2.5, 1.5, 0.0),
    (-1.6, 1.0, 1.5, 0.0),

    (-2.2, 1.0, 1.5, 0.0),
    (-2.2, 2.5, 1.5, 0.0),
    (-2.2, 4.0, 1.5, 0.0),

    (-2.8, 4.0, 1.7, 0.0),
    (-2.8, 2.5, 1.7, 0.0),
    (-2.8, 1.0, 1.7, 0.0),

    (-3.1, 1.0, 1.7, 0.0),
    (-3.1, 2.5, 1.7, 0.0),
    (-3.1, 4.0, 1.7, 0.0),

    (0.0, 0.0, 1.7, 0.0),
]


# ============================================================
# MAIN + АВТОНОМНЫЙ ПОЛЁТ
# ============================================================

pioneer = None
video_thread = None

try:
    print("Подключение к Pioneer...")
    pioneer = Pioneer()
    print("Pioneer подключен")

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True,
    )
    video_thread.start()

    print("Камера и RKNN-распознавание запущены")
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

        pioneer.go_to_local_point(x, y, z, yaw)

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
        except Exception as error:
            print("Ошибка при аварийной посадке:", error)

except Exception as error:
    print("\nКРИТИЧЕСКАЯ ОШИБКА:", error)

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

    # Даём потоку камеры корректно закрыть VideoWriter до release RKNN.
    if video_thread is not None and video_thread.is_alive():
        video_thread.join(timeout=3.0)

    try:
        rknn.release()
    except Exception:
        pass

    print()
    print("=" * 60)
    print("РЕЗУЛЬТАТ")
    print("=" * 60)

    with aruco_lock:
        registered_count = len(registered_boat_ids)
        unregistered_count = len(unregistered_boat_ids)
        total_boats = len(aruco_ids)

        print(f"Найдено уникальных ArUco: {total_boats}")
        print(f"ID: {aruco_ids}")
        print()
        print(f"Обнаружено {registered_count} зарегистрированных суден")
        print(f"Обнаружено {unregistered_count} незарегистрированных суден")
        print(f"Общее количество суден: {total_boats}")

    print(f"Видеозапись с разметкой: {VIDEO_OUTPUT}")
    print("=" * 60)

    if pioneer is not None:
        try:
            pioneer.close_connection()
        except Exception:
            pass

    print("Программа завершена.")
