import time
import threading
import os

import cv2
import numpy as np
from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera
from rknnlite.api import RKNNLite


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"
VIDEO_OUTPUT = "flight.mp4"
VIDEO_TEMP_OUTPUT = "flight.part.mp4"
VIDEO_FPS = 30

RKNN_MODEL = "best-rk3576.rknn"
RKNN_CONF = 0.60
RKNN_SIZE = 640
NMS_IOU = 0.45

# Улучшение чтения ArUco
ARUCO_SCALE = 2.0
ARUCO_CROP_PADDING = 24
ARUCO_CLAHE_CLIP = 2.0
ARUCO_CLAHE_GRID = (8, 8)

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

aruco_parameters = cv2.aruco.DetectorParameters()

# Более точное уточнение углов метки.
aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
aruco_parameters.cornerRefinementWinSize = 5
aruco_parameters.cornerRefinementMaxIterations = 40
aruco_parameters.cornerRefinementMinAccuracy = 0.01

# Позволяем увереннее работать с небольшими метками.
aruco_parameters.adaptiveThreshWinSizeMin = 3
aruco_parameters.adaptiveThreshWinSizeMax = 31
aruco_parameters.adaptiveThreshWinSizeStep = 4
aruco_parameters.minMarkerPerimeterRate = 0.02
aruco_parameters.maxMarkerPerimeterRate = 4.0
aruco_parameters.polygonalApproxAccuracyRate = 0.03
aruco_parameters.minCornerDistanceRate = 0.03
aruco_parameters.minDistanceToBorder = 2
aruco_parameters.errorCorrectionRate = 0.6

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters,
)

aruco_clahe = cv2.createCLAHE(
    clipLimit=ARUCO_CLAHE_CLIP,
    tileGridSize=ARUCO_CLAHE_GRID,
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


def detect_aruco_in_boat_crops(frame, boat_detections):
    """
    Ищет ArUco только внутри YOLO-детекций судов.

    Для каждого bbox:
      1) добавляет небольшой padding;
      2) переводит crop в grayscale;
      3) применяет CLAHE;
      4) увеличивает crop в ARUCO_SCALE раз;
      5) ищет ArUco;
      6) переводит углы и центр обратно в координаты исходного кадра.

    Возвращает список словарей:
      id, points, center, matched_class
    """
    frame_h, frame_w = frame.shape[:2]
    found = []
    seen_ids_this_frame = set()

    for boat in boat_detections:
        x1, y1, x2, y2 = boat["box"]

        crop_x1 = max(0, int(x1) - ARUCO_CROP_PADDING)
        crop_y1 = max(0, int(y1) - ARUCO_CROP_PADDING)
        crop_x2 = min(frame_w, int(x2) + ARUCO_CROP_PADDING)
        crop_y2 = min(frame_h, int(y2) + ARUCO_CROP_PADDING)

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = aruco_clahe.apply(gray)

        if ARUCO_SCALE != 1.0:
            gray = cv2.resize(
                gray,
                None,
                fx=ARUCO_SCALE,
                fy=ARUCO_SCALE,
                interpolation=cv2.INTER_CUBIC,
            )

        corners, ids, _ = aruco_detector.detectMarkers(gray)

        if ids is None:
            continue

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)

            # Один и тот же ID может попасть в пересекающиеся YOLO crop'ы.
            if marker_id in seen_ids_this_frame:
                continue

            scaled_points = marker_corners.reshape((4, 2)).astype(np.float32)

            # Возвращаем координаты из увеличенного crop в исходный кадр.
            scaled_points[:, 0] = scaled_points[:, 0] / ARUCO_SCALE + crop_x1
            scaled_points[:, 1] = scaled_points[:, 1] / ARUCO_SCALE + crop_y1
            points = np.rint(scaled_points).astype(np.int32)

            center_x = int(round(float(np.mean(points[:, 0]))))
            center_y = int(round(float(np.mean(points[:, 1]))))

            # Финальная защита: центр метки должен быть внутри исходного bbox лодки,
            # а не только внутри crop с padding.
            if not point_inside_box(center_x, center_y, boat["box"]):
                continue

            seen_ids_this_frame.add(marker_id)

            found.append({
                "id": marker_id,
                "points": points.tolist(),
                "center": (center_x, center_y),
                "matched_class": str(boat["class"]),
            })

    return found


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
# МНОГОПОТОЧНЫЙ ВИДЕОКОНВЕЙЕР
# ============================================================

# Архитектура:
#   camera_worker    -> всегда хранит только САМЫЙ СВЕЖИЙ кадр
#   detection_worker -> RKNN + ArUco независимо от FPS трансляции
#   stream_worker    -> рисует последние известные результаты и стримит
#   record_worker    -> независимо пишет уже размеченное видео на диск
#   flight_worker    -> автономный полёт не зависит от видеопотока
#
# Это принципиально убирает очередь из устаревших кадров. Если RKNN работает
# медленнее камеры, детектор просто пропускает старые кадры и берёт свежий.

from queue import Queue, Empty, Full

state_lock = threading.Lock()
latest_frame = None
latest_frame_id = 0
latest_frame_time = 0.0

latest_result = {
    "frame_id": -1,
    "detections": [],
    "aruco": [],
    "inference_ms": 0.0,
}

# Небольшая очередь только для записи. При переполнении старый кадр
# отбрасывается: web-stream никогда не ждёт диск.
record_queue = Queue(maxsize=12)

# Устанавливается только после writer.release() и финализации MP4.
record_finished = threading.Event()

# Частота визуальной трансляции. ImageViewer сам кодирует кадр, поэтому 25 FPS
# обычно даёт заметно более плавную картинку без бессмысленной нагрузки 30+ FPS.
STREAM_FPS = 25


def put_latest(queue, item):
    """Положить элемент без блокировки, при переполнении удалить самый старый."""
    try:
        queue.put_nowait(item)
        return
    except Full:
        pass

    try:
        queue.get_nowait()
    except Empty:
        pass

    try:
        queue.put_nowait(item)
    except Full:
        pass


def camera_worker():
    """Поток №1: только захват камеры. Никакой нейросети и кодирования."""
    global running, latest_frame, latest_frame_id, latest_frame_time

    camera = None
    try:
        print("[CAMERA] Подключение...")
        camera = Camera()
        print("[CAMERA] Захват кадров запущен")

        while running:
            frame = camera.get_cv_frame(timeout=1.0)
            if frame is None:
                continue

            # Один copy здесь нужен, чтобы SDK мог переиспользовать свой буфер.
            with state_lock:
                latest_frame = frame.copy()
                latest_frame_id += 1
                latest_frame_time = time.monotonic()

    except Exception as error:
        print("[CAMERA ERROR]", error)
        running = False
    finally:
        if camera is not None:
            try:
                camera.stop()
            except Exception:
                pass
        print("[CAMERA] Остановлена")


def detection_worker():
    """Поток №2: RKNN + ArUco. Всегда анализирует самый свежий доступный кадр."""
    global running, latest_result

    last_processed_id = -1
    printed_shapes = False

    while running:
        with state_lock:
            frame_id = latest_frame_id
            if latest_frame is None or frame_id == last_processed_id:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.003)
            continue

        last_processed_id = frame_id
        started = time.perf_counter()
        boat_detections = []
        aruco_items = []

        try:
            input_data = preprocess(frame)
            outputs = rknn.inference(inputs=[input_data])

            if not printed_shapes:
                print("[RKNN] Output shapes:", [np.asarray(x).shape for x in outputs])
                printed_shapes = True

            detections = postprocess(outputs, frame.shape)
            for x1, y1, x2, y2, cls, conf in detections:
                boat_detections.append({
                    "box": tuple(map(int, (x1, y1, x2, y2))),
                    "class": class_name_from_id(int(cls)),
                    "confidence": float(conf),
                })

            # ArUco ищем только внутри YOLO-детекций судов.
            # Crop увеличивается и проходит CLAHE, что повышает читаемость
            # мелких/контрастно сложных меток без обработки всего кадра.
            aruco_items = detect_aruco_in_boat_crops(frame, boat_detections)

            for marker in aruco_items:
                marker_id = int(marker["id"])
                matched_class = str(marker["matched_class"])

                new_marker = False

                with aruco_lock:
                    if marker_id not in aruco_ids_set:
                        aruco_ids_set.add(marker_id)
                        aruco_ids.append(marker_id)
                        new_marker = True

                    if new_marker:
                        kind = class_kind(matched_class)
                        print(f"Обнаружено судно с id {marker_id}")

                        if kind == "unregistered":
                            unregistered_boat_ids.add(marker_id)
                            print("Обнаружено незарегистрированное судно")
                        elif kind == "registered":
                            registered_boat_ids.add(marker_id)
                            print("Обнаружено зарегистрированное судно")
                        else:
                            print(f"Обнаружено судно класса «{matched_class}»")

        except Exception as error:
            print("[DETECT ERROR]", error)

        inference_ms = (time.perf_counter() - started) * 1000.0

        with state_lock:
            latest_result = {
                "frame_id": frame_id,
                "detections": boat_detections,
                "aruco": aruco_items,
                "inference_ms": inference_ms,
            }


def draw_overlay(frame, result):
    """Быстрая отрисовка последних результатов детектора."""
    detections = result.get("detections", [])
    aruco_items = result.get("aruco", [])

    for boat in detections:
        draw_box_label(
            frame,
            boat["box"],
            boat["class"],
            boat["confidence"],
        )

    for marker in aruco_items:
        points = np.asarray(marker["points"], dtype=np.int32)
        center_x, center_y = marker["center"]

        cv2.polylines(
            frame,
            [points.reshape((-1, 1, 2))],
            True,
            COLOR_ARUCO,
            2,
        )
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        label = f"ARUCO ID: {marker['id']}"
        if marker.get("matched_class"):
            label += f" [{marker['matched_class']}]"

        top_left = points[0]
        cv2.putText(
            frame,
            label,
            (int(top_left[0]), max(20, int(top_left[1]) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_ARUCO,
            2,
        )

    with aruco_lock:
        aruco_count = len(aruco_ids)
        ids_text = ",".join(map(str, aruco_ids))

    cv2.rectangle(frame, (10, 10), (580, 155), (0, 0, 0), -1)
    cv2.putText(frame, "PIONEER CAMERA", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"BOATS: {len(detections)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"UNIQUE BOATS / ARUCO: {aruco_count}", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, f"AI: {result.get('inference_ms', 0.0):.0f} ms", (20, 128),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if ids_text:
        cv2.putText(frame, f"IDs: {ids_text}", (220, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    return frame


def stream_worker():
    """Поток №3: отрисовка + web-stream. Не ждёт RKNN и не ждёт запись."""
    global running

    viewer = None
    last_streamed_id = -1

    try:
        viewer = ImageViewer()

        print("=" * 60)
        print("Видеопоток запущен")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}")
        print("=" * 60)

        while running:
            with state_lock:
                frame_id = latest_frame_id
                if latest_frame is None or frame_id == last_streamed_id:
                    frame = None
                    result = None
                else:
                    frame = latest_frame.copy()
                    # Копируем только контейнеры результатов, изображения здесь нет.
                    result = {
                        "frame_id": latest_result.get("frame_id", -1),
                        "detections": list(latest_result.get("detections", [])),
                        "aruco": list(latest_result.get("aruco", [])),
                        "inference_ms": latest_result.get("inference_ms", 0.0),
                    }

            if frame is None:
                time.sleep(0.002)
                continue

            last_streamed_id = frame_id
            draw_overlay(frame, result)

            # Трансляция не ждёт диск.
            viewer.imshow(STREAM_NAME, frame, fps=STREAM_FPS)

            # В запись отправляется уже размеченный кадр.
            put_latest(record_queue, frame.copy())

    except Exception as error:
        print("[STREAM ERROR]", error)
        running = False
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        print("[STREAM] Остановлен")


def record_worker():
    """
    Поток №4: запись размеченного видео независимо от трансляции.

    ВАЖНО:
    - пишем сначала во временный flight.part.mp4;
    - после release() атомарно переименовываем его в flight.mp4;
    - record_finished выставляется только после полной финализации контейнера.
    """
    global running

    writer = None
    frame_count = 0
    video_size = None

    # Старый временный файл от предыдущего аварийного запуска не используем.
    try:
        if os.path.exists(VIDEO_TEMP_OUTPUT):
            os.remove(VIDEO_TEMP_OUTPUT)
    except Exception as error:
        print("[RECORD] Не удалось удалить старый временный файл:", error)

    try:
        # После running=False обязательно дописываем всю уже накопленную очередь.
        while running or not record_queue.empty():
            try:
                frame = record_queue.get(timeout=0.2)
            except Empty:
                continue

            if writer is None:
                height, width = frame.shape[:2]
                video_size = (width, height)

                writer = cv2.VideoWriter(
                    VIDEO_TEMP_OUTPUT,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    VIDEO_FPS,
                    video_size,
                )

                if not writer.isOpened():
                    raise RuntimeError(
                        f"Не удалось открыть файл записи: {VIDEO_TEMP_OUTPUT}"
                    )

                print(
                    f"[RECORD] Запись начата: {VIDEO_TEMP_OUTPUT} "
                    f"{width}x{height} @ {VIDEO_FPS} FPS"
                )

            # На случай неожиданного изменения разрешения камеры.
            if frame.shape[1] != video_size[0] or frame.shape[0] != video_size[1]:
                frame = cv2.resize(frame, video_size, interpolation=cv2.INTER_LINEAR)

            writer.write(frame)
            frame_count += 1

    except Exception as error:
        print("[RECORD ERROR]", error)

    finally:
        # Критически важно: release записывает служебные таблицы MP4 (moov atom).
        if writer is not None:
            try:
                writer.release()
                writer = None
                print(f"[RECORD] VideoWriter закрыт, кадров: {frame_count}")
            except Exception as error:
                print("[RECORD] Ошибка закрытия VideoWriter:", error)

        # Только полностью закрытый файл становится итоговым flight.mp4.
        try:
            if frame_count > 0 and os.path.exists(VIDEO_TEMP_OUTPUT):
                file_size = os.path.getsize(VIDEO_TEMP_OUTPUT)

                if file_size <= 0:
                    raise RuntimeError("Временный видеофайл имеет нулевой размер")

                os.replace(VIDEO_TEMP_OUTPUT, VIDEO_OUTPUT)

                print(
                    f"[RECORD] Видео финализировано: {VIDEO_OUTPUT} "
                    f"({file_size / 1024 / 1024:.1f} MB)"
                )

            elif frame_count == 0:
                print("[RECORD] Нет записанных кадров — видео не создано")

        except Exception as error:
            print("[RECORD] Ошибка финализации видео:", error)
            print(
                f"[RECORD] Временный файл оставлен как: {VIDEO_TEMP_OUTPUT}"
            )

        finally:
            record_finished.set()


# ============================================================
# МАРШРУТ ПОЛЁТА — СОХРАНЁН ИЗ ИСХОДНОГО КОДА
# ============================================================

route = [
    (0.0, 1.0, 1.7, 0.0),

    (-1.6, 0.5, 2.2, 0.0),
    (-1.6, 1.0, 2.2, 0.0),
    (-1.6, 2.0, 2.2, 0.0),
    (-1.6, 2.5, 2.2, 0.0),
    (-1.6, 4.0, 2.2, 0.0),

    (-2.0, 4.0, 2.2, 0.0),
    (-2.0, 2.5, 2.2, 0.0),
    (-2.0, 2.0, 2.2, 0.0),
    (-2.0, 1.0, 2.2, 0.0),
    (-2.0, 0.5, 2.2, 0.0),

    (-2.4, 0.5, 2.2, 0.0),
    (-2.4, 1.0, 2.2, 0.0),
    (-2.4, 2.0, 2.2, 0.0),
    (-2.4, 2.5, 2.2, 0.0),
    (-2.4, 4.0, 2.2, 0.0),


    (-2.8, 4.0, 2.2, 0.0),
    (-2.8, 2.5, 2.2, 0.0),
    (-2.8, 2.0, 2.2, 0.0),
    (-2.8, 1.0, 2.2, 0.0),
    (-2.8, 0.5, 2.2, 0.0),

    (-3.1, 0.5, 2.2, 0.0),
    (-3.1, 1.0, 2.2, 0.0),
    (-3.1, 2.0, 2.2, 0.0),
    (-3.1, 2.5, 2.2, 0.0),
    (-3.1, 4.0, 2.2, 0.0),


    (0.0, 1.0, 1.7, 0.0)
    ]




def stop_and_finalize_video(threads):
    """
    Корректно останавливает видеоконвейер и обязательно ждёт release VideoWriter.

    Сначала останавливаются производители кадров, затем record_worker дописывает
    очередь и закрывает MP4. Это предотвращает битый flight.mp4.
    """
    global running

    if not running and record_finished.is_set():
        return

    print("[VIDEO] Остановка видеоконвейера...")
    running = False

    # Сначала ждём camera/detect/stream. Они больше не должны добавлять кадры.
    for thread in threads:
        if thread.name in ("camera", "detect", "stream") and thread.is_alive():
            thread.join(timeout=5.0)

    # record_worker должен завершиться ПОСЛЕДНИМ.
    record_thread = next(
        (thread for thread in threads if thread.name == "record"),
        None,
    )

    if record_thread is not None and record_thread.is_alive():
        print("[RECORD] Дописываем очередь и финализируем MP4...")
        record_thread.join()  # намеренно без timeout

    # Дополнительная страховка на случай нестандартного выхода record_worker.
    record_finished.wait(timeout=2.0)

    print("[VIDEO] Видеозапись полностью завершена")


# ============================================================
# MAIN + ОТДЕЛЬНЫЙ ПОТОК ПОЛЁТА
# ============================================================

pioneer = None
flight_exception = None
mission_done = threading.Event()


def flight_worker():
    """Поток №5: автономный полёт полностью отдельно от видео и AI."""
    global pioneer, running, flight_exception

    try:
        print("Подключение к Pioneer...")
        pioneer = Pioneer()
        print("Pioneer подключен")

        print("ARM...")
        pioneer.arm()
        print("ARM: OK")

        print("Взлёт...")
        pioneer.takeoff()
        if not running:
            return

        print("Взлёт завершён")

        for x, y, z, yaw in route:
            if not running:
                return

            print(f"Точка: x={x}, y={y}, z={z}, yaw={yaw}")
            pioneer.go_to_local_point(x, y, z, yaw)

            while running and not pioneer.point_reached():
                time.sleep(0.06)

            if not running:
                return

            print(f"Точка достигнута: x={x}, y={y}, z={z}")

        print("Посадка...")
        pioneer.land()
        while running and not pioneer.point_reached():
            time.sleep(0.05)

        pioneer.disarm()
        print("Миссия завершена. Посадка выполнена.")
        mission_done.set()

    except Exception as error:
        flight_exception = error
        print("[FLIGHT ERROR]", error)
        mission_done.set()


def emergency_land():
    if pioneer is None:
        return
    try:
        print("Аварийная посадка...")
        pioneer.land()
        time.sleep(3)
        pioneer.disarm()
        print("Аппарат посажен.")
    except Exception as error:
        print("Ошибка аварийной посадки:", error)


threads = []

try:
    threads = [
        threading.Thread(target=camera_worker, name="camera", daemon=True),
        threading.Thread(target=detection_worker, name="detect", daemon=True),
        threading.Thread(target=stream_worker, name="stream", daemon=True),
        threading.Thread(target=record_worker, name="record", daemon=False),
    ]

    for thread in threads:
        thread.start()

    # Даём камере начать отдавать кадры до взлёта.
    time.sleep(1.0)

    flight_thread = threading.Thread(target=flight_worker, name="flight", daemon=True)
    threads.append(flight_thread)
    flight_thread.start()

    # Главный поток только контролирует завершение и Ctrl+C.
    while running and not mission_done.is_set():
        time.sleep(0.1)

    if flight_exception is not None:
        raise flight_exception

except KeyboardInterrupt:
    print("\nОстановка оператором")
    running = False
    emergency_land()

except Exception as error:
    print("\nКРИТИЧЕСКАЯ ОШИБКА:", error)
    running = False
    emergency_land()

finally:
    # Штатная посадка, Ctrl+C и любое Python-исключение сходятся сюда.
    # Сначала гарантированно финализируем видеозапись.
    stop_and_finalize_video(threads)

    # Flight-поток тоже корректно дожидаемся, если он ещё завершает обработку.
    for thread in threads:
        if thread.name == "flight" and thread.is_alive():
            thread.join(timeout=5.0)

    # RKNN освобождаем только после остановки detection_worker.
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
