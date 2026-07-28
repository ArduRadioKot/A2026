import time
import threading
import cv2
import numpy as np

from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera
from rknnlite.api import RKNNLite


STREAM_NAME = "pioneer"

RKNN_MODEL = "best-rk3576.rknn"
RKNN_CONF = 0.6
RKNN_SIZE = 640
NMS_IOU = 0.45
CLASS_NAMES = ["boat"]


running = True


print("[RKNN] Загрузка модели...")

rknn = RKNNLite()

ret = rknn.load_rknn(RKNN_MODEL)
if ret != 0:
    raise RuntimeError("Не удалось загрузить RKNN модель")

ret = rknn.init_runtime()
if ret != 0:
    raise RuntimeError("Не удалось запустить RKNN runtime")

print("[RKNN] Модель загружена")


aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    cv2.aruco.DetectorParameters()
)

aruco_ids = []
aruco_ids_set = set()
aruco_lock = threading.Lock()


servo_camera = ServoCamera()
servo_camera.set_angle(-80)


def preprocess(frame):
    """Подготовка кадра для RKNN YOLO11."""
    img = cv2.resize(frame, (RKNN_SIZE, RKNN_SIZE), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.ascontiguousarray(img)
    img = np.expand_dims(img, 0)
    return img


def xywh_to_xyxy(boxes):
    """YOLO bbox: center_x, center_y, width, height -> x1,y1,x2,y2."""
    result = boxes.copy()
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def box_iou_one_to_many(box, boxes):
    """IoU одного bbox со списком bbox."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - intersection

    return intersection / np.maximum(union, 1e-6)


def numpy_nms(boxes, scores, iou_threshold=0.45):
    """NMS только на NumPy."""
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


def _flatten_yolo_output(outputs):
    """
    Приводит типичный экспорт Ultralytics YOLO11 к массиву [N, 4 + classes].

    Поддерживаются наиболее частые варианты:
      [1, C, N]
      [1, N, C]
      [C, N]
      [N, C]

    Для RKNN, который вернул несколько выходов, сначала ищем выход, похожий
    на уже декодированный Ultralytics prediction tensor.
    """
    candidates = []

    for output in outputs:
        arr = np.asarray(output)
        arr = np.squeeze(arr)

        if arr.ndim != 2:
            continue

        # У YOLO число каналов (4 + classes) обычно сильно меньше числа boxes.
        if arr.shape[0] < arr.shape[1] and 5 <= arr.shape[0] <= 512:
            arr = arr.T

        if arr.shape[1] >= 5 and arr.shape[1] <= 512:
            candidates.append(arr.astype(np.float32, copy=False))

    if not candidates:
        shapes = [np.asarray(x).shape for x in outputs]
        raise RuntimeError(
            "Не удалось распознать формат выхода RKNN. "
            f"Получены shapes={shapes}. "
            "Выведи эти shapes — для multi-output DFL экспорта нужен другой декодер."
        )

    # Обычно нужный tensor содержит больше всего предсказаний.
    return max(candidates, key=lambda x: x.shape[0])


def postprocess(outputs, original_shape):
    """
    Декодирование YOLO11 Ultralytics output.

    Возвращает список:
        [x1, y1, x2, y2, class_id, confidence]
    уже в координатах исходного кадра.
    """
    pred = _flatten_yolo_output(outputs)

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
    scores = scores[mask]
    class_ids = class_ids[mask]

    boxes = xywh_to_xyxy(boxes_xywh)

    # Некоторые экспорты могут отдавать нормализованные координаты 0..1.
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
            detections.append([
                float(x1), float(y1), float(x2), float(y2),
                int(class_ids[i]), float(scores[i])
            ])

    detections.sort(key=lambda x: x[5], reverse=True)
    return detections


def video_worker():

    global running

    camera = None
    viewer = None

    try:
        camera = Camera()
        viewer = ImageViewer()

        print("=" * 50)
        print("Стрим запущен")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}")
        print("=" * 50)

        while running:

            frame = camera.get_cv_frame(timeout=2)

            if frame is None:
                continue


            # RKNN inference

            input_data = preprocess(frame)

            outputs = rknn.inference(
                inputs=[input_data]
            )

            if not hasattr(video_worker, "printed_output_shapes"):
                print("[RKNN] Output shapes:", [np.asarray(x).shape for x in outputs])
                video_worker.printed_output_shapes = True

            detections = postprocess(outputs, frame.shape)


            boat_count = 0


            for det in detections:

                x1, y1, x2, y2, cls, conf = det

                if conf < RKNN_CONF:
                    continue

                boat_count += 1

                class_name = CLASS_NAMES[int(cls)] if int(cls) < len(CLASS_NAMES) else f"class_{int(cls)}"

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (255,0,0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{class_name} {conf:.2f}",
                    (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2
                )


            # ArUco

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, _ = aruco_detector.detectMarkers(gray)

            if ids is not None:

                for marker_id in ids.flatten():

                    marker_id = int(marker_id)

                    with aruco_lock:

                        if marker_id not in aruco_ids_set:
                            aruco_ids_set.add(marker_id)
                            aruco_ids.append(marker_id)

                            print(
                                "[ARUCO] Новая метка:",
                                marker_id
                            )


            cv2.putText(
                frame,
                f"BOATS: {boat_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )


            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=20
            )


    finally:

        if camera:
            camera.stop()

        if viewer:
            viewer.close()


pioneer = None


try:

    print("Подключение Pioneer...")

    pioneer = Pioneer()

    thread = threading.Thread(
        target=video_worker,
        daemon=True
    )

    thread.start()


    while True:

        time.sleep(0.1)


except KeyboardInterrupt:

    print("Остановка")


finally:

    running = False

    rknn.release()

    if pioneer:
        pioneer.close_connection()

    print("Завершено")

# NMS replaced manually: use numpy implementation instead of cv2.dnn.NMSBoxes
