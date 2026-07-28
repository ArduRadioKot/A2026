# Установка:
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# =========================
# НАСТРОЙКИ
# =========================

MODEL_PATH = "best.pt"

# 0 — встроенная/основная камера ноутбука
# Если не работает, попробуй 1 или 2
CAMERA_ID = 0

CONFIDENCE = 0.35
IMAGE_SIZE = 640


# =========================
# ЗАГРУЗКА МОДЕЛИ
# =========================

print("Загрузка модели...")

model = YOLO(MODEL_PATH)

print("Модель загружена!")
print("Классы:", model.names)


# =========================
# ЗАПУСК КАМЕРЫ
# =========================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    raise RuntimeError(
        "Не удалось открыть камеру. "
        "Попробуй CAMERA_ID = 1 или CAMERA_ID = 2"
    )

# Запрашиваем разрешение камеры
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Камера запущена")
print("Нажми Q или ESC для выхода")


# =========================
# ОСНОВНОЙ ЦИКЛ
# =========================

while True:

    ret, frame = cap.read()

    if not ret or frame is None:
        print("Не удалось получить кадр")
        continue

    # -------------------------
    # YOLO
    # -------------------------

    results = model.predict(
        source=frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]

    # -------------------------
    # ОБРАБОТКА ОБЪЕКТОВ
    # -------------------------

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # Центр объекта
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        class_name = model.names[class_id]

        # Вывод в терминал
        print(
            f"{class_name} | "
            f"conf={confidence:.2f} | "
            f"center=({center_x}, {center_y})"
        )

    # -------------------------
    # РИСУЕМ YOLO РАЗМЕТКУ
    # -------------------------

    output_frame = result.plot()

    # Центр кадра
    frame_height, frame_width = output_frame.shape[:2]

    camera_center_x = frame_width // 2
    camera_center_y = frame_height // 2

    cv2.drawMarker(
        output_frame,
        (camera_center_x, camera_center_y),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2
    )

    # FPS
    inference_ms = result.speed.get("inference", 0)

    if inference_ms > 0:
        fps = 1000 / inference_ms
    else:
        fps = 0

    cv2.putText(
        output_frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # -------------------------
    # ПОКАЗ КАДРА
    # -------------------------

    cv2.imshow(
        "YOLO Boat Detection",
        output_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Q или ESC
    if key == ord("q") or key == 27:
        break


# =========================
# ЗАВЕРШЕНИЕ
# =========================

cap.release()
cv2.destroyAllWindows()

print("Программа завершена")