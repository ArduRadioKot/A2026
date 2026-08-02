#!/usr/bin/env python3

import time
import cv2
import numpy as np
import rospy
import threading
import os
from pathlib import Path

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from user.library import DroneLibrary
from flask import Flask, Response


# ============================================================
# НАСТРОЙКИ ИИ-МОДЕЛИ
# ============================================================

# Папка с экспортированными .npy-весами.
# По умолчанию должна находиться рядом с этим скриптом.
WEIGHTS_DIR = Path(__file__).resolve().parent / "digits64_numpy"

MODEL_INPUT_SIZE = (64, 64)
BN_EPS = 1e-5

# Частота фонового распознавания. Для Raspberry Pi 4 разумно 3–6 Гц.
AI_INTERVAL = 0.20

# Минимальная уверенность, при которой цифра считается прочитанной.
AI_MIN_CONFIDENCE = 0.35

# Сколько одинаковых результатов подряд нужно для подтверждения.
AI_CONFIRM_FRAMES = 4

# Подготовка изображения цифры.
DIGIT_PADDING = 8
MIN_DIGIT_AREA = 8

# Эвристику 1/3/7 оставляем выключенной: чистый crop обычно надёжнее.
ENABLE_137_CORRECTION = False
ONE_MAX_ASPECT = 0.38
SEVEN_MAX_ASPECT = 0.78
ONE_MIN_RELATIVE_PROB = 0.22
SEVEN_MIN_RELATIVE_PROB = 0.30


class FastDigitModel:
    """
    Быстрый inference без PyTorch.

    Ускорения:
    1. BatchNorm заранее объединяется с Conv2d.
    2. Свёртки выполняются через im2col + BLAS matrix multiplication.
    3. Нет тысяч вызовов cv2.filter2D на каждый кадр.
    4. Модель запускается только один раз для каждого crop.
    """

    def __init__(self, weights_dir):
        self.weights_dir = Path(weights_dir)

        if not self.weights_dir.is_dir():
            raise FileNotFoundError(
                f"Папка с весами не найдена: {self.weights_dir.resolve()}"
            )

        self.w = {}
        required = [
            "features.0.weight",
            "features.0.bias",
            "features.1.weight",
            "features.1.bias",
            "features.1.running_mean",
            "features.1.running_var",

            "features.4.weight",
            "features.4.bias",
            "features.5.weight",
            "features.5.bias",
            "features.5.running_mean",
            "features.5.running_var",

            "features.8.weight",
            "features.8.bias",
            "features.9.weight",
            "features.9.bias",
            "features.9.running_mean",
            "features.9.running_var",

            "classifier.1.weight",
            "classifier.1.bias",
            "classifier.4.weight",
            "classifier.4.bias",
        ]

        missing = []

        for name in required:
            file_path = self.weights_dir / f"{name}.npy"

            if not file_path.exists():
                missing.append(file_path.name)
                continue

            self.w[name] = np.ascontiguousarray(
                np.load(file_path).astype(np.float32, copy=False)
            )

        if missing:
            raise FileNotFoundError(
                "Не найдены файлы весов:\n  - " + "\n  - ".join(missing)
            )

        # Объединяем Conv + BatchNorm один раз при запуске.
        self.conv1_w, self.conv1_b = self._fuse_conv_bn(
            "features.0", "features.1"
        )
        self.conv2_w, self.conv2_b = self._fuse_conv_bn(
            "features.4", "features.5"
        )
        self.conv3_w, self.conv3_b = self._fuse_conv_bn(
            "features.8", "features.9"
        )

        self.fc1_w = self.w["classifier.1.weight"]
        self.fc1_b = self.w["classifier.1.bias"]
        self.fc2_w = self.w["classifier.4.weight"]
        self.fc2_b = self.w["classifier.4.bias"]

        self._validate_shapes()

        # Веса свёрток сразу преобразуем в матрицы для GEMM.
        self.conv1_matrix = np.ascontiguousarray(
            self.conv1_w.reshape(self.conv1_w.shape[0], -1).T
        )
        self.conv2_matrix = np.ascontiguousarray(
            self.conv2_w.reshape(self.conv2_w.shape[0], -1).T
        )
        self.conv3_matrix = np.ascontiguousarray(
            self.conv3_w.reshape(self.conv3_w.shape[0], -1).T
        )

        # Буфер для последнего результата.
        self.last_confidence = 0.0
        self.last_probabilities = None

    def _fuse_conv_bn(self, conv_name, bn_name):
        conv_w = self.w[f"{conv_name}.weight"]
        conv_b = self.w[f"{conv_name}.bias"]

        gamma = self.w[f"{bn_name}.weight"]
        beta = self.w[f"{bn_name}.bias"]
        mean = self.w[f"{bn_name}.running_mean"]
        var = self.w[f"{bn_name}.running_var"]

        scale = gamma / np.sqrt(var + BN_EPS)

        fused_w = conv_w * scale[:, None, None, None]
        fused_b = beta + (conv_b - mean) * scale

        return (
            np.ascontiguousarray(fused_w.astype(np.float32)),
            np.ascontiguousarray(fused_b.astype(np.float32)),
        )

    def _validate_shapes(self):
        expected = {
            "conv1": ((32, 1, 3, 3), (32,)),
            "conv2": ((64, 32, 3, 3), (64,)),
            "conv3": ((128, 64, 3, 3), (128,)),
            "fc1": ((256, 8192), (256,)),
            "fc2": ((10, 256), (10,)),
        }

        actual = {
            "conv1": (self.conv1_w.shape, self.conv1_b.shape),
            "conv2": (self.conv2_w.shape, self.conv2_b.shape),
            "conv3": (self.conv3_w.shape, self.conv3_b.shape),
            "fc1": (self.fc1_w.shape, self.fc1_b.shape),
            "fc2": (self.fc2_w.shape, self.fc2_b.shape),
        }

        errors = []

        for name, shape in expected.items():
            if actual[name] != shape:
                errors.append(
                    f"{name}: ожидалось {shape}, найдено {actual[name]}"
                )

        if errors:
            raise ValueError(
                "Формы весов не соответствуют архитектуре:\n  - "
                + "\n  - ".join(errors)
            )

    @staticmethod
    def _conv3x3_same(x, weight_matrix, bias):
        """
        x: (C, H, W)
        weight_matrix: (C*3*3, OUT_C)

        Аналог Conv2d(kernel=3, padding=1, stride=1).
        """
        channels, height, width = x.shape

        padded = np.pad(
            x,
            ((0, 0), (1, 1), (1, 1)),
            mode="constant",
        )

        # (C, H, W, 3, 3)
        windows = np.lib.stride_tricks.sliding_window_view(
            padded,
            (3, 3),
            axis=(1, 2),
        )

        # Порядок элементов совпадает с flatten весов:
        # channel -> kernel_y -> kernel_x
        columns = np.ascontiguousarray(
            windows.transpose(1, 2, 0, 3, 4).reshape(
                height * width,
                channels * 9,
            )
        )

        # (H*W, OUT_C)
        out = columns @ weight_matrix
        out += bias

        return out.reshape(height, width, -1).transpose(2, 0, 1)

    @staticmethod
    def _relu(x):
        np.maximum(x, 0.0, out=x)
        return x

    @staticmethod
    def _max_pool2x2(x):
        channels, height, width = x.shape

        # В модели размеры всегда чётные.
        return x.reshape(
            channels,
            height // 2,
            2,
            width // 2,
            2,
        ).max(axis=(2, 4))

    @staticmethod
    def _softmax(logits):
        shifted = logits - np.max(logits)
        values = np.exp(shifted)
        return values / np.sum(values)

    @staticmethod
    def _prepare_image(img):
        if img is None or img.size == 0:
            raise ValueError("В модель передано пустое изображение.")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(
            img,
            MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_AREA,
        )

        x = img.astype(np.float32) / 255.0
        x = (x - 0.1307) / 0.3081

        return np.ascontiguousarray(x[None, :, :])

    def predict(self, img):
        x = self._prepare_image(img)

        x = self._conv3x3_same(x, self.conv1_matrix, self.conv1_b)
        x = self._max_pool2x2(self._relu(x))

        x = self._conv3x3_same(x, self.conv2_matrix, self.conv2_b)
        x = self._max_pool2x2(self._relu(x))

        x = self._conv3x3_same(x, self.conv3_matrix, self.conv3_b)
        x = self._max_pool2x2(self._relu(x))

        x = np.ascontiguousarray(x.reshape(-1))

        x = self.fc1_w @ x + self.fc1_b
        self._relu(x)

        logits = self.fc2_w @ x + self.fc2_b
        probabilities = self._softmax(logits)

        cls = int(np.argmax(probabilities))

        # Геометрическая коррекция применяется только если сеть выбрала 3
        # и при этом классы 1 или 7 имеют заметную вероятность.
        if ENABLE_137_CORRECTION and cls == 3:
            points = cv2.findNonZero((img > 127).astype(np.uint8) * 255)

            if points is not None:
                _, _, digit_w, digit_h = cv2.boundingRect(points)
                aspect = digit_w / max(digit_h, 1)

                p1 = float(probabilities[1])
                p3 = float(probabilities[3])
                p7 = float(probabilities[7])

                if (
                    aspect <= ONE_MAX_ASPECT
                    and p1 >= p3 * ONE_MIN_RELATIVE_PROB
                ):
                    cls = 1
                elif (
                    aspect <= SEVEN_MAX_ASPECT
                    and p7 >= p3 * SEVEN_MIN_RELATIVE_PROB
                ):
                    cls = 7

        confidence = float(probabilities[cls])

        self.last_confidence = confidence
        self.last_probabilities = probabilities

        return cls


# ============================================================
# ВЕБ-СЕРВЕР ДЛЯ ВИДЕОПОТОКА (FLASK)
# ============================================================
app = Flask(__name__)
frame_lock = threading.Lock()
latest_frame_for_web = None
system_status = "Инициализация"  # Статус для отображения на видео

# Состояние фонового распознавания.
ai_lock = threading.Lock()
model_inference_lock = threading.Lock()
digit_model = None
ai_current_class = None
ai_current_confidence = 0.0
ai_confirmed_class = None
ai_last_crop = None
ai_last_probabilities = None
ai_running = True

def process_frame_for_display(frame):
    """
    Обрабатывает кадр для отображения в браузере:
    - Рисует рамку вокруг кубика
    - Выводит координаты и размер
    - Показывает статус системы
    """
    if frame is None:
        return None
    
    # Копируем кадр, чтобы не портить оригинал
    display_frame = frame.copy()
    
    # 1. Детекция кубика для отрисовки
    detection = detect_cube(display_frame)
    
    if detection is not None:
        x, y, w, h = detection["x"], detection["y"], detection["w"], detection["h"]
        cx, cy = detection["cx"], detection["cy"]
        area = detection["area"]
        
        # Рисуем зеленый прямоугольник вокруг кубика
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Рисуем точку в центре
        cv2.circle(display_frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        
        # Выводим информацию об объекте
        info_text = f"Object: ({int(cx)}, {int(cy)}) {w}x{h} Area:{int(area)}"
        cv2.putText(display_frame, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(display_frame, "No object detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 2. Статус системы (в правом верхнем углу)
    h, w = display_frame.shape[:2]
    status_color = (0, 255, 0) if "РАСПОЗНАН" in system_status or system_status == "КУБИК НАЙДЕН" else (255, 255, 0)
    cv2.putText(display_frame, f"Status: {system_status}", (w - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # 3. Результат ИИ-распознавания.
    with ai_lock:
        current_digit = ai_current_class
        current_confidence = ai_current_confidence
        confirmed_digit = ai_confirmed_class

    if current_digit is None:
        ai_text = "AI digit: --"
        ai_color = (0, 165, 255)
    else:
        ai_text = f"AI digit: {current_digit}  {current_confidence * 100:.1f}%"
        ai_color = (0, 255, 0) if current_confidence >= AI_MIN_CONFIDENCE else (0, 165, 255)

    cv2.putText(
        display_frame,
        ai_text,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        ai_color,
        2,
        cv2.LINE_AA,
    )

    if confirmed_digit is not None:
        cv2.putText(
            display_frame,
            f"CONFIRMED DIGIT: {confirmed_digit}",
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 4. Разрешение кадра (в правом нижнем углу)
    cv2.putText(display_frame, f"{w}x{h}", (w - 80, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return display_frame


def generate_frames():
    """Генератор кадров для MJPEG потока в браузер"""
    global latest_frame_for_web
    while True:
        with frame_lock:
            if latest_frame_for_web is not None:
                frame_copy = latest_frame_for_web.copy()
            else:
                frame_copy = None
        
        if frame_copy is not None:
            # Обрабатываем кадр (рисуем рамки, текст и т.д.)
            processed_frame = process_frame_for_display(frame_copy)
            
            if processed_frame is not None:
                # Сжимаем в JPEG (качество 70% для баланса скорость/четкость)
                ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)  # ~20 FPS

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_ai_crop_frames():
    """MJPEG-поток точного изображения 64x64, которое получает модель."""
    while True:
        with ai_lock:
            crop_frame = None if ai_last_crop is None else ai_last_crop.copy()

        if crop_frame is not None:
            preview = cv2.resize(
                crop_frame,
                (320, 320),
                interpolation=cv2.INTER_NEAREST,
            )
            preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)

            ret, buffer = cv2.imencode(
                ".jpg",
                preview,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            if ret:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buffer.tobytes()
                    + b"\r\n"
                )

        time.sleep(0.10)


@app.route('/ai_crop')
def ai_crop_feed():
    return Response(
        generate_ai_crop_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )

def start_web_server():
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)


# ============================================================
# НАСТРОЙКИ ЦВЕТА И КАМЕРЫ
# ============================================================
LOWER_HSV = np.array([15, 80, 80])
UPPER_HSV = np.array([40, 255, 255])

IMG_W = 320
IMG_H = 240
IMG_CX = IMG_W // 2       # 160
IMG_CY = IMG_H // 2       # 120


# ============================================================
# НАСТРОЙКИ ПОИСКА (ЗМЕЙКА)
# ============================================================
SEARCH_DEPTH = 0.35
SEARCH_PITCH = -15
SEARCH_SPEED = 25
APPROACH_SPEED = 15

LANES = 5                  # Количество полос для змейки
LONG_TIME = 12.0           # Время поиска на одной полосе (сек)
SHIFT_TIME = 2.0           # Время перехода на соседнюю полосу (сек)

MIN_AREA = 40
LOCK_FRAMES = 3

# Параметры повторного поиска
REACQUIRE_ENABLED = True
REACQUIRE_TIMEOUT = 0.7    # Время ожидания перед повторным поиском (сек)
REACQUIRE_SCAN_TIME = 0.5  # Время на проверку каждого угла (сек)


# ============================================================
# НАСТРОЙКИ P-РЕГУЛЯТОРА ЦЕНТРОВКИ
# ============================================================
# "Мертвая зона" в пикселях. Если отклонение меньше этого значения, 
# дрон считает, что кубик уже по центру, и не делает поправок.
CENTER_THRESHOLD_X = 15  # Допустимое отклонение по горизонтали (X)
CENTER_THRESHOLD_Y = 15  # Допустимое отклонение по вертикали (Y)

# Коэффициенты усиления P-регулятора
YAW_KP = 15.0            # Коэффициент для рыскания (Yaw)
PITCH_KP = 12.0          # Коэффициент для тангажа (Pitch)

# Ограничения на величину поправки за один шаг (для плавности)
MAX_YAW_CORRECTION = 15.0    # Макс. изменение курса за шаг (градусы)
MAX_PITCH_CORRECTION = 10.0  # Макс. изменение тангажа за шаг (градусы)

# Абсолютные пределы тангажа (чтобы дрон не задрал камеру вертикально вверх/вниз)
MIN_PITCH = -30.0
MAX_PITCH = 30.0


# ============================================================
# НАСТРОЙКИ КЛАССИФИКАЦИИ И ПРИБЛИЖЕНИЯ
# ============================================================
CENTER_CONFIRM_FRAMES = 5      # Сколько кадров подряд кубик должен быть в центре для завершения центровки
CLASSIFY_CONFIRM_FRAMES = 5    # Сколько кадров подряд один класс для завершения приближения
MIN_SIZE_FOR_CLASSIFY = 30     # Минимальный размер кубика (px) для начала классификации


bridge = CvBridge()
latest_frame = None
latest_frame_time = 0.0

# Глобальная переменная для хранения распознанного класса
detected_class = None


# ============================================================
# ФУНКЦИИ ЛАЗЕРА
# ============================================================
def turn_on_laser(drone):
    """Включает лазер и оставляет его включённым постоянно"""
    drone.set_laser(1)
    print("[LASER] Лазер включён")


# ============================================================
# ИИ-КЛАССИФИКАТОР
# ============================================================

def classify_frame(frame, detection=None):
    """
    Распознаёт цифру внутри найденного цветного объекта.

    Возвращает:
        int или None
    """
    global digit_model

    if digit_model is None or frame is None:
        return None

    if detection is None:
        detection = detect_cube(frame)

    if detection is None:
        return None

    # Используем настоящий HSV-контур — подготовка полностью совпадает
    # с отдельным тестовым распознавателем.
    contour = detection["contour"]
    crop_img, _ = crop(frame, contour)

    with model_inference_lock:
        predicted_class = digit_model.predict(crop_img)
        confidence = digit_model.last_confidence

    if confidence < AI_MIN_CONFIDENCE:
        return None

    return predicted_class


def ai_recognition_worker():
    """
    Фоновый поток распознавания.

    Камера и управление дроном не блокируются тяжёлым inference.
    """
    global ai_current_class
    global ai_current_confidence
    global ai_confirmed_class
    global ai_last_crop
    global ai_last_probabilities
    global ai_running

    previous_class = None
    same_count = 0

    while ai_running and not rospy.is_shutdown():
        with frame_lock:
            frame = None if latest_frame_for_web is None else latest_frame_for_web.copy()

        if frame is None or digit_model is None:
            time.sleep(AI_INTERVAL)
            continue

        detection = detect_cube(frame)

        if detection is None:
            with ai_lock:
                ai_current_class = None
                ai_current_confidence = 0.0
                ai_last_crop = None
                ai_last_probabilities = None

            previous_class = None
            same_count = 0
            time.sleep(AI_INTERVAL)
            continue

        contour = detection["contour"]

        try:
            # Та же подготовка изображения, что и в тестовом распознавателе.
            crop_img, _ = crop(frame, contour)

            with model_inference_lock:
                predicted = digit_model.predict(crop_img)
                confidence = digit_model.last_confidence
                probabilities = digit_model.last_probabilities.copy()

            valid_prediction = (
                predicted if confidence >= AI_MIN_CONFIDENCE else None
            )

            if valid_prediction is not None:
                if valid_prediction == previous_class:
                    same_count += 1
                else:
                    previous_class = valid_prediction
                    same_count = 1

                if same_count >= AI_CONFIRM_FRAMES:
                    confirmed = valid_prediction
                else:
                    confirmed = None
            else:
                previous_class = None
                same_count = 0
                confirmed = None

            with ai_lock:
                ai_current_class = predicted
                ai_current_confidence = confidence
                ai_last_crop = crop_img
                ai_last_probabilities = probabilities

                if confirmed is not None:
                    ai_confirmed_class = confirmed

        except Exception as exc:
            print("[AI] Ошибка распознавания:", exc)

        time.sleep(AI_INTERVAL)


# ============================================================
# ROS CAMERA
# ============================================================
def camera_callback(msg):
    global latest_frame, latest_frame_time, latest_frame_for_web
    try:
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        latest_frame = frame
        latest_frame_time = time.time()
        
        with frame_lock:
            latest_frame_for_web = frame
            
    except Exception as exc:
        print("Ошибка камеры:", exc)


# ============================================================
# HSV ДЕТЕКТОР
# ============================================================
def detect_cube(img):
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        if area > best_area:
            best_area = area
            best = contour

    if best is None:
        return None

    x, y, w, h = cv2.boundingRect(best)
    return {
        "cx": x + w / 2,
        "cy": y + h / 2,
        "area": best_area,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "contour": best,
    }


def crop(img, contour):
    """
    Извлекает только цифру с карточки.

    Важно: HSV-контур соответствует всей цветной карточке, поэтому простой
    threshold захватывает рамку, верхнюю полоску и подпись. Здесь они
    отбрасываются через внутренний отступ и анализ connected components.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w].copy()

    if roi.size == 0:
        return np.zeros(MODEL_INPUT_SIZE, dtype=np.uint8), (x, y, w, h)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Отрезаем края карточки, где обычно находятся рамка и фон телефона.
    margin_x = max(2, int(w * 0.10))
    margin_top = max(2, int(h * 0.12))
    margin_bottom = max(2, int(h * 0.22))

    x1 = min(margin_x, max(0, w - 1))
    x2 = max(x1 + 1, w - margin_x)
    y1 = min(margin_top, max(0, h - 1))
    y2 = max(y1 + 1, h - margin_bottom)

    inner = gray[y1:y2, x1:x2]

    if inner.size == 0:
        inner = gray

    inner = cv2.GaussianBlur(inner, (3, 3), 0)

    # Тёмная цифра становится белой, фон — чёрным.
    _, binary = cv2.threshold(
        inner,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Убираем световые блики и мелкий шум без разрыва тонких штрихов.
    binary = cv2.medianBlur(binary, 3)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    inner_h, inner_w = binary.shape
    image_area = inner_h * inner_w
    center_x = inner_w / 2.0
    center_y = inner_h / 2.0

    candidates = []

    for label in range(1, count):
        cx, cy, cw, ch, area = stats[label]

        if area < max(12, int(image_area * 0.002)):
            continue

        # Отбрасываем рамку и компоненты, касающиеся краёв.
        touches_edge = (
            cx <= 1
            or cy <= 1
            or cx + cw >= inner_w - 1
            or cy + ch >= inner_h - 1
        )
        if touches_edge:
            continue

        # Подпись словами обычно низкая и расположена внизу.
        if ch < inner_h * 0.20:
            continue

        comp_cx, comp_cy = centroids[label]
        distance = (
            ((comp_cx - center_x) / max(inner_w, 1)) ** 2
            + ((comp_cy - center_y) / max(inner_h, 1)) ** 2
        )

        # Предпочитаем крупную компоненту около центра.
        score = float(area) * (1.0 - min(distance, 0.8))
        candidates.append((score, label))

    if candidates:
        _, best_label = max(candidates, key=lambda item: item[0])
        digit_mask = np.where(labels == best_label, 255, 0).astype(np.uint8)
    else:
        # Запасной вариант: берём все достаточно крупные центральные пиксели.
        digit_mask = binary.copy()

        border = max(1, int(min(inner_w, inner_h) * 0.04))
        digit_mask[:border, :] = 0
        digit_mask[-border:, :] = 0
        digit_mask[:, :border] = 0
        digit_mask[:, -border:] = 0

    points = cv2.findNonZero(digit_mask)

    if points is None:
        fallback = cv2.resize(
            cv2.bitwise_not(inner),
            MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_AREA,
        )
        return fallback, (x, y, w, h)

    bx, by, bw, bh = cv2.boundingRect(points)
    digit = digit_mask[by:by + bh, bx:bx + bw]

    canvas_w, canvas_h = MODEL_INPUT_SIZE

    # MNIST-подобным моделям обычно подходит цифра размером около 46–50 px.
    target = 48
    scale = min(
        target / max(bw, 1),
        target / max(bh, 1),
    )

    new_w = max(1, int(round(bw * scale)))
    new_h = max(1, int(round(bh * scale)))

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=interpolation,
    )

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    offset_x = (canvas_w - new_w) // 2
    offset_y = (canvas_h - new_h) // 2

    canvas[
        offset_y:offset_y + new_h,
        offset_x:offset_x + new_w,
    ] = digit

    # Очень лёгкое сглаживание, без morphology OPEN, которое портило 1 и 7.
    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)

    return canvas, (x, y, w, h)


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def normalize_course(course):
    while course >= 360: course -= 360
    while course < 0: course += 360
    return course

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ПОИСКА
# ============================================================
def scan_for_cube(drone, duration, status_msg="Сканирование"):
    """
    Универсальная функция сканирования.
    Проверяет наличие кубика в течение заданного времени.
    Возвращает detection, если кубик найден, иначе None.
    """
    global system_status
    system_status = status_msg
    
    start = time.time()
    detected_count = 0
    
    while time.time() - start < duration:
        if rospy.is_shutdown():
            return None
            
        detection = detect_cube(latest_frame)
        if detection is not None:
            detected_count += 1
            if detected_count >= LOCK_FRAMES:
                return detection
        else:
            detected_count = 0
        time.sleep(0.08)
    
    return None

def smart_reacquire(drone, last_known_course):
    """
    Умный повторный поиск.
    Сканирует окрестности последней известной позиции цели.
    """
    global system_status
    system_status = "Повторный поиск"
    
    if not REACQUIRE_ENABLED:
        return None
    
    print("[VISION] цель потеряна, выполняю интеллектуальное сканирование")
    drone.set_speed(0)
    
    # Сканируем углы вокруг последней известной позиции
    scan_offsets = [-30, -20, -10, 0, 10, 20, 30]
    
    for offset in scan_offsets:
        if rospy.is_shutdown():
            return None
            
        course = normalize_course(last_known_course + offset)
        drone.change_course(course)
        time.sleep(REACQUIRE_SCAN_TIME)
        
        detection = scan_for_cube(drone, 0.3, f"Сканирование {course}°")
        if detection is not None:
            print(f"[VISION] цель найдена снова на курсе {course}°")
            return detection
    
    print("[VISION] повторно найти цель не удалось")
    return None


# ============================================================
# ФАЗА 1: ЦЕНТРОВКА КУБИКА
# ============================================================
def center_cube(drone):
    """
    Фаза 1: Центровка кубика.
    Дрон стоит на месте и центрирует кубик с помощью P-регулятора,
    пока он не окажется устойчиво в центре кадра 
    (CENTER_CONFIRM_FRAMES кадров подряд).
    """
    global system_status
    system_status = "Центровка кубика"
    
    print("\n================================\n ЦЕНТРОВКА КУБИКА\n================================")
    
    lost_since = None
    last_known_course = drone.get_course()
    current_pitch = SEARCH_PITCH
    center_confirm_count = 0
    
    while True:
        if rospy.is_shutdown():
            return False
        
        detection = detect_cube(latest_frame)
        
        if detection is None:
            if lost_since is None:
                lost_since = time.time()
            drone.set_speed(0)
            
            if time.time() - lost_since > REACQUIRE_TIMEOUT:
                result = smart_reacquire(drone, last_known_course)
                if result is None:
                    system_status = "Цель потеряна при центровке"
                    return False
                lost_since = None
                center_confirm_count = 0
            time.sleep(0.05)
            continue
        
        lost_since = None
        cx, cy = detection["cx"], detection["cy"]
        
        error_x = cx - IMG_CX
        error_y = cy - IMG_CY
        
        last_known_course = drone.get_course()
        
        # P-регулятор для YAW
        current_course = drone.get_course()
        yaw_centered = True
        if abs(error_x) > CENTER_THRESHOLD_X:
            norm_error_x = error_x / IMG_CX
            yaw_correction = clamp(norm_error_x * YAW_KP, -MAX_YAW_CORRECTION, MAX_YAW_CORRECTION)
            target_course = normalize_course(current_course + yaw_correction)
            drone.change_course(target_course)
            yaw_centered = False
        
        # P-регулятор для PITCH
        try:
            current_pitch = drone.get_pitch()
        except AttributeError:
            pass
        
        pitch_centered = True
        if abs(error_y) > CENTER_THRESHOLD_Y:
            norm_error_y = -error_y / IMG_CY
            pitch_correction = clamp(norm_error_y * PITCH_KP, -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION)
            target_pitch = clamp(current_pitch + pitch_correction, MIN_PITCH, MAX_PITCH)
            drone.change_pitch(target_pitch)
            current_pitch = target_pitch
            pitch_centered = False
        
        drone.set_speed(0)  # Стоим на месте во время центровки
        
        # Считаем кадры, когда кубик устойчиво в центре
        if yaw_centered and pitch_centered:
            center_confirm_count += 1
        else:
            center_confirm_count = 0
        
        # Проверка: кубик устойчиво в центре
        if center_confirm_count >= CENTER_CONFIRM_FRAMES:
            print(f"[CENTER] Кубик отцентрован ({CENTER_CONFIRM_FRAMES} кадров подряд)")
            return True
        
        time.sleep(0.08)


# ============================================================
# ФАЗА 2: ПРИБЛИЖЕНИЕ С КЛАССИФИКАЦИЕЙ
# ============================================================
def approach_with_classification(drone):
    """
    Фаза 2: Приближение к кубику с классификацией.
    Дрон движется вперёд, продолжая центрировать кубик.
    Приближение завершается, когда классификатор 
    CLASSIFY_CONFIRM_FRAMES кадров подряд возвращает один и тот же класс.
    
    Returns:
        int or None: распознанный класс, или None при потере цели
    """
    global system_status, detected_class
    system_status = "Приближение с классификацией"
    
    print("\n================================\n ПРИБЛИЖЕНИЕ С КЛАССИФИКАЦИЕЙ\n================================")
    
    lost_since = None
    last_known_course = drone.get_course()
    current_pitch = SEARCH_PITCH
    
    last_class = None
    consecutive_count = 0
    final_class = None
    
    while True:
        if rospy.is_shutdown():
            return None
        
        detection = detect_cube(latest_frame)
        
        if detection is None:
            if lost_since is None:
                lost_since = time.time()
            drone.set_speed(0)
            
            if time.time() - lost_since > REACQUIRE_TIMEOUT:
                result = smart_reacquire(drone, last_known_course)
                if result is None:
                    system_status = "Цель потеряна при приближении"
                    return None
                lost_since = None
                # Сбрасываем счётчик классификации после повторного поиска
                last_class = None
                consecutive_count = 0
            time.sleep(0.05)
            continue
        
        lost_since = None
        cx, cy = detection["cx"], detection["cy"]
        w, h = detection["w"], detection["h"]
        size = max(w, h)
        
        last_known_course = drone.get_course()
        
        error_x = cx - IMG_CX
        error_y = cy - IMG_CY
        
        print(f"[TARGET] cx={round(cx)} cy={round(cy)} size={w}x{h} | class_count={consecutive_count}/{CLASSIFY_CONFIRM_FRAMES}")
        
        # P-регулятор для YAW (продолжаем центрировать во время приближения)
        current_course = drone.get_course()
        if abs(error_x) > CENTER_THRESHOLD_X:
            norm_error_x = error_x / IMG_CX
            yaw_correction = clamp(norm_error_x * YAW_KP, -MAX_YAW_CORRECTION, MAX_YAW_CORRECTION)
            target_course = normalize_course(current_course + yaw_correction)
            drone.change_course(target_course)
        
        # P-регулятор для PITCH
        try:
            current_pitch = drone.get_pitch()
        except AttributeError:
            pass
        
        if abs(error_y) > CENTER_THRESHOLD_Y:
            norm_error_y = -error_y / IMG_CY
            pitch_correction = clamp(norm_error_y * PITCH_KP, -MAX_PITCH_CORRECTION, MAX_PITCH_CORRECTION)
            target_pitch = clamp(current_pitch + pitch_correction, MIN_PITCH, MAX_PITCH)
            drone.change_pitch(target_pitch)
            current_pitch = target_pitch
        
        # Классификация, если размер кубика достаточен
        if size >= MIN_SIZE_FOR_CLASSIFY:
            predicted_class = classify_frame(latest_frame, detection)
            
            if predicted_class == last_class:
                consecutive_count += 1
            else:
                last_class = predicted_class
                consecutive_count = 1
            
            # Проверка: n кадров подряд один класс
            if consecutive_count >= CLASSIFY_CONFIRM_FRAMES:
                final_class = predicted_class
                drone.set_speed(0)
                detected_class = final_class
                system_status = f"Класс {final_class} распознан"
                print(f"\n################################\n#  КЛАСС {final_class} РАСПОЗНАН  #\n################################")
                # Включаем лазер и оставляем его включённым
                turn_on_laser(drone)
                return final_class
        
        # Управление скоростью
        speed = APPROACH_SPEED
        
        # Если кубик сильно смещён, останавливаемся для центровки
        if abs(error_x) > (CENTER_THRESHOLD_X * 2) or abs(error_y) > (CENTER_THRESHOLD_Y * 2):
            speed = 0
        
        drone.set_speed(speed)
        time.sleep(0.08)


# ============================================================
# ЛОГИКА ПОИСКА ЗМЕЙКОЙ
# ============================================================
def search_pool(drone):
    """
    Поиск кубика методом змейки.
    После обнаружения кубика выполняет:
    1. Центровку (удержание в центре)
    2. Приближение с классификацией
    """
    global system_status
    system_status = "Начало поиска"
    
    print("\n================================\n ПОИСК КУБИКА (ЗМЕЙКА)\n================================")
    drone.set_speed(0)
    drone.set_depth(SEARCH_DEPTH)
    drone.change_pitch(SEARCH_PITCH)
    time.sleep(2)
    
    for lane in range(LANES):
        if rospy.is_shutdown():
            return False
            
        print(f"\n========== ПОЛОСА {lane + 1} ИЗ {LANES} ==========")
        
        # Чередование направления: 0° и 180°
        heading = 0 if lane % 2 == 0 else 180
        
        # Устанавливаем курс и начинаем движение
        drone.change_course(heading)
        time.sleep(1.0)
        drone.set_speed(SEARCH_SPEED)
        
        # Сканируем текущую полосу
        detection = scan_for_cube(drone, LONG_TIME, f"Поиск (полоса {heading}°)")
        
        if detection is not None:
            drone.set_speed(0)
            print("\n================================\n ЦЕЛЬ ОБНАРУЖЕНА\n================================")
            
            # ФАЗА 1: Центровка
            if center_cube(drone):
                # ФАЗА 2: Приближение с классификацией
                final_class = approach_with_classification(drone)
                if final_class is not None:
                    print(f"\n[MISSION] Миссия завершена. Распознан класс: {final_class}")
                    return True
            
            return False
        
        drone.set_speed(0)
        
        # Переход на следующую полосу (если не последняя)
        if lane < LANES - 1:
            print("[SEARCH] переход на соседнюю полосу")
            drone.change_course(90)
            time.sleep(1.0)
            drone.set_speed(SEARCH_SPEED)
            
            # Во время перехода тоже ищем кубик
            detection = scan_for_cube(drone, SHIFT_TIME, "Переход на полосу")
            
            if detection is not None:
                drone.set_speed(0)
                print("\n================================\n ЦЕЛЬ ОБНАРУЖЕНА ПРИ ПЕРЕХОДЕ\n================================")
                
                # ФАЗА 1: Центровка
                if center_cube(drone):
                    # ФАЗА 2: Приближение с классификацией
                    final_class = approach_with_classification(drone)
                    if final_class is not None:
                        print(f"\n[MISSION] Миссия завершена. Распознан класс: {final_class}")
                        return True
                
                return False
            
            drone.set_speed(0)
    
    drone.set_speed(0)
    system_status = "Кубик не найден"
    print("\nКубик не найден.")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Запуск поиска кубика...")
    print("📺 Обработанный видеопоток: http://localhost:5000/video")
    print("   (или http://<IP_ДРОНА>:5000/video с другого устройства)")
    print("🧠 Вход модели 64x64: http://localhost:5000/ai_crop")

    # Загружаем веса один раз до начала миссии.
    print(f"[AI] Загрузка весов: {WEIGHTS_DIR}")
    digit_model = FastDigitModel(WEIGHTS_DIR)
    print("[AI] Модель успешно загружена")

    drone = DroneLibrary()

    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    ai_thread = threading.Thread(target=ai_recognition_worker, daemon=True)
    ai_thread.start()

    rospy.Subscriber("/raspicam_node/image", Image, camera_callback, queue_size=1)

    drone.start()

    try:
        search_pool(drone)
    except KeyboardInterrupt:
        print("\nОстановка пользователем")
    except Exception as exc:
        print(f"ОШИБКА: {type(exc).__name__} {exc}")
    finally:
        ai_running = False
        print("\nОстановка дрона...")
        drone.set_speed(0)
        # Лазер НЕ выключаем - он должен гореть после успешного распознавания
        try:
            drone.change_pitch(0)
        except Exception:
            pass
        
        # Выводим результат классификации, если он есть
        if detected_class is not None:
            print(f"\n[РЕЗУЛЬТАТ] Распознанный класс: {detected_class}")
            print("[РЕЗУЛЬТАТ] Лазер включён и будет гореть до выключения дрона")
        else:
            print("\n[РЕЗУЛЬТАТ] Класс не был распознан")
        
        time.sleep(0.5)
        drone.stop()
        drone.set_offline_mode()
        
        print("Программа завершена")
        rospy.signal_shutdown("Mission complete")
        time.sleep(1.0)
        cv2.destroyAllWindows()
