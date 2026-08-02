import time
import math
import threading
from collections import deque

import cv2
import numpy as np

from pioneer_sdk2 import Pioneer
from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera
пп

# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"

TARGET_MARKER_ID = 5

# Точка после взлёта
ENTRY_POINT = (0.5, 1.5, 1.7, 0.0)

# Точка входа в область поиска
SEARCH_ENTRY_POINT = (0.0, 1.0, 1.7, 0.0)

# Границы змейки
X_START = 1.5
X_END = -1.5
X_STEP = -0.4

Y_MIN = 0.5
Y_MAX = 4.0

SEARCH_HEIGHT = 1.8
RETURN_HEIGHT = 1.7
YAW = 0.0

# Стартовая точка
HOME_X = 0.0
HOME_Y = 0.0

# Варианты посадочных точек.
# Координаты заданы в САНТИМЕТРАХ от стартовой точки (0, 0).
# Перед полётом они переводятся в метры: 450 см -> 4.50 м.
LANDING_POINT_NUMBER = 1

LANDING_POINTS_CM = {
    1: (450, 70),
    2: (345, 325),
    3: (440, 710),
    4: (-415, 700),
    5: (-350, 435),
    6: (-455, 120),
    7: (-350, 435),
    8: (345, 325),
    9: (440, 710),
}

LANDING_APPROACH_HEIGHT = RETURN_HEIGHT

# Yaw при подходе к посадочной точке.
# Подстрой под направление на бассейн.
LANDING_YAW = 0.0

# ============================================================
# ПРИЁМ ЛАЗЕРНОГО ПАКЕТА ПО КАМЕРЕ
# Формат НЕ меняем: 1010 + 4 бита числа + CRC4.
# CRC принимается, но намеренно НЕ проверяется.
# Алгоритм детекции и декодирования взят из присланного приёмника:
# красная точка ищется в центре ArUco 5, длина импульса измеряется
# по количеству кадров.
# ============================================================
LASER_RX_PREAMBLE = (1, 0, 1, 0)
LASER_RX_PREAMBLE_LEN = 4
LASER_RX_MESSAGE_LEN = 4
LASER_RX_CRC_LEN = 4

LASER_RX_DOT_DURATION = 0.3
LASER_RX_DASH_DURATION = 0.9

LASER_RX_VIDEO_FPS = 20
EXPECTED_DOT_FRAMES = max(1, round(LASER_RX_DOT_DURATION * LASER_RX_VIDEO_FPS))
EXPECTED_DASH_FRAMES = max(1, round(LASER_RX_DASH_DURATION * LASER_RX_VIDEO_FPS))
LASER_RX_FRAME_SPLIT = (
    EXPECTED_DOT_FRAMES + EXPECTED_DASH_FRAMES
) / 2.0

LASER_RX_MIN_PULSE_FRAMES = max(
    2,
    round(EXPECTED_DOT_FRAMES * 0.45)
)
LASER_RX_MAX_PULSE_FRAMES = round(
    EXPECTED_DASH_FRAMES * 1.8
)
LASER_RX_STATE_CONFIRM_FRAMES = 2

# Область поиска лазера около центра ArUco.
LASER_ROI_SCALE = 0.34
LASER_MARKER_MEMORY_FRAMES = 10

# Порог красной точки.
LASER_MIN_RED = 150
LASER_RED_DOMINANCE = 55
LASER_MIN_RED_PIXELS = 3
LASER_MIN_SATURATION = 120
LASER_MIN_VALUE = 140

# ============================================================
# ВИЗУАЛЬНАЯ ЦЕНТРОВКА НАД ПОСАДОЧНОЙ ЗОНОЙ
# ============================================================
#
# Посадочная зона: оранжевый круг с белой буквой "Н".
# Основной детектор использует оранжевый цвет круга.
#
# HSV-диапазон оранжевого. При необходимости подстрой по реальной камере.
LANDING_ORANGE_HSV_LOW = (5, 90, 90)
LANDING_ORANGE_HSV_HIGH = (30, 255, 255)

# Минимальная площадь оранжевого контура на кадре, px^2.
LANDING_MIN_CONTOUR_AREA = 800

# Погрешность центровки больше, чем для ArUco:
# идеально попадать в центр посадочного круга не требуется.
LANDING_CENTER_TOLERANCE_PX = 120

# P-регулятор посадочной центровки.
LANDING_CENTER_KP = 0.0008
LANDING_CENTER_MIN_SPEED = 0.035
LANDING_CENTER_MAX_SPEED = 0.11

# Сглаживание центра посадочного круга.
LANDING_CENTER_FILTER_ALPHA = 0.25

# Нужно удержаться в допустимой зоне перед зависанием.
LANDING_CENTER_STABLE_TIME = 1.0

# Максимальное время визуальной центровки.
LANDING_CENTER_TIMEOUT = 12.0

# Время зависания над посадочной зоной перед посадкой.
LANDING_HOVER_TIME = 3.0

HOVER_TIME = 5.0

# Частота основных циклов
LOOP_DELAY = 0.05

# Желаемая средняя скорость перелёта между точками, м/с.
# Для go_to_local_point() скорость задаётся косвенно:
# time = расстояние / FLIGHT_SPEED.
FLIGHT_SPEED = 0.5

# Минимальное время, передаваемое в go_to_local_point(), секунд.
MIN_FLIGHT_TIME = 1

# Камера
CAMERA_ANGLE = -80
VIDEO_FPS = 20

# ============================================================
# УСТОЙЧИВОЕ РАСПОЗНАВАНИЕ ARUCO ЧЕРЕЗ СЕТКУ
# ============================================================

# Сколько секунд хранить последнее корректное положение метки,
# если сетка временно помешала распознаванию.
ARUCO_DETECTION_MEMORY = 0.45

# Масштаб дополнительного прохода распознавания.
ARUCO_UPSCALE = 1.6

# Подавление длинных линий сетки используется только как один
# из вариантов изображения. Исходный кадр всегда проверяется первым.
ARUCO_ENABLE_NET_SUPPRESSION = True

# ============================================================
# ЦЕНТРИРОВАНИЕ ПО ARUCO
# ============================================================

# Допустимая погрешность центрирования, пикселей
CENTER_TOLERANCE_PX = 100

# P-регулятор: скорость = ошибка_в_пикселях * CENTER_KP
CENTER_KP = 0.0010

# Ограничения скорости коррекции, м/с
CENTER_MIN_SPEED = 0.04
CENTER_MAX_SPEED = 0.16

# Сглаживание координат ArUco:
# 0.0 = сильное сглаживание, 1.0 = без сглаживания
CENTER_FILTER_ALPHA = 0.30

# Время удержания маркера в центре
CENTER_STABLE_TIME = 1

# Допустимое время временной потери маркера.
# Пока таймаут не истёк, дрон продолжает осторожно двигаться
# в направлении последней команды коррекции, чтобы снова увидеть ArUco.
MARKER_LOST_TIMEOUT = 3.0

# Скорость при временной потере ArUco уменьшаем относительно
# последней команды центрирования.
LOST_MARKER_SPEED_FACTOR = 0.55
LOST_MARKER_MAX_SPEED = 0.08

# Максимальное время центрирования
CENTER_TIMEOUT = 40.0


# ============================================================
# ОБЩЕЕ СОСТОЯНИЕ
# ============================================================

running = True

marker_lock = threading.Lock()

target_marker_visible = False
target_marker_center = None
target_frame_size = None

# Память последнего достоверного распознавания ArUco.
last_target_detection_time = -10_000.0
last_target_center = None
last_target_corners = None
last_target_detection_mode = "NONE"

# Состояние визуального детектора посадочной зоны.
landing_zone_visible = False
landing_zone_center = None

camera = None
viewer = None

laser_rx_lock = threading.Lock()
laser_rx_digit = None
laser_rx_running = False

laser_last_marker_roi = None
laser_last_marker_seen_frame = -10_000
laser_frame_number = 0


# ============================================================
# ЛАЗЕРНЫЙ ПРИЁМНИК ПО КАМЕРЕ
# ============================================================

def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def get_received_laser_digit():
    with laser_rx_lock:
        return laser_rx_digit


def clear_received_laser_digit():
    global laser_rx_digit
    with laser_rx_lock:
        laser_rx_digit = None


class FrameLaserDecoder:
    """
    Декодирует лазерные импульсы по числу кадров.

    Формат пакета сохраняется:
        1010 + 4 бита числа + 4 бита CRC.

    CRC-биты считываются, но результат CRC намеренно не проверяется.
    Как только после преамбулы получены 4 бита сообщения со значением
    0..8, число сразу публикуется и центрирование прекращается.
    """

    def __init__(self):
        self.bit_history = deque(maxlen=100)
        self.reset_all()

    def reset_all(self):
        self.raw_state = False
        self.stable_state = False
        self.same_raw_frames = 0
        self.on_frames = 0
        self.on_start_time = None

        self.synced = False
        self.payload_bits = []

        self.last_bit = None
        self.last_pulse_frames = 0
        self.last_pulse_seconds = 0.0
        self.status = "WAIT ARUCO 5"

        self.bit_history.clear()

    def reset_sync(self):
        self.synced = False
        self.payload_bits = []

    def _find_preamble(self):
        if len(self.bit_history) < LASER_RX_PREAMBLE_LEN:
            return False

        tail = list(self.bit_history)[-LASER_RX_PREAMBLE_LEN:]

        if tuple(tail) == LASER_RX_PREAMBLE:
            self.synced = True
            self.payload_bits = []
            self.status = "PREAMBLE OK"
            print("[LASER RX] Преамбула 1010 найдена")
            return True

        return False

    def _accept_message_early(self):
        """
        Принимаем число сразу после первых 4 бит сообщения.
        CRC для принятия решения не нужен.
        """
        global laser_rx_digit, laser_rx_running

        if len(self.payload_bits) < LASER_RX_MESSAGE_LEN:
            return False

        message_bits = self.payload_bits[:LASER_RX_MESSAGE_LEN]
        value = bits_to_int(message_bits)

        if 0 <= value <= 8:
            with laser_rx_lock:
                if laser_rx_digit is None:
                    laser_rx_digit = value

            self.status = f"RECEIVED {value}"
            laser_rx_running = False

            print(
                f"[LASER RX] Получено число {value}; "
                "CRC не проверяется"
            )
            return True

        print(
            f"[LASER RX] Число {value} вне диапазона 0..8, "
            "ожидаю новый пакет"
        )
        self.reset_sync()
        return False

    def add_bit(self, bit, pulse_frames, pulse_seconds):
        self.last_bit = bit
        self.last_pulse_frames = pulse_frames
        self.last_pulse_seconds = pulse_seconds
        self.bit_history.append(bit)

        print(
            f"[LASER RX] BIT={bit} "
            f"frames={pulse_frames} "
            f"time={pulse_seconds:.3f}s"
        )

        if not self.synced:
            self.status = "SEARCH PREAMBLE"
            self._find_preamble()
            return

        self.payload_bits.append(bit)
        self.status = (
            f"DATA {len(self.payload_bits)}/"
            f"{LASER_RX_MESSAGE_LEN + LASER_RX_CRC_LEN}"
        )

        # Число принимается сразу после 4 бит сообщения.
        if len(self.payload_bits) == LASER_RX_MESSAGE_LEN:
            self._accept_message_early()

        # Если значение было недопустимым, оставшиеся CRC-биты
        # не имеют значения — снова ищем преамбулу.

    def finish_pulse(self):
        frames = self.on_frames
        seconds = 0.0

        if self.on_start_time is not None:
            seconds = time.monotonic() - self.on_start_time

        self.on_frames = 0
        self.on_start_time = None

        if frames < LASER_RX_MIN_PULSE_FRAMES:
            return

        if frames > LASER_RX_MAX_PULSE_FRAMES:
            self.status = "BAD LONG PULSE"
            self.reset_sync()
            print(
                f"[LASER RX] Слишком длинный импульс: "
                f"{frames} кадров, {seconds:.3f} с"
            )
            return

        bit = 1 if frames >= LASER_RX_FRAME_SPLIT else 0
        self.add_bit(bit, frames, seconds)

    def update(self, laser_visible):
        if not laser_rx_running:
            return

        if self.stable_state:
            self.on_frames += 1

        if laser_visible == self.raw_state:
            self.same_raw_frames += 1
        else:
            self.raw_state = laser_visible
            self.same_raw_frames = 1

        if (
            self.raw_state != self.stable_state
            and self.same_raw_frames >= LASER_RX_STATE_CONFIRM_FRAMES
        ):
            if self.raw_state:
                self.stable_state = True
                self.on_frames = LASER_RX_STATE_CONFIRM_FRAMES
                self.on_start_time = time.monotonic()
                self.status = "LASER ON"
            else:
                self.stable_state = False
                self.finish_pulse()


laser_decoder = FrameLaserDecoder()


def find_laser_roi(marker_corners, frame_shape, frame_number):
    """
    Возвращает небольшую область вокруг центра ArUco 5.
    При кратком исчезновении метки использует последнюю ROI.
    """
    global laser_last_marker_roi, laser_last_marker_seen_frame

    found_roi = None

    if marker_corners is not None:
        pts = marker_corners.reshape(4, 2).astype(np.float32)

        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))

        side_lengths = [
            np.linalg.norm(pts[(index + 1) % 4] - pts[index])
            for index in range(4)
        ]
        marker_size = float(np.mean(side_lengths))

        roi_size = max(8, int(marker_size * LASER_ROI_SCALE))
        half = roi_size // 2

        frame_h, frame_w = frame_shape[:2]

        x1 = max(0, int(center_x) - half)
        y1 = max(0, int(center_y) - half)
        x2 = min(frame_w, int(center_x) + half + 1)
        y2 = min(frame_h, int(center_y) + half + 1)

        if x2 > x1 and y2 > y1:
            found_roi = (x1, y1, x2, y2)
            laser_last_marker_roi = found_roi
            laser_last_marker_seen_frame = frame_number

    if (
        found_roi is None
        and laser_last_marker_roi is not None
        and frame_number - laser_last_marker_seen_frame
        <= LASER_MARKER_MEMORY_FRAMES
    ):
        found_roi = laser_last_marker_roi

    return found_roi


def detect_red_laser(frame, roi_coords):
    """Ищет красную лазерную точку внутри ROI центра ArUco."""
    if roi_coords is None:
        return False, 0, 0

    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 0, 0

    blue, green, red = cv2.split(roi)

    red_mask_bgr = (
        (red >= LASER_MIN_RED)
        & (
            red.astype(np.int16) - green.astype(np.int16)
            >= LASER_RED_DOMINANCE
        )
        & (
            red.astype(np.int16) - blue.astype(np.int16)
            >= LASER_RED_DOMINANCE
        )
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_low = cv2.inRange(
        hsv,
        np.array(
            [0, LASER_MIN_SATURATION, LASER_MIN_VALUE],
            dtype=np.uint8
        ),
        np.array([12, 255, 255], dtype=np.uint8)
    )

    mask_high = cv2.inRange(
        hsv,
        np.array(
            [168, LASER_MIN_SATURATION, LASER_MIN_VALUE],
            dtype=np.uint8
        ),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    red_mask_hsv = (mask_low > 0) | (mask_high > 0)
    combined = red_mask_bgr & red_mask_hsv

    red_pixels = int(np.count_nonzero(combined))
    peak_red = int(np.max(red)) if red.size else 0

    return (
        red_pixels >= LASER_MIN_RED_PIXELS,
        red_pixels,
        peak_red
    )


def start_laser_receiver():
    global laser_rx_running

    clear_received_laser_digit()
    laser_decoder.reset_all()
    laser_rx_running = True

    print(
        "[LASER RX] Камерный приём запущен: "
        "красная точка в центре ArUco 5"
    )


def stop_laser_receiver():
    global laser_rx_running
    laser_rx_running = False
    print("[LASER RX] Камерный приём остановлен")


def landing_number_from_signal(value):
    """0 означает посадочную зону 9; 1..8 остаются без изменения."""
    if value == 0:
        return 9
    if 1 <= value <= 8:
        return value
    return 1


# ============================================================
# ARUCO: УСТОЙЧИВОЕ РАСПОЗНАВАНИЕ ЧЕРЕЗ СЕТКУ
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

# Более широкий набор окон адаптивной бинаризации.
aruco_parameters.adaptiveThreshWinSizeMin = 3
aruco_parameters.adaptiveThreshWinSizeMax = 61
aruco_parameters.adaptiveThreshWinSizeStep = 4
aruco_parameters.adaptiveThreshConstant = 7

# Не отбрасываем метку из-за повреждённого сеткой контура.
aruco_parameters.minMarkerPerimeterRate = 0.012
aruco_parameters.maxMarkerPerimeterRate = 4.0
aruco_parameters.polygonalApproxAccuracyRate = 0.07
aruco_parameters.minCornerDistanceRate = 0.025
aruco_parameters.minDistanceToBorder = 2

# Уточнение найденных углов.
aruco_parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)
aruco_parameters.cornerRefinementWinSize = 7
aruco_parameters.cornerRefinementMaxIterations = 50
aruco_parameters.cornerRefinementMinAccuracy = 0.01

# Более подробное перспективное изображение клеток.
aruco_parameters.perspectiveRemovePixelPerCell = 10
aruco_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15

# Разрешаем некоторое повреждение границы и клеток сеткой.
aruco_parameters.maxErroneousBitsInBorderRate = 0.50
aruco_parameters.errorCorrectionRate = 0.85
aruco_parameters.detectInvertedMarker = True

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


def suppress_net_lines(gray):
    """
    Строит вариант серого изображения с ослабленными длинными
    горизонтальными и вертикальными линиями сетки.

    Важно: результат используется только как дополнительный проход.
    Обычное изображение проверяется детектором раньше.
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    frame_h, frame_w = gray.shape[:2]

    horizontal_length = max(17, frame_w // 32)
    vertical_length = max(17, frame_h // 24)

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length)
    )

    horizontal_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel
    )
    vertical_lines = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel
    )

    net_mask = cv2.bitwise_or(
        horizontal_lines,
        vertical_lines
    )

    net_mask = cv2.dilate(
        net_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1
    )

    restored = cv2.inpaint(
        enhanced,
        net_mask,
        3,
        cv2.INPAINT_TELEA
    )

    return restored, net_mask


def _target_result_from_detection(corners, ids, scale):
    """
    Из результата detectMarkers() выбирает только TARGET_MARKER_ID
    и переводит координаты обратно в размер исходного кадра.
    """
    if ids is None or len(ids) == 0:
        return None

    flat_ids = ids.flatten().astype(int)
    indexes = np.where(flat_ids == TARGET_MARKER_ID)[0]

    if len(indexes) == 0:
        return None

    target_index = int(indexes[0])
    target_corners = corners[target_index].astype(np.float32)

    if scale != 1.0:
        target_corners = target_corners / scale

    return (
        [target_corners],
        np.array([[TARGET_MARKER_ID]], dtype=np.int32)
    )


def detect_target_aruco_robust(frame):
    """
    Ищет целевую ArUco по нескольким вариантам изображения.

    Порядок:
      1. обычный серый кадр;
      2. CLAHE;
      3. лёгкое размытие CLAHE;
      4. увеличенный CLAHE;
      5. подавление линий сетки;
      6. увеличенный вариант после подавления сетки.

    Возвращает:
        corners, ids, detection_mode, net_mask
    """
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(gray)

    variants = [
        ("GRAY", gray, 1.0),
        ("CLAHE", enhanced, 1.0),
        (
            "CLAHE_BLUR",
            cv2.GaussianBlur(enhanced, (3, 3), 0),
            1.0
        ),
    ]

    upscale = float(ARUCO_UPSCALE)

    if upscale > 1.0:
        variants.append(
            (
                "CLAHE_SCALE",
                cv2.resize(
                    enhanced,
                    None,
                    fx=upscale,
                    fy=upscale,
                    interpolation=cv2.INTER_CUBIC
                ),
                upscale
            )
        )

    net_mask = None

    if ARUCO_ENABLE_NET_SUPPRESSION:
        restored, net_mask = suppress_net_lines(gray)

        variants.append(
            ("NET_SUPPRESSED", restored, 1.0)
        )

        variants.append(
            (
                "NET_BLUR",
                cv2.GaussianBlur(restored, (3, 3), 0),
                1.0
            )
        )

        if upscale > 1.0:
            variants.append(
                (
                    "NET_SCALE",
                    cv2.resize(
                        restored,
                        None,
                        fx=upscale,
                        fy=upscale,
                        interpolation=cv2.INTER_CUBIC
                    ),
                    upscale
                )
            )

    for mode, image, scale in variants:
        corners, ids, _ = aruco_detector.detectMarkers(image)

        target_result = _target_result_from_detection(
            corners,
            ids,
            scale
        )

        if target_result is not None:
            target_corners, target_ids = target_result
            return (
                target_corners,
                target_ids,
                mode,
                net_mask
            )

    return [], None, "NONE", net_mask


# ============================================================
# PIONEER
# ============================================================

pioneer = Pioneer()


# ============================================================
# КАМЕРА
# ============================================================

servo_camera = ServoCamera()

if servo_camera.set_angle(CAMERA_ANGLE):
    print(f"[CAMERA] Угол камеры: {CAMERA_ANGLE} градусов")
else:
    print("[CAMERA] Не удалось установить угол камеры")


# ============================================================
# ВИДЕО + ARUCO
# ============================================================

def video_worker():
    """
    Получает изображение с камеры Pioneer,
    обнаруживает ArUco и транслирует изображение
    в браузер через ImageViewer.
    """

    global running
    global camera
    global viewer

    global target_marker_visible
    global target_marker_center
    global target_frame_size
    global last_target_detection_time
    global last_target_center
    global last_target_corners
    global last_target_detection_mode
    global landing_zone_visible
    global landing_zone_center
    global laser_frame_number

    try:
        print("[VIDEO] Подключение камеры...")

        camera = Camera()
        viewer = ImageViewer()

        print()
        print("=" * 60)
        print("Видеопоток запущен")
        print(f"Открой в браузере: http://10.42.0.1:8889/{STREAM_NAME}")
        print("=" * 60)
        print()

        while running:

            frame = camera.get_cv_frame(timeout=2.0)

            if frame is None:
                time.sleep(0.02)
                continue

            frame_h, frame_w = frame.shape[:2]

            corners, ids, detection_mode, net_mask = (
                detect_target_aruco_robust(frame)
            )

            current_target_visible = False
            current_target_center = None
            current_target_corners = None
            target_from_memory = False

            if ids is not None:

                for marker_corners, marker_id in zip(
                    corners,
                    ids.flatten()
                ):
                    marker_id = int(marker_id)

                    points = (
                        marker_corners
                        .reshape((4, 2))
                        .astype(int)
                    )

                    top_left = points[0]
                    top_right = points[1]
                    bottom_right = points[2]
                    bottom_left = points[3]

                    # Обводка маркера
                    cv2.line(
                        frame,
                        tuple(top_left),
                        tuple(top_right),
                        (0, 255, 0),
                        2
                    )

                    cv2.line(
                        frame,
                        tuple(top_right),
                        tuple(bottom_right),
                        (0, 255, 0),
                        2
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_right),
                        tuple(bottom_left),
                        (0, 255, 0),
                        2
                    )

                    cv2.line(
                        frame,
                        tuple(bottom_left),
                        tuple(top_left),
                        (0, 255, 0),
                        2
                    )

                    center_x = int(
                        (
                            top_left[0]
                            + top_right[0]
                            + bottom_right[0]
                            + bottom_left[0]
                        ) / 4
                    )

                    center_y = int(
                        (
                            top_left[1]
                            + top_right[1]
                            + bottom_right[1]
                            + bottom_left[1]
                        ) / 4
                    )

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"ARUCO ID: {marker_id}",
                        (
                            int(top_left[0]),
                            max(20, int(top_left[1]) - 10)
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )

                    if marker_id == TARGET_MARKER_ID:
                        current_target_visible = True
                        current_target_center = (
                            center_x,
                            center_y
                        )
                        current_target_corners = marker_corners

                        last_target_detection_time = time.monotonic()
                        last_target_center = current_target_center
                        last_target_corners = (
                            marker_corners.copy()
                        )
                        last_target_detection_mode = detection_mode

                        cv2.putText(
                            frame,
                            "TARGET",
                            (
                                center_x + 10,
                                center_y - 10
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

            # При кратком пропуске распознавания используем последнее
            # достоверное положение. Это не создаёт новую метку, а лишь
            # переживает несколько кадров, закрытых линией сетки.
            if not current_target_visible:
                detection_age = (
                    time.monotonic()
                    - last_target_detection_time
                )

                if (
                    last_target_center is not None
                    and last_target_corners is not None
                    and detection_age <= ARUCO_DETECTION_MEMORY
                ):
                    current_target_visible = True
                    current_target_center = last_target_center
                    current_target_corners = (
                        last_target_corners.copy()
                    )
                    detection_mode = (
                        f"MEM:{last_target_detection_mode}"
                    )
                    target_from_memory = True

            # ====================================================
            # ЛАЗЕРНЫЙ ПРИЁМ ПО КАДРАМ В ЦЕНТРЕ ARUCO 5
            # ====================================================
            laser_frame_number += 1

            laser_roi = find_laser_roi(
                current_target_corners,
                frame.shape,
                laser_frame_number
            )

            laser_visible, red_pixels, peak_red = detect_red_laser(
                frame,
                laser_roi
            )

            if laser_rx_running:
                if laser_roi is not None:
                    laser_decoder.update(laser_visible)
                else:
                    laser_decoder.status = "WAIT ARUCO 5"

            if laser_roi is not None:
                rx1, ry1, rx2, ry2 = laser_roi
                cv2.rectangle(
                    frame,
                    (rx1, ry1),
                    (rx2, ry2),
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"LASER {'ON' if laser_visible else 'OFF'} "
                        f"RPIX={red_pixels} RMAX={peak_red}"
                    ),
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"RX: {laser_decoder.status}",
                    (20, 158),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            # ====================================================
            # ПОСАДОЧНАЯ ЗОНА: ОРАНЖЕВЫЙ КРУГ С БЕЛОЙ "Н"
            # ====================================================

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            orange_mask = cv2.inRange(
                hsv,
                LANDING_ORANGE_HSV_LOW,
                LANDING_ORANGE_HSV_HIGH
            )

            # Убираем мелкий шум.
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (7, 7)
            )

            orange_mask = cv2.morphologyEx(
                orange_mask,
                cv2.MORPH_OPEN,
                kernel
            )

            orange_mask = cv2.morphologyEx(
                orange_mask,
                cv2.MORPH_CLOSE,
                kernel
            )

            contours, _ = cv2.findContours(
                orange_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            current_landing_visible = False
            current_landing_center = None

            if contours:
                valid_contours = [
                    c for c in contours
                    if cv2.contourArea(c) >= LANDING_MIN_CONTOUR_AREA
                ]

                if valid_contours:
                    landing_contour = max(
                        valid_contours,
                        key=cv2.contourArea
                    )

                    area = cv2.contourArea(landing_contour)
                    perimeter = cv2.arcLength(
                        landing_contour,
                        True
                    )

                    circularity = 0.0
                    if perimeter > 0:
                        circularity = (
                            4.0 * math.pi * area
                            / (perimeter * perimeter)
                        )

                    # Круг не обязан быть идеально круглым из-за перспективы.
                    if circularity >= 0.45:
                        moments = cv2.moments(landing_contour)

                        if moments["m00"] != 0:
                            landing_cx = int(
                                moments["m10"] / moments["m00"]
                            )
                            landing_cy = int(
                                moments["m01"] / moments["m00"]
                            )

                            current_landing_visible = True
                            current_landing_center = (
                                landing_cx,
                                landing_cy
                            )

                            cv2.drawContours(
                                frame,
                                [landing_contour],
                                -1,
                                (0, 165, 255),
                                3
                            )

                            cv2.circle(
                                frame,
                                (landing_cx, landing_cy),
                                7,
                                (255, 255, 255),
                                -1
                            )

                            cv2.putText(
                                frame,
                                "LANDING ZONE",
                                (
                                    landing_cx + 12,
                                    landing_cy
                                ),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.65,
                                (0, 165, 255),
                                2
                            )

            # Центр изображения
            image_center_x = frame_w // 2
            image_center_y = frame_h // 2

            cv2.drawMarker(
                frame,
                (image_center_x, image_center_y),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=30,
                thickness=2
            )

            with marker_lock:
                target_marker_visible = current_target_visible
                target_marker_center = current_target_center
                target_frame_size = (frame_w, frame_h)
                landing_zone_visible = current_landing_visible
                landing_zone_center = current_landing_center

            # Информация поверх картинки
            cv2.rectangle(
                frame,
                (10, 10),
                (500, 130),
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
                f"SEARCH ARUCO: {TARGET_MARKER_ID}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

            status = (
                "TARGET: FOUND"
                if current_target_visible
                else "TARGET: ---"
            )

            cv2.putText(
                frame,
                status,
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"DETECT: {detection_mode}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (
                    (0, 180, 255)
                    if target_from_memory
                    else (255, 255, 255)
                ),
                2
            )

            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=VIDEO_FPS
            )

    except Exception as e:
        print("[VIDEO ERROR]", e)

    finally:
        print("[VIDEO] Остановка")

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
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def marker_is_visible():
    with marker_lock:
        return target_marker_visible


def get_marker_data():
    with marker_lock:
        return (
            target_marker_visible,
            target_marker_center,
            target_frame_size
        )


def get_landing_zone_data():
    with marker_lock:
        return (
            landing_zone_visible,
            landing_zone_center,
            target_frame_size
        )


def wait_point():
    """Ожидание достижения текущей точки."""
    while not pioneer.point_reached():
        time.sleep(LOOP_DELAY)


def calculate_flight_time(x, y, z, speed=FLIGHT_SPEED):
    """
    Рассчитывает желаемое время достижения точки для Pioneer SDK2.

    time = расстояние / скорость

    go_to_local_point() в SDK2 принимает time как целое число секунд,
    поэтому используем ceil(), чтобы не задавать время меньше расчётного.
    """
    if speed <= 0:
        raise ValueError("FLIGHT_SPEED должен быть больше 0")

    pos = pioneer.get_local_position_lps()

    if pos is None or len(pos) < 3:
        return MIN_FLIGHT_TIME

    dx = x - pos[0]
    dy = y - pos[1]
    dz = z - pos[2]

    distance = math.sqrt(
        dx * dx +
        dy * dy +
        dz * dz
    )

    flight_time = math.ceil(distance / speed)

    return max(MIN_FLIGHT_TIME, flight_time)


def go_to_point_with_speed(x, y, z, yaw=0.0, speed=FLIGHT_SPEED):
    """
    Отправляет штатную команду go_to_local_point(),
    автоматически рассчитывая параметр time из желаемой скорости.
    """
    flight_time = calculate_flight_time(
        x=x,
        y=y,
        z=z,
        speed=speed
    )

    print(
        f"[FLIGHT] -> X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
        f"speed≈{speed:.2f} m/s, time={flight_time}s"
    )

    pioneer.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        time=flight_time
    )

    return flight_time


def generate_x_columns():
    """
    Автоматически создаёт X-координаты полос змейки.

    Для настроек:
        X_START = -1.6
        X_END   = -3.1
        X_STEP  = -0.4

    получится:
        -1.6, -2.0, -2.4, -2.8, -3.1
    """

    if X_STEP == 0:
        raise ValueError("X_STEP не может быть 0")

    if X_START > X_END and X_STEP > 0:
        raise ValueError(
            "Для движения к меньшему X "
            "значение X_STEP должно быть отрицательным"
        )

    if X_START < X_END and X_STEP < 0:
        raise ValueError(
            "Для движения к большему X "
            "значение X_STEP должно быть положительным"
        )

    columns = []
    x = X_START

    if X_STEP < 0:
        while x >= X_END:
            columns.append(round(x, 4))
            x += X_STEP
    else:
        while x <= X_END:
            columns.append(round(x, 4))
            x += X_STEP

    if (
        not columns
        or abs(columns[-1] - X_END) > 0.001
    ):
        columns.append(X_END)

    return columns


def fly_and_search(x, y, z, yaw):
    """
    Летим к точке через go_to_local_point(..., time=...),
    одновременно проверяя появление ArUco ID 5.

    Возвращает True сразу после обнаружения маркера.
    """

    go_to_point_with_speed(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        speed=FLIGHT_SPEED
    )

    while True:

        if marker_is_visible():
            print(
                f"[ARUCO] Найден целевой ID "
                f"{TARGET_MARKER_ID}"
            )

            # Перебиваем текущую целевую точку:
            # задаём текущую позицию как новую цель.
            stop_at_current_position()

            return True

        if pioneer.point_reached():
            return marker_is_visible()

        time.sleep(LOOP_DELAY)


def stop_at_current_position():
    """
    Отменяет продолжение предыдущего перелёта:
    текущая позиция отправляется как новая целевая точка.
    """
    pos = pioneer.get_local_position_lps()

    if pos is None or len(pos) < 3:
        return

    x, y, z = pos[:3]

    pioneer.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=YAW,
        time=MIN_FLIGHT_TIME
    )

    time.sleep(0.2)


# ============================================================
# ЦЕНТРИРОВАНИЕ НАД ARUCO
# ============================================================

class CenteringError(Exception):
    """Ошибка центрирования: требуется возврат домой."""
    pass



def center_over_target():
    """
    Плавное центрирование над ArUco с P-регулятором.

    Если ArUco кратковременно пропадает из кадра, дрон не замирает
    сразу, а продолжает медленно лететь в направлении последней
    корректирующей команды. Это помогает повторно захватить метку,
    если она ушла за край кадра.
    """

    print("[CENTER] Начинаю плавное центрирование")

    stop_at_current_position()

    started = time.time()
    stable_since = None
    marker_last_seen = time.time()

    filtered_x = None
    filtered_y = None

    # Последняя ненулевая команда движения к метке.
    # При временной потере ArUco будем продолжать её с меньшей скоростью.
    last_recovery_vx = 0.0
    last_recovery_vy = 0.0
    marker_was_visible = True

    def clamp(value, low, high):
        return max(low, min(high, value))

    def clamp_abs(value, max_abs):
        return clamp(value, -max_abs, max_abs)

    def speed_from_error(error_px):
        abs_error = abs(error_px)

        if abs_error <= CENTER_TOLERANCE_PX:
            return 0.0

        speed = abs_error * CENTER_KP
        speed = clamp(
            speed,
            CENTER_MIN_SPEED,
            CENTER_MAX_SPEED
        )

        return speed if error_px > 0 else -speed

    while True:
        now = time.time()

        received_digit = get_received_laser_digit()
        if received_digit is not None:
            pioneer.set_manual_speed_body_fixed(
                vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0
            )
            stop_at_current_position()
            print(
                f"[CENTER] Получен лазерный сигнал {received_digit}. "
                "Центрирование немедленно прервано."
            )
            return received_digit

        if now - started > CENTER_TIMEOUT:
            pioneer.set_manual_speed_body_fixed(
                vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0
            )
            raise CenteringError(
                "Не удалось отцентрироваться над ArUco "
                "за отведённое время"
            )

        visible, center, frame_size = get_marker_data()

        if (
            not visible
            or center is None
            or frame_size is None
        ):
            stable_since = None
            lost_for = now - marker_last_seen

            if lost_for > MARKER_LOST_TIMEOUT:
                pioneer.set_manual_speed_body_fixed(
                    vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0
                )
                raise CenteringError(
                    f"ArUco потерян более {MARKER_LOST_TIMEOUT:.1f} с"
                )

            # Продолжаем двигаться туда, куда только что корректировались,
            # но медленнее и с ограничением максимальной скорости.
            recovery_vx = clamp_abs(
                last_recovery_vx * LOST_MARKER_SPEED_FACTOR,
                LOST_MARKER_MAX_SPEED
            )
            recovery_vy = clamp_abs(
                last_recovery_vy * LOST_MARKER_SPEED_FACTOR,
                LOST_MARKER_MAX_SPEED
            )

            if marker_was_visible:
                print(
                    f"[CENTER LOST] ArUco временно потерян. "
                    f"Продолжаю в последнюю сторону: "
                    f"vx={recovery_vx:+.3f}, vy={recovery_vy:+.3f}"
                )
            marker_was_visible = False

            pioneer.set_manual_speed_body_fixed(
                vx=recovery_vx,
                vy=recovery_vy,
                vz=0.0,
                yaw_rate=0.0
            )

            time.sleep(0.06)
            continue

        # Маркер снова появился. После потери лучше начать фильтрацию заново,
        # чтобы старые координаты не тянули управление в неверную сторону.
        if not marker_was_visible:
            print("[CENTER] ArUco снова найден")
            filtered_x = None
            filtered_y = None

        marker_was_visible = True
        marker_last_seen = now

        marker_x_px, marker_y_px = center
        frame_w, frame_h = frame_size

        if filtered_x is None:
            filtered_x = float(marker_x_px)
            filtered_y = float(marker_y_px)
        else:
            filtered_x = (
                CENTER_FILTER_ALPHA * marker_x_px
                + (1.0 - CENTER_FILTER_ALPHA) * filtered_x
            )
            filtered_y = (
                CENTER_FILTER_ALPHA * marker_y_px
                + (1.0 - CENTER_FILTER_ALPHA) * filtered_y
            )

        image_x_px = frame_w / 2.0
        image_y_px = frame_h / 2.0

        error_x = filtered_x - image_x_px
        error_y = filtered_y - image_y_px

        centered_x = abs(error_x) <= CENTER_TOLERANCE_PX
        centered_y = abs(error_y) <= CENTER_TOLERANCE_PX

        print(
            f"[CENTER] "
            f"dx={error_x:+.1f}px "
            f"dy={error_y:+.1f}px"
        )

        if centered_x and centered_y:
            pioneer.set_manual_speed_body_fixed(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0
            )

            if stable_since is None:
                stable_since = now

            if now - stable_since >= CENTER_STABLE_TIME:
                print("[CENTER] Центрирование завершено")
                stop_at_current_position()
                return get_received_laser_digit()

            time.sleep(0.05)
            continue

        stable_since = None

        # По горизонтали кадра: маркер справа -> летим вправо.
        vx = speed_from_error(error_x)

        # По вертикали кадра оставляем знак как в исходной версии:
        # маркер ниже центра -> движение назад.
        vy = -speed_from_error(error_y)

        # Запоминаем последнее направление только когда команда ненулевая.
        # По нему будем осторожно продолжать движение при потере изображения.
        if abs(vx) > 1e-6 or abs(vy) > 1e-6:
            last_recovery_vx = vx
            last_recovery_vy = vy

        print(
            f"[CENTER CMD] "
            f"vx={vx:+.3f} "
            f"vy={vy:+.3f}"
        )

        pioneer.set_manual_speed_body_fixed(
            vx=vx,
            vy=vy,
            vz=0.0,
            yaw_rate=0.0
        )

        time.sleep(0.08)


# ============================================================
# ЦЕНТРИРОВКА НАД ПОСАДОЧНОЙ ЗОНОЙ
# ============================================================

def center_over_landing_zone():
    """
    Небольшая визуальная коррекция над оранжевой посадочной зоной.

    Центровка специально неточная: достаточно попасть в широкую
    область LANDING_CENTER_TOLERANCE_PX вокруг центра кадра.
    """

    print("[LANDING VISION] Поиск оранжевой посадочной зоны")

    started = time.time()
    stable_since = None

    filtered_x = None
    filtered_y = None

    def clamp(value, low, high):
        return max(low, min(high, value))

    def speed_from_error(error_px):
        if abs(error_px) <= LANDING_CENTER_TOLERANCE_PX:
            return 0.0

        speed = abs(error_px) * LANDING_CENTER_KP
        speed = clamp(
            speed,
            LANDING_CENTER_MIN_SPEED,
            LANDING_CENTER_MAX_SPEED
        )

        return speed if error_px > 0 else -speed

    while True:
        now = time.time()

        if now - started > LANDING_CENTER_TIMEOUT:
            print(
                "[LANDING VISION] Зона не отцентрирована за таймаут. "
                "Продолжаю посадку по координатной точке."
            )
            stop_at_current_position()
            return False

        visible, center, frame_size = get_landing_zone_data()

        if (
            not visible
            or center is None
            or frame_size is None
        ):
            stable_since = None

            pioneer.set_manual_speed_body_fixed(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0
            )

            time.sleep(0.08)
            continue

        zone_x, zone_y = center
        frame_w, frame_h = frame_size

        if filtered_x is None:
            filtered_x = float(zone_x)
            filtered_y = float(zone_y)
        else:
            filtered_x = (
                LANDING_CENTER_FILTER_ALPHA * zone_x
                + (1.0 - LANDING_CENTER_FILTER_ALPHA) * filtered_x
            )

            filtered_y = (
                LANDING_CENTER_FILTER_ALPHA * zone_y
                + (1.0 - LANDING_CENTER_FILTER_ALPHA) * filtered_y
            )

        error_x = filtered_x - frame_w / 2.0
        error_y = filtered_y - frame_h / 2.0

        centered_x = (
            abs(error_x) <= LANDING_CENTER_TOLERANCE_PX
        )
        centered_y = (
            abs(error_y) <= LANDING_CENTER_TOLERANCE_PX
        )

        print(
            f"[LANDING CENTER] "
            f"dx={error_x:+.1f}px "
            f"dy={error_y:+.1f}px"
        )

        if centered_x and centered_y:
            pioneer.set_manual_speed_body_fixed(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0
            )

            if stable_since is None:
                stable_since = now

            if (
                now - stable_since
                >= LANDING_CENTER_STABLE_TIME
            ):
                print(
                    "[LANDING CENTER] Посадочная зона "
                    "достаточно отцентрирована"
                )

                stop_at_current_position()
                return True

            time.sleep(0.06)
            continue

        stable_since = None

        vx = speed_from_error(error_x)
        vy = -speed_from_error(error_y)

        pioneer.set_manual_speed_body_fixed(
            vx=vx,
            vy=vy,
            vz=0.0,
            yaw_rate=0.0
        )

        time.sleep(0.08)


# ============================================================
# ПОСАДОЧНЫЕ ТОЧКИ
# ============================================================

def get_landing_point(number=LANDING_POINT_NUMBER):
    """
    Возвращает выбранную посадочную точку в метрах.

    LANDING_POINTS_CM хранит координаты в сантиметрах
    относительно стартовой точки (0, 0).

    Пример:
        (450, 70) см -> (4.50, 0.70) м
    """
    if number not in LANDING_POINTS_CM:
        raise ValueError(
            f"Нет посадочной точки №{number}. "
            f"Доступны: {sorted(LANDING_POINTS_CM)}"
        )

    x_cm, y_cm = LANDING_POINTS_CM[number]

    return (
        x_cm / 100.0,
        y_cm / 100.0
    )


def return_to_landing_point_and_land(number=LANDING_POINT_NUMBER):
    """
    Подлёт к выбранной посадочной точке, визуальная коррекция
    по оранжевому кругу, зависание 3 секунды и посадка.
    """
    landing_x, landing_y = get_landing_point(number)

    x_cm, y_cm = LANDING_POINTS_CM[number]

    print(
        f"[LANDING POINT] №{number}: "
        f"({x_cm}, {y_cm}) cm -> "
        f"X={landing_x:.3f} m, Y={landing_y:.3f} m"
    )

    # Сначала штатный точный подлёт по координатам.
    go_to_point_with_speed(
        x=landing_x,
        y=landing_y,
        z=LANDING_APPROACH_HEIGHT,
        yaw=LANDING_YAW,
        speed=FLIGHT_SPEED
    )

    wait_point()

    print("[LANDING POINT] Координатная точка достигнута")

    # Затем небольшая визуальная коррекция по оранжевому кругу.
    centered = center_over_landing_zone()

    if centered:
        print(
            f"[LANDING HOVER] Зависание "
            f"{LANDING_HOVER_TIME:.1f} секунд"
        )
    else:
        print(
            "[LANDING HOVER] Визуальная центровка не завершена. "
            "Зависание над координатной точкой."
        )

    # Удерживаем текущую позицию перед посадкой.
    hover_pos = pioneer.get_local_position_lps()

    if hover_pos is not None and len(hover_pos) >= 3:
        pioneer.go_to_local_point(
            x=hover_pos[0],
            y=hover_pos[1],
            z=hover_pos[2],
            yaw=LANDING_YAW,
            time=MIN_FLIGHT_TIME
        )

    time.sleep(LANDING_HOVER_TIME)

    print("[LAND] Посадка")
    pioneer.land()
    wait_point()
    print("[LAND] Посадка завершена")


# ============================================================
# ВОЗВРАТ ДОМОЙ
# ============================================================

def return_home_and_land():

    print("[HOME] Возврат на стартовую точку")

    go_to_point_with_speed(
        x=HOME_X,
        y=HOME_Y,
        z=RETURN_HEIGHT,
        yaw=0.0,
        speed=FLIGHT_SPEED
    )

    wait_point()

    print("[HOME] Точка старта достигнута")
    print("[LAND] Посадка")

    pioneer.land()
    wait_point()

    print("[LAND] Посадка завершена")


def land_here():
    """Посадка в текущем месте, без возврата домой."""
    print("[LAND] Посадка в текущем месте")

    try:
        stop_at_current_position()
    except Exception:
        pass

    pioneer.land()

    try:
        wait_point()
    except Exception:
        pass

    print("[LAND] Посадка в текущем месте завершена")


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

video_thread = None

try:

    # --------------------------------------------------------
    # Запускаем трансляцию раньше полёта
    # --------------------------------------------------------

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )

    video_thread.start()

    time.sleep(1.0)

    print("[FLIGHT] ARM")
    pioneer.arm()

    print("[FLIGHT] Взлёт")
    pioneer.takeoff()
    # не добавлять wait_point() после взлёта, т.к. он не нужен

    print("[FLIGHT] Взлёт завершён")

    # --------------------------------------------------------
    # 1. Первая точка
    # --------------------------------------------------------

    go_to_point_with_speed(
        x=ENTRY_POINT[0],
        y=ENTRY_POINT[1],
        z=ENTRY_POINT[2],
        yaw=ENTRY_POINT[3],
        speed=FLIGHT_SPEED
    )

    wait_point()

    # --------------------------------------------------------
    # 2. Вход в область поиска
    # --------------------------------------------------------

    go_to_point_with_speed(
        x=SEARCH_ENTRY_POINT[0],
        y=SEARCH_ENTRY_POINT[1],
        z=SEARCH_ENTRY_POINT[2],
        yaw=SEARCH_ENTRY_POINT[3],
        speed=FLIGHT_SPEED
    )

    wait_point()

    # --------------------------------------------------------
    # 3. Генерируем змейку
    # --------------------------------------------------------

    x_columns = generate_x_columns()

    print("[SEARCH] Полосы X:", x_columns)

    target_found = False

    # --------------------------------------------------------
    # Выполняем только ОДИН полный проход змейкой
    # --------------------------------------------------------

    print("[SEARCH] Единственный проход змейкой")

    for column_index, x in enumerate(x_columns):

        # Чередуем направление по Y
        if column_index % 2 == 0:
            y_start = Y_MIN
            y_end = Y_MAX
        else:
            y_start = Y_MAX
            y_end = Y_MIN

        # Переход на начало полосы
        target_found = fly_and_search(
            x,
            y_start,
            SEARCH_HEIGHT,
            YAW
        )

        if target_found:
            break

        # Непрерывный пролёт вдоль всей полосы
        target_found = fly_and_search(
            x,
            y_end,
            SEARCH_HEIGHT,
            YAW
        )

        if target_found:
            break

    # --------------------------------------------------------
    # 4. Если маркер НЕ найден — сразу домой и посадка
    # --------------------------------------------------------

    if not target_found:
        print(
            f"[SEARCH] ArUco ID {TARGET_MARKER_ID} "
            "не найден за один проход."
        )
        print("[SEARCH] Поиск завершён. Возврат домой.")
        return_to_landing_point_and_land(LANDING_POINT_NUMBER)

    else:
        # ----------------------------------------------------
        # 5. Маркер найден — центрирование
        # ----------------------------------------------------

        print("[SEARCH] Маршрут поиска остановлен")

        start_laser_receiver()

        try:
            received_digit = center_over_target()

        except CenteringError as e:
            print()
            print("[CENTER ERROR]", e)
            received_digit = get_received_laser_digit()

            if received_digit is None:
                print(
                    "[LASER RX] Сигнал 0..8 не получен. "
                    "Использую посадочную зону №1."
                )
                selected_landing_number = 1
            else:
                selected_landing_number = landing_number_from_signal(
                    received_digit
                )

            stop_laser_receiver()
            return_to_landing_point_and_land(selected_landing_number)

        else:
            stop_laser_receiver()

            # После успешной центровки используем принятый сигнал.
            # Если сигнала нет, по условию выбираем зону №1.
            if received_digit is None:
                received_digit = get_received_laser_digit()

            if received_digit is None:
                selected_landing_number = 1
                print(
                    "[LASER RX] Сигнал 0..8 не получен. "
                    "Использую посадочную зону №1."
                )
            else:
                selected_landing_number = landing_number_from_signal(
                    received_digit
                )
                print(
                    f"[LASER RX] Число {received_digit} -> "
                    f"посадочная зона №{selected_landing_number}"
                )

            # Если сигнал уже принят, центрирование было прервано сразу.
            # Дополнительное зависание над ArUco не требуется.
            return_to_landing_point_and_land(selected_landing_number)



except KeyboardInterrupt:

    print("\n[STOP] Остановка оператором")
    print("[SAFETY] Посадка в текущем месте")

    try:
        land_here()
    except Exception as landing_error:
        print("[LAND ERROR]", landing_error)
        try:
            pioneer.land()
        except Exception:
            pass


except Exception as e:

    print()
    print("[ERROR]", repr(e))
    print("[SAFETY] Пытаюсь вернуться на стартовую точку")

    try:
        return_home_and_land()

    except Exception as return_error:
        print("[HOME ERROR]", repr(return_error))
        print("[SAFETY] Возврат не удался. Посадка в текущем месте.")

        try:
            land_here()
        except Exception:
            try:
                pioneer.land()
            except Exception:
                pass


finally:

    stop_laser_receiver()
    running = False

    if video_thread is not None:
        video_thread.join(timeout=3.0)

    try:
        pioneer.close_connection()
    except Exception:
        pass

    print("[END] Программа завершена")
