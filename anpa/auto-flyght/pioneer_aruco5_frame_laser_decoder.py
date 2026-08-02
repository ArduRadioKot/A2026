import time
import threading
from collections import deque

import cv2
import numpy as np

from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera


# ============================================================
# STREAM / CAMERA
# ============================================================

STREAM_NAME = "pioneer"
STREAM_FPS = 20

running = True

servo_camera = ServoCamera()
servo_camera.set_angle(-80)


# ============================================================
# LASER PACKET
# Совместимо с laser_packet_drone.py:
#
# PREAMBLE: 16 бит 1010101010101010
# MESSAGE : 4 бита
# CRC8    : 8 бит
#
# 0 -> лазер включён 0.3 с
# 1 -> лазер включён 0.9 с
# gap -> лазер выключен 0.2 с
# ============================================================

PREAMBLE = 0b1010101010101010
PREAMBLE_LEN = 16
MSG_LEN = 4
CRC_LEN = 8
PACKET_LEN = PREAMBLE_LEN + MSG_LEN + CRC_LEN

PREAMBLE_BITS = [
    (PREAMBLE >> i) & 1
    for i in range(PREAMBLE_LEN - 1, -1, -1)
]

DOT_DURATION = 0.3
DASH_DURATION = 0.9
GAP_DURATION = 0.2

# При 20 FPS ожидается примерно:
# 0 -> 6 кадров
# 1 -> 18 кадров
EXPECTED_DOT_FRAMES = max(1, round(DOT_DURATION * STREAM_FPS))
EXPECTED_DASH_FRAMES = max(1, round(DASH_DURATION * STREAM_FPS))

# Граница между 0 и 1 по числу кадров.
FRAME_SPLIT = (EXPECTED_DOT_FRAMES + EXPECTED_DASH_FRAMES) / 2.0

# Очень короткие вспышки считаем шумом.
MIN_PULSE_FRAMES = max(2, round(EXPECTED_DOT_FRAMES * 0.45))

# Если вспышка намного длиннее ожидаемой "1", считаем её ошибкой.
MAX_PULSE_FRAMES = round(EXPECTED_DASH_FRAMES * 1.8)

# Небольшая фильтрация состояния лазера:
# смена ON/OFF принимается только после нескольких одинаковых кадров подряд.
STATE_CONFIRM_FRAMES = 2


# ============================================================
# ARUCO ID 5
# ============================================================

TARGET_ARUCO_ID = 5

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    cv2.aruco.DetectorParameters()
)

# Последняя известная область метки.
# Это позволяет пережить короткое выпадение распознавания ArUco.
last_marker_roi = None
last_marker_seen_frame = -10_000
MARKER_MEMORY_FRAMES = 10


# ============================================================
# ДЕТЕКЦИЯ КРАСНОГО ЛАЗЕРА
# ============================================================

# Лазер находится в центре ArUco 5.
# Поэтому анализируем маленькую центральную область метки,
# а не весь кадр.

LASER_ROI_SCALE = 0.34

# Порог "красности".
MIN_RED = 150
RED_DOMINANCE = 55

# Минимальное количество красных пикселей внутри ROI.
# Для маленькой точки лучше использовать абсолютное число,
# а не среднюю яркость всей области.
MIN_RED_PIXELS = 3

# Дополнительный HSV-порог насыщенности/яркости.
MIN_SATURATION = 120
MIN_VALUE = 140


# ============================================================
# CRC8
# ============================================================

def crc8(data: int, length_bits: int, poly: int = 0x07) -> int:
    """CRC8, идентичный передатчику."""
    n_bytes = (length_bits + 7) // 8
    data_bytes = data.to_bytes(n_bytes, byteorder="big")

    crc = 0x00

    for byte in data_bytes:
        crc ^= byte

        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF

    return crc


def bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


# ============================================================
# FRAME-BASED DECODER
# ============================================================

class FrameLaserDecoder:
    """
    Декодирует импульсы по количеству кадров, на которых виден лазер.

    Основная идея:
      ~6 кадров  -> бит 0
      ~18 кадров -> бит 1

    При этом длительность по time.monotonic() тоже сохраняется
    только для диагностики. Решение о бите принимается по кадрам.
    """

    def __init__(self):
        self.raw_state = False
        self.stable_state = False
        self.same_raw_frames = 0

        self.on_frames = 0
        self.on_start_time = None

        self.bit_history = deque(maxlen=200)

        self.synced = False
        self.payload_bits = []

        self.last_message = None
        self.last_crc = None

        self.valid_packets = 0
        self.crc_errors = 0
        self.bad_pulses = 0

        self.status = "WAIT ARUCO 5"

        self.last_pulse_frames = 0
        self.last_pulse_seconds = 0.0
        self.last_bit = None

    def reset_sync(self):
        self.synced = False
        self.payload_bits = []

    def _find_preamble(self):
        if len(self.bit_history) < PREAMBLE_LEN:
            return False

        tail = list(self.bit_history)[-PREAMBLE_LEN:]

        if tail == PREAMBLE_BITS:
            self.synced = True
            self.payload_bits = []
            self.status = "PREAMBLE OK"
            print("[LASER] PREAMBLE FOUND")
            return True

        return False

    def _decode_payload(self):
        if len(self.payload_bits) != MSG_LEN + CRC_LEN:
            return

        message_bits = self.payload_bits[:MSG_LEN]
        crc_bits = self.payload_bits[MSG_LEN:]

        message = bits_to_int(message_bits)
        received_crc = bits_to_int(crc_bits)
        calculated_crc = crc8(message, MSG_LEN)

        if received_crc == calculated_crc:
            self.last_message = message
            self.last_crc = received_crc
            self.valid_packets += 1
            self.status = f"OK MSG={message}"

            print()
            print("=" * 60)
            print("[LASER] PACKET OK")
            print(f"[LASER] MESSAGE: {message}")
            print(f"[LASER] CRC: 0x{received_crc:02X}")
            print("=" * 60)
            print()

        else:
            self.crc_errors += 1
            self.status = "CRC ERROR"

            print()
            print("=" * 60)
            print("[LASER] CRC ERROR")
            print(f"[LASER] MESSAGE: {message}")
            print(f"[LASER] RECEIVED CRC: 0x{received_crc:02X}")
            print(f"[LASER] CALCULATED CRC: 0x{calculated_crc:02X}")
            print("=" * 60)
            print()

        # После пакета снова ищем преамбулу.
        self.reset_sync()

    def add_bit(self, bit, pulse_frames, pulse_seconds):
        self.last_bit = bit
        self.last_pulse_frames = pulse_frames
        self.last_pulse_seconds = pulse_seconds

        self.bit_history.append(bit)

        print(
            f"[LASER] BIT={bit} "
            f"frames={pulse_frames} "
            f"time={pulse_seconds:.3f}s"
        )

        if not self.synced:
            self.status = "SEARCH PREAMBLE"
            self._find_preamble()
            return

        self.payload_bits.append(bit)
        self.status = (
            f"DATA {len(self.payload_bits)}/{MSG_LEN + CRC_LEN}"
        )

        if len(self.payload_bits) >= MSG_LEN + CRC_LEN:
            self._decode_payload()

    def finish_pulse(self):
        frames = self.on_frames
        seconds = 0.0

        if self.on_start_time is not None:
            seconds = time.monotonic() - self.on_start_time

        self.on_frames = 0
        self.on_start_time = None

        if frames < MIN_PULSE_FRAMES:
            # Короткий красный блик/шум.
            return

        if frames > MAX_PULSE_FRAMES:
            self.bad_pulses += 1
            self.status = "BAD LONG PULSE"
            self.reset_sync()

            print(
                f"[LASER] BAD PULSE: {frames} frames, "
                f"{seconds:.3f}s"
            )
            return

        # Классификация именно ПО КОЛИЧЕСТВУ КАДРОВ.
        bit = 1 if frames >= FRAME_SPLIT else 0
        self.add_bit(bit, frames, seconds)

    def update(self, laser_visible):
        """
        Вызывать один раз на каждый полученный кадр.
        """

        # Пока устойчиво считаем, что лазер ON,
        # каждый кадр увеличивает длину текущего импульса.
        if self.stable_state:
            self.on_frames += 1

        # Дребезг/случайный красный пиксель не должен создавать фронт.
        if laser_visible == self.raw_state:
            self.same_raw_frames += 1
        else:
            self.raw_state = laser_visible
            self.same_raw_frames = 1

        if (
            self.raw_state != self.stable_state
            and self.same_raw_frames >= STATE_CONFIRM_FRAMES
        ):
            # OFF -> ON
            if self.raw_state:
                self.stable_state = True
                self.on_frames = STATE_CONFIRM_FRAMES
                self.on_start_time = time.monotonic()
                self.status = "LASER ON"

            # ON -> OFF
            else:
                self.stable_state = False
                self.finish_pulse()


decoder = FrameLaserDecoder()


# ============================================================
# ARUCO / ROI
# ============================================================

def find_aruco5_roi(frame, frame_number):
    """
    Находит ArUco ID 5 и возвращает маленькую ROI вокруг его центра:
      (x1, y1, x2, y2), corners, ids

    Размер ROI зависит от физического размера метки в кадре,
    поэтому при изменении высоты дрона область автоматически масштабируется.
    """
    global last_marker_roi, last_marker_seen_frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    found_roi = None

    if ids is not None:
        flat_ids = ids.flatten()

        for i, marker_id in enumerate(flat_ids):
            if int(marker_id) != TARGET_ARUCO_ID:
                continue

            pts = corners[i].reshape(4, 2)

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            side_lengths = [
                np.linalg.norm(pts[(j + 1) % 4] - pts[j])
                for j in range(4)
            ]
            marker_size = float(np.mean(side_lengths))

            roi_size = max(8, int(marker_size * LASER_ROI_SCALE))
            half = roi_size // 2

            h, w = frame.shape[:2]

            x1 = max(0, int(cx) - half)
            y1 = max(0, int(cy) - half)
            x2 = min(w, int(cx) + half + 1)
            y2 = min(h, int(cy) + half + 1)

            if x2 > x1 and y2 > y1:
                found_roi = (x1, y1, x2, y2)
                last_marker_roi = found_roi
                last_marker_seen_frame = frame_number

            break

    # Короткая память позиции полезна, если засветка лазера
    # на 1-2 кадра ухудшила распознавание самой ArUco.
    if (
        found_roi is None
        and last_marker_roi is not None
        and frame_number - last_marker_seen_frame <= MARKER_MEMORY_FRAMES
    ):
        found_roi = last_marker_roi

    return found_roi, corners, ids


# ============================================================
# RED LASER DETECTOR
# ============================================================

def detect_red_laser(frame, roi_coords):
    """
    Ищет красную точку только около центра ArUco 5.

    Возвращает:
      visible,
      red_pixels,
      peak_red,
      mask
    """

    if roi_coords is None:
        return False, 0, 0, None

    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return False, 0, 0, None

    b, g, r = cv2.split(roi)

    # Красный канал должен быть не просто ярким,
    # а заметно сильнее зелёного и синего.
    red_mask_bgr = (
        (r >= MIN_RED)
        & ((r.astype(np.int16) - g.astype(np.int16)) >= RED_DOMINANCE)
        & ((r.astype(np.int16) - b.astype(np.int16)) >= RED_DOMINANCE)
    )

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Красный в HSV лежит около 0 и 180.
    mask1 = cv2.inRange(
        hsv,
        np.array([0, MIN_SATURATION, MIN_VALUE], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8)
    )

    mask2 = cv2.inRange(
        hsv,
        np.array([168, MIN_SATURATION, MIN_VALUE], dtype=np.uint8),
        np.array([179, 255, 255], dtype=np.uint8)
    )

    red_mask_hsv = (mask1 > 0) | (mask2 > 0)

    combined = red_mask_bgr & red_mask_hsv

    red_pixels = int(np.count_nonzero(combined))
    peak_red = int(np.max(r)) if r.size else 0

    visible = red_pixels >= MIN_RED_PIXELS

    debug_mask = (combined.astype(np.uint8) * 255)

    return visible, red_pixels, peak_red, debug_mask


# ============================================================
# VIDEO WORKER
# ============================================================

def video_worker():
    global running

    camera = None
    viewer = None
    frame_number = 0

    fps_times = deque(maxlen=30)

    try:
        camera = Camera()
        viewer = ImageViewer()

        print("=" * 70)
        print("Стрим + ArUco 5 + frame-based laser decoder")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}")
        print()
        print(
            f"Ожидаемые импульсы при {STREAM_FPS} FPS: "
            f"0≈{EXPECTED_DOT_FRAMES} frames, "
            f"1≈{EXPECTED_DASH_FRAMES} frames, "
            f"split={FRAME_SPLIT:.1f}"
        )
        print("=" * 70)

        while running:
            frame = camera.get_cv_frame(timeout=2)

            if frame is None:
                continue

            frame_number += 1
            now = time.monotonic()
            fps_times.append(now)

            measured_fps = 0.0
            if len(fps_times) >= 2:
                dt = fps_times[-1] - fps_times[0]
                if dt > 0:
                    measured_fps = (len(fps_times) - 1) / dt

            # ------------------------------------------------
            # ArUco 5
            # ------------------------------------------------

            roi_coords, corners, ids = find_aruco5_roi(
                frame,
                frame_number
            )

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    corners,
                    ids
                )

            marker_visible_now = False
            if ids is not None:
                marker_visible_now = TARGET_ARUCO_ID in [
                    int(v) for v in ids.flatten()
                ]

            # ------------------------------------------------
            # Red laser inside center of ArUco 5
            # ------------------------------------------------

            laser_visible, red_pixels, peak_red, _ = detect_red_laser(
                frame,
                roi_coords
            )

            # Декодируем только когда у нас есть актуальная/недавняя ROI.
            if roi_coords is not None:
                decoder.update(laser_visible)

                if decoder.status == "WAIT ARUCO 5":
                    decoder.status = "SEARCH PREAMBLE"
            else:
                decoder.status = "WAIT ARUCO 5"

            # ------------------------------------------------
            # Draw ROI
            # ------------------------------------------------

            if roi_coords is not None:
                x1, y1, x2, y2 = roi_coords

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2
                )

                # Центр области.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                cv2.drawMarker(
                    frame,
                    (cx, cy),
                    (255, 255, 255),
                    cv2.MARKER_CROSS,
                    12,
                    2
                )

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            y = 30
            dy = 30

            def text(line):
                nonlocal y
                cv2.putText(
                    frame,
                    line,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )
                y += dy

            text(f"FPS: {measured_fps:.1f}")
            text(
                f"ARUCO 5: "
                f"{'FOUND' if marker_visible_now else ('MEMORY' if roi_coords else 'NO')}"
            )
            text(
                f"LASER: {'ON' if laser_visible else 'OFF'} "
                f"RED_PIXELS: {red_pixels} "
                f"RMAX: {peak_red}"
            )
            text(
                f"PULSE: {decoder.last_pulse_frames} frames "
                f"({decoder.last_pulse_seconds:.2f}s) "
                f"BIT: {decoder.last_bit}"
            )
            text(f"DECODER: {decoder.status}")

            if decoder.last_message is not None:
                text(f"MESSAGE: {decoder.last_message}")

            text(
                f"OK: {decoder.valid_packets} "
                f"CRC_ERR: {decoder.crc_errors} "
                f"BAD: {decoder.bad_pulses}"
            )

            recent_bits = "".join(
                str(v)
                for v in list(decoder.bit_history)[-28:]
            )
            text(f"BITS: {recent_bits}")

            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=STREAM_FPS
            )

    finally:
        if camera is not None:
            camera.stop()

        if viewer is not None:
            viewer.close()


# ============================================================
# MAIN
# ============================================================

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

    if pioneer is not None:
        pioneer.close_connection()

    print("Завершено")
