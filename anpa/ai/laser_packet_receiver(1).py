import time
import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Формат пакета: преамбула 8 бит + сообщение 4 бита + CRC4 = 16 бит
# Тайминги согласованы с передатчиком (laser_packet_drone.py):
#   бит '0' -> вспышка 0.3 с (dot)
#   бит '1' -> вспышка 0.9 с (dash)
#   пауза между сигналами -> 0.2 с
# ---------------------------------------------------------------------------

PREAMBLE = 0b10101010   # 8 бит, фиксированный синхросигнал
PREAMBLE_LEN = 8
MSG_LEN = 4
CRC_LEN = 4
PACKET_LEN = PREAMBLE_LEN + MSG_LEN + CRC_LEN   # 16 бит

DOT_DURATION = 0.3     # длительность бита '0', сек
DASH_DURATION = 0.9    # длительность бита '1', сек
GAP_DURATION = 0.2     # пауза между сигналами, сек

MIN_FLASH_S = DOT_DURATION * 0.5          # короче — считаем шумом
SPLIT_S = (DOT_DURATION + DASH_DURATION) / 2.0   # порог 0/1 по длительности
START_GAP_S = GAP_DURATION * 3 * 0.7      # порог распознавания стартовой паузы


def crc4(data: int, length_bits: int, poly: int = 0b0011) -> int:
    """CRC4, полином x^4 + x + 1, обработка данных MSB-first."""
    crc = 0

    for bit_index in range(length_bits - 1, -1, -1):
        data_bit = (data >> bit_index) & 1
        feedback = ((crc >> 3) & 1) ^ data_bit
        crc = (crc << 1) & 0x0F
        if feedback:
            crc ^= poly

    return crc


def parse_packet(bits: list):
    """Проверяет преамбулу и CRC4, возвращает (message, ok)."""
    if len(bits) != PACKET_LEN:
        return None, False

    value = 0
    for b in bits:
        value = (value << 1) | b

    crc_recv = value & ((1 << CRC_LEN) - 1)
    message = (value >> CRC_LEN) & ((1 << MSG_LEN) - 1)
    preamble_recv = (value >> (MSG_LEN + CRC_LEN)) & ((1 << PREAMBLE_LEN) - 1)

    if preamble_recv != PREAMBLE:
        return None, False

    if crc4(message, MSG_LEN) != crc_recv:
        return None, False

    return message, True


class CameraReceiver:
    """
    Следит за яркостью ROI. По длительности "светлых" интервалов
    восстанавливает биты, накапливает их в пакет, проверяет
    преамбулу и CRC4.
    """

    def __init__(self, camera_index=0, roi=None, threshold=180, show_window=False):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError('Не удалось открыть камеру')

        self.roi = roi              # (x, y, w, h) или None -> центр кадра
        self.threshold = threshold
        self.show_window = show_window

        self._is_on = False
        self._on_start = None
        self._last_off_time = time.time()
        self._bits = []

    def _get_roi_coords(self, frame):
        if self.roi is not None:
            return self.roi
        h, w = frame.shape[:2]
        size = min(h, w) // 6
        cx, cy = w // 2, h // 2
        return (cx - size, cy - size, size * 2, size * 2)

    def _brightness(self, frame):
        x, y, w, h = self._get_roi_coords(frame)
        roi = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _reset_packet(self):
        self._bits = []

    def _handle_edge_off(self, duration_s):
        """Вызывается, когда вспышка закончилась (переход светло -> темно)."""
        if duration_s < MIN_FLASH_S:
            return   # шум

        bit = 1 if duration_s >= SPLIT_S else 0
        self._bits.append(bit)

        if len(self._bits) == PACKET_LEN:
            message, ok = parse_packet(self._bits)
            self._reset_packet()
            return message, ok
        return None

    def receive_packet(self, timeout_s=30.0):
        """
        Блокирующе ждёт один валидный пакет.
        Возвращает (message, ok). ok=False при таймауте или ошибке CRC/преамбулы.
        """
        self._reset_packet()
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            ret, frame = self.cap.read()
            if not ret:
                continue

            now = time.time()
            brightness = self._brightness(frame)
            bright = brightness > self.threshold

            if self.show_window:
                x, y, w, h = self._get_roi_coords(frame)
                color = (0, 255, 0) if bright else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.imshow('receiver', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if bright and not self._is_on:
                # начало вспышки — проверяем, не была ли пауза перед ней
                # достаточно длинной, чтобы считать это стартом нового пакета
                gap = now - self._last_off_time
                if gap >= START_GAP_S and self._bits:
                    # длинная пауза посреди пакета — считаем его битым, начинаем заново
                    self._reset_packet()

                self._is_on = True
                self._on_start = now

            elif not bright and self._is_on:
                self._is_on = False
                self._last_off_time = now
                duration_s = now - self._on_start

                result = self._handle_edge_off(duration_s)
                if result is not None:
                    return result

        return None, False

    def close(self):
        self.cap.release()
        if self.show_window:
            cv2.destroyAllWindows()


def main():
    receiver = CameraReceiver(camera_index=0, threshold=180, show_window=True)
    try:
        print('Ожидание пакета...')
        while True:
            message, ok = receiver.receive_packet(timeout_s=30.0)
            if ok:
                print(f'Принято сообщение: {message}')
            else:
                print('Пакет не принят: таймаут либо ошибка преамбулы/CRC')
    except KeyboardInterrupt:
        pass
    finally:
        receiver.close()


if __name__ == '__main__':
    main()
