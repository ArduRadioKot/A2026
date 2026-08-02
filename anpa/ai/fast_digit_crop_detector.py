import argparse
import time
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# Настройки детекции контура
# ============================================================

LOWER_HSV = np.array([15, 80, 80], dtype=np.uint8)
UPPER_HSV = np.array([40, 255, 255], dtype=np.uint8)

C_EPS = 15

IMG_H = 240
IMG_W = 320

# В исходном коде координаты были записаны наоборот относительно
# стандартного порядка width/height. Оставлены без изменения.
IMG_CX = 120
IMG_CY = 160

MODEL_INPUT_SIZE = (64, 64)
IMG_CNT = 10

P = 0.01

CONFIDENCE_THRESHOLD = 0.0
BN_EPS = 1e-5

# Настройки подготовки цифры
DIGIT_PADDING = 8
MIN_DIGIT_AREA = 8

# Коррекция похожих классов 1 / 3 / 7.
# Эти коэффициенты можно подстроить под вашу камеру.
ENABLE_137_CORRECTION = False
ONE_MAX_ASPECT = 0.38
SEVEN_MAX_ASPECT = 0.78
ONE_MIN_RELATIVE_PROB = 0.22
SEVEN_MIN_RELATIVE_PROB = 0.30


# ============================================================
# Быстрая модель NumPy
# ============================================================

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
# Ваши функции поиска и crop
# ============================================================

def hsv_max_contour(img, lower, upper, min_area=10):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv_img,
        lowerb=lower,
        upperb=upper,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    max_area = 0
    max_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area and area > max_area:
            max_area = area
            max_contour = contour

    return max_contour, mask


def find_contour_mid(contour):
    moments = cv2.moments(contour)

    cx = None
    cy = None

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]

    return cx, cy


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
# Основной цикл
# ============================================================

def run(weights_dir, camera_index=0, show_mask=False):
    digit_model = FastDigitModel(weights_dir)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"Не удалось открыть камеру с индексом {camera_index}."
        )

    # Низкое разрешение ускоряет поиск HSV-контура.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    img_cnt = 0
    last_cls = None
    data = None

    fps = 0.0
    previous_time = time.perf_counter()

    while True:
        ret, img = cap.read()

        if not ret:
            continue

        max_contour, mask = hsv_max_contour(
            img,
            lower=LOWER_HSV,
            upper=UPPER_HSV,
        )

        current_cls = None
        confidence = 0.0

        if max_contour is not None:
            cx, cy = find_contour_mid(max_contour)

            if cx is not None and cy is not None:
                if (
                    (cx - IMG_CX) ** 2
                    + (cy - IMG_CY) ** 2
                    >= C_EPS ** 2
                ):
                    dx = -P * (cx - IMG_CX)
                    dy = -P * (cy - IMG_CY)

                    # Здесь можно отправить скорость приводам:
                    # set_ang_vel(dx, dy)

            crop_img, (x, y, w, h) = crop(img, max_contour)

            # Именно crop_img передаётся в нейросеть.
            current_cls = digit_model.predict(crop_img)
            confidence = digit_model.last_confidence

            if confidence >= CONFIDENCE_THRESHOLD:
                if last_cls == current_cls:
                    img_cnt += 1

                    if img_cnt > IMG_CNT:
                        data = current_cls
                else:
                    last_cls = current_cls
                    img_cnt = 1
                    data = None

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2,
            )

            cv2.circle(
                img,
                center=(int(cx), int(cy)),
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )

            cv2.putText(
                img,
                f"class: {current_cls}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                img,
                f"confidence: {confidence * 100:.1f}%",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            probabilities = digit_model.last_probabilities
            if probabilities is not None:
                cv2.putText(
                    img,
                    (
                        f"p1:{probabilities[1]:.2f} "
                        f"p3:{probabilities[3]:.2f} "
                        f"p7:{probabilities[7]:.2f}"
                    ),
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            if data is not None:
                cv2.putText(
                    img,
                    f"confirmed: {data}",
                    (10, 108),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("crop sent to model", crop_img)
        else:
            img_cnt = 0
            last_cls = None
            data = None

        now = time.perf_counter()
        frame_time = max(now - previous_time, 1e-6)
        previous_time = now

        instant_fps = 1.0 / frame_time

        if fps == 0.0:
            fps = instant_fps
        else:
            fps = fps * 0.9 + instant_fps * 0.1

        cv2.putText(
            img,
            f"FPS: {fps:.1f}",
            (10, img.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("camera", img)

        if show_mask:
            cv2.imshow("HSV mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Быстрое распознавание цифр из HSV crop "
            "через NumPy/OpenCV."
        )
    )

    parser.add_argument(
        "--weights",
        default="digits64_numpy",
        help="Папка с файлами .npy.",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Индекс камеры.",
    )

    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Показывать HSV-маску.",
    )

    args = parser.parse_args()

    # OpenCV может использовать несколько CPU-потоков.
    cv2.setUseOptimized(True)

    run(
        weights_dir=args.weights,
        camera_index=args.camera,
        show_mask=args.show_mask,
    )


if __name__ == "__main__":
    main()