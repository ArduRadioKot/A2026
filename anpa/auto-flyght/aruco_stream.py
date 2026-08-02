import time

import cv2
import numpy as np

from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"
TARGET_MARKER_ID = 5

CAMERA_ANGLE = -80
VIDEO_FPS = 20

# Сколько секунд хранить последнее корректное положение метки,
# если сетка временно закрыла её на нескольких кадрах.
ARUCO_DETECTION_MEMORY = 0.45

# Масштаб дополнительных проходов распознавания.
ARUCO_UPSCALE = 1.6

# Использовать дополнительную обработку для подавления линий сетки.
ARUCO_ENABLE_NET_SUPPRESSION = True


# ============================================================
# НАСТРОЙКА ARUCO
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_parameters.adaptiveThreshWinSizeMin = 3
aruco_parameters.adaptiveThreshWinSizeMax = 61
aruco_parameters.adaptiveThreshWinSizeStep = 4
aruco_parameters.adaptiveThreshConstant = 7

aruco_parameters.minMarkerPerimeterRate = 0.012
aruco_parameters.maxMarkerPerimeterRate = 4.0
aruco_parameters.polygonalApproxAccuracyRate = 0.07
aruco_parameters.minCornerDistanceRate = 0.025
aruco_parameters.minDistanceToBorder = 2

aruco_parameters.cornerRefinementMethod = (
    cv2.aruco.CORNER_REFINE_SUBPIX
)
aruco_parameters.cornerRefinementWinSize = 7
aruco_parameters.cornerRefinementMaxIterations = 50
aruco_parameters.cornerRefinementMinAccuracy = 0.01

aruco_parameters.perspectiveRemovePixelPerCell = 10
aruco_parameters.perspectiveRemoveIgnoredMarginPerCell = 0.15

aruco_parameters.maxErroneousBitsInBorderRate = 0.50
aruco_parameters.errorCorrectionRate = 0.85
aruco_parameters.detectInvertedMarker = True

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


# ============================================================
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# ============================================================

def suppress_net_lines(gray):
    """
    Создаёт дополнительный вариант изображения с ослабленными
    горизонтальными и вертикальными линиями сетки.
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


def select_target(corners, ids, scale):
    """
    Выбирает из найденных меток только TARGET_MARKER_ID.
    Возвращает координаты в масштабе исходного кадра.
    """
    if ids is None or len(ids) == 0:
        return None

    flat_ids = ids.flatten().astype(int)
    indexes = np.where(flat_ids == TARGET_MARKER_ID)[0]

    if len(indexes) == 0:
        return None

    index = int(indexes[0])
    marker_corners = corners[index].astype(np.float32)

    if scale != 1.0:
        marker_corners = marker_corners / scale

    return marker_corners


def detect_target_aruco_robust(frame):
    """
    Ищет TARGET_MARKER_ID по нескольким вариантам изображения.

    Возвращает:
        marker_corners или None,
        название успешного режима,
        маску сетки.
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

        marker_corners = select_target(
            corners,
            ids,
            scale
        )

        if marker_corners is not None:
            return marker_corners, mode, net_mask

    return None, "NONE", net_mask


def draw_target(frame, marker_corners, from_memory=False):
    """
    Рисует рамку, центр и подпись целевой ArUco.
    """
    points = (
        marker_corners
        .reshape(4, 2)
        .astype(int)
    )

    top_left = points[0]
    top_right = points[1]
    bottom_right = points[2]
    bottom_left = points[3]

    outline_color = (
        (0, 180, 255)
        if from_memory
        else (0, 255, 0)
    )

    cv2.line(
        frame,
        tuple(top_left),
        tuple(top_right),
        outline_color,
        3
    )

    cv2.line(
        frame,
        tuple(top_right),
        tuple(bottom_right),
        outline_color,
        3
    )

    cv2.line(
        frame,
        tuple(bottom_right),
        tuple(bottom_left),
        outline_color,
        3
    )

    cv2.line(
        frame,
        tuple(bottom_left),
        tuple(top_left),
        outline_color,
        3
    )

    center_x = int(np.mean(points[:, 0]))
    center_y = int(np.mean(points[:, 1]))

    cv2.circle(
        frame,
        (center_x, center_y),
        6,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        f"ARUCO ID: {TARGET_MARKER_ID}",
        (
            int(top_left[0]),
            max(25, int(top_left[1]) - 12)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        outline_color,
        2
    )

    cv2.putText(
        frame,
        (
            "TARGET MEMORY"
            if from_memory
            else "TARGET FOUND"
        ),
        (
            center_x + 12,
            center_y - 12
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (
            (0, 180, 255)
            if from_memory
            else (0, 0, 255)
        ),
        2
    )

    return center_x, center_y


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

camera = None
viewer = None

last_detection_time = -10_000.0
last_marker_corners = None
last_detection_mode = "NONE"

try:
    servo_camera = ServoCamera()

    if servo_camera.set_angle(CAMERA_ANGLE):
        print(
            f"[CAMERA] Угол камеры установлен: "
            f"{CAMERA_ANGLE} градусов"
        )
    else:
        print(
            "[CAMERA] Не удалось установить угол камеры"
        )

    print("[VIDEO] Подключение камеры...")

    camera = Camera()
    viewer = ImageViewer()

    print()
    print("=" * 60)
    print("ТЕСТ ARUCO БЕЗ ПОЛЁТА")
    print(
        f"Открой в браузере: "
        f"http://10.42.0.1:8889/{STREAM_NAME}"
    )
    print(
        f"Ищется ArUco ID: {TARGET_MARKER_ID}"
    )
    print("Остановка: Ctrl+C")
    print("=" * 60)
    print()

    while True:
        frame = camera.get_cv_frame(timeout=2.0)

        if frame is None:
            time.sleep(0.02)
            continue

        frame_h, frame_w = frame.shape[:2]

        marker_corners, detection_mode, net_mask = (
            detect_target_aruco_robust(frame)
        )

        target_visible = False
        from_memory = False

        if marker_corners is not None:
            target_visible = True

            last_detection_time = time.monotonic()
            last_marker_corners = marker_corners.copy()
            last_detection_mode = detection_mode

        else:
            detection_age = (
                time.monotonic()
                - last_detection_time
            )

            if (
                last_marker_corners is not None
                and detection_age <= ARUCO_DETECTION_MEMORY
            ):
                marker_corners = last_marker_corners.copy()
                detection_mode = (
                    f"MEM:{last_detection_mode}"
                )
                target_visible = True
                from_memory = True

        if target_visible:
            center_x, center_y = draw_target(
                frame,
                marker_corners,
                from_memory=from_memory
            )
        else:
            center_x = None
            center_y = None

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

        cv2.rectangle(
            frame,
            (10, 10),
            (520, 145),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            "ARUCO VIDEO TEST",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"SEARCH ID: {TARGET_MARKER_ID}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            (
                "FOUND"
                if target_visible
                else "NOT FOUND"
            ),
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (
                (0, 255, 255)
                if target_visible
                else (0, 0, 255)
            ),
            2
        )

        cv2.putText(
            frame,
            f"DETECT: {detection_mode}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (
                (0, 180, 255)
                if from_memory
                else (255, 255, 255)
            ),
            2
        )

        if center_x is not None and center_y is not None:
            cv2.putText(
                frame,
                f"CENTER: {center_x}, {center_y}",
                (275, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        viewer.imshow(
            STREAM_NAME,
            frame,
            fps=VIDEO_FPS
        )

except KeyboardInterrupt:
    print("\n[STOP] Остановка оператором")

except Exception as error:
    print("[ERROR]", repr(error))

finally:
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

    print("[END] Видеотест завершён")
