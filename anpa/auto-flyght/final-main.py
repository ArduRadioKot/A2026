import math
import threading
import time

import cv2

from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"
STREAM_FPS = 20
CAMERA_ANGLE = -80

TARGET_MARKER_ID = 5

# Точка после взлёта
ENTRY_POINT = (0.5, 1.5, 1.7, 0.0)

# Точка входа в область поиска
SEARCH_ENTRY_POINT = (0.0, 1.0, 1.7, 0.0)

# Змейка поиска
X_START = 1.5
X_END = -1.5
X_STEP = -0.4
Y_MIN = 0.5
Y_MAX = 4.0
SEARCH_HEIGHT = 1.8
YAW = 0.0

# Высота перелёта к посадочной зоне
LANDING_APPROACH_HEIGHT = 1.7
LANDING_YAW = 0.0

# Нормальная посадочная зона после завершения поиска
LANDING_POINT_NUMBER = 1

# При любой необработанной ошибке дрон летит в зону №7
PANIC_LANDING_POINT_NUMBER = 7

# Координаты в сантиметрах относительно стартовой точки
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

FLIGHT_SPEED = 0.5
MIN_FLIGHT_TIME = 1
LOOP_DELAY = 0.05

# Зависание над найденной ArUco-меткой — 5 секунд.
ARUCO_HOVER_TIME = 3

# Зависание над посадочной зоной перед посадкой — 5 секунд.
LANDING_HOVER_TIME = 3


# ============================================================
# ОБЩЕЕ СОСТОЯНИЕ
# ============================================================

running = True
marker_lock = threading.Lock()
target_marker_visible = False

pioneer = None
video_thread = None


# ============================================================
# ARUCO
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()
aruco_parameters.detectInvertedMarker = True
aruco_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters,
)


def marker_is_visible():
    with marker_lock:
        return target_marker_visible


# ============================================================
# ВИДЕОПОТОК
# ============================================================


def video_worker():
    """Камера, браузерный стрим и распознавание ArUco."""
    global running
    global target_marker_visible

    camera = None
    viewer = None
    frame_counter = 0

    try:
        print("[VIDEO] Создание Camera()...")
        camera = Camera()

        print("[VIDEO] Создание ImageViewer()...")
        viewer = ImageViewer()

        print()
        print("=" * 64)
        print("СТРИМ КАМЕРЫ ЗАПУЩЕН")
        print(f"Откройте: http://10.42.0.1:8889/{STREAM_NAME}")
        print(f"Целевая ArUco: ID {TARGET_MARKER_ID}")
        print("=" * 64)
        print()

        while running:
            try:
                frame = camera.get_cv_frame(timeout=2)
            except Exception as error:
                print("[VIDEO] Ошибка получения кадра:", repr(error))
                time.sleep(0.2)
                continue

            if frame is None:
                time.sleep(0.02)
                continue

            frame_counter += 1
            frame_h, frame_w = frame.shape[:2]

            # Один лёгкий проход, чтобы обработка не блокировала стрим.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco_detector.detectMarkers(gray)

            current_target_visible = False
            detected_count = 0

            if ids is not None and len(ids) > 0:
                detected_count = len(ids)
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for marker_corners, marker_id in zip(corners, ids.flatten()):
                    marker_id = int(marker_id)
                    points = marker_corners.reshape(4, 2).astype(int)
                    center_x = int(points[:, 0].mean())
                    center_y = int(points[:, 1].mean())

                    if marker_id == TARGET_MARKER_ID:
                        current_target_visible = True
                        label = f"TARGET ID {marker_id}"
                        text_color = (0, 0, 255)
                    else:
                        label = f"ID {marker_id}"
                        text_color = (0, 255, 255)

                    cv2.circle(
                        frame,
                        (center_x, center_y),
                        6,
                        text_color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (center_x + 10, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        text_color,
                        2,
                    )

            with marker_lock:
                target_marker_visible = current_target_visible

            cv2.drawMarker(
                frame,
                (frame_w // 2, frame_h // 2),
                (255, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=30,
                thickness=2,
            )

            cv2.rectangle(frame, (10, 10), (480, 125), (0, 0, 0), -1)

            cv2.putText(
                frame,
                "PIONEER CAMERA + FLIGHT",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"FRAME: {frame_counter}  ARUCO: {detected_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                (
                    f"TARGET {TARGET_MARKER_ID}: FOUND"
                    if current_target_visible
                    else f"TARGET {TARGET_MARKER_ID}: ---"
                ),
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255) if current_target_visible else (0, 0, 255),
                2,
            )

            try:
                viewer.imshow(STREAM_NAME, frame, fps=STREAM_FPS)
            except Exception as error:
                print("[VIDEO] Ошибка ImageViewer.imshow:", repr(error))
                time.sleep(0.2)

    except Exception as error:
        print("[VIDEO ERROR]", repr(error))

    finally:
        print("[VIDEO] Остановка видеопотока")

        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                print("[VIDEO] camera.stop error:", repr(error))

        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:
                print("[VIDEO] viewer.close error:", repr(error))


# ============================================================
# ПОЛЁТНЫЕ ФУНКЦИИ
# ============================================================


def wait_point():
    while not pioneer.point_reached():
        time.sleep(LOOP_DELAY)


def calculate_flight_time(x, y, z, speed=FLIGHT_SPEED):
    if speed <= 0:
        raise ValueError("FLIGHT_SPEED должен быть больше 0")

    position = pioneer.get_local_position_lps()
    if position is None or len(position) < 3:
        return MIN_FLIGHT_TIME

    dx = x - position[0]
    dy = y - position[1]
    dz = z - position[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    return max(MIN_FLIGHT_TIME, math.ceil(distance / speed))


def go_to_point_with_speed(x, y, z, yaw=0.0, speed=FLIGHT_SPEED):
    flight_time = calculate_flight_time(x, y, z, speed)

    print(
        f"[FLIGHT] -> X={x:.2f}, Y={y:.2f}, Z={z:.2f}, "
        f"speed≈{speed:.2f} m/s, time={flight_time}s"
    )

    pioneer.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        time=flight_time,
    )


def stop_at_current_position():
    position = pioneer.get_local_position_lps()
    if position is None or len(position) < 3:
        return

    pioneer.go_to_local_point(
        x=position[0],
        y=position[1],
        z=position[2],
        yaw=YAW,
        time=MIN_FLIGHT_TIME,
    )
    time.sleep(0.2)


def hover_at_current_position(duration, log_duration):
    """Удерживает текущую позицию в течение указанного времени."""
    position = pioneer.get_local_position_lps()
    if position is None or len(position) < 3:
        print("[HOVER] Не удалось получить текущую позицию")
        time.sleep(duration)
        return

    print(f"[HOVER] Зависание на {log_duration} секунд")

    pioneer.go_to_local_point(
        x=position[0],
        y=position[1],
        z=position[2],
        yaw=YAW,
        time=MIN_FLIGHT_TIME,
    )

    time.sleep(duration)
    print("[HOVER] Зависание завершено")


def generate_x_columns():
    if X_STEP == 0:
        raise ValueError("X_STEP не может быть 0")

    if X_START > X_END and X_STEP > 0:
        raise ValueError("X_STEP должен быть отрицательным")

    if X_START < X_END and X_STEP < 0:
        raise ValueError("X_STEP должен быть положительным")

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

    if not columns or abs(columns[-1] - X_END) > 0.001:
        columns.append(X_END)

    return columns


def fly_and_search(x, y, z, yaw):
    """Летит к точке и сразу останавливает маршрут при обнаружении ArUco."""
    go_to_point_with_speed(x, y, z, yaw, FLIGHT_SPEED)

    while True:
        if marker_is_visible():
            print(f"[ARUCO] Найден целевой ID {TARGET_MARKER_ID}")
            stop_at_current_position()

            # После обнаружения ArUco зависаем над меткой 5 секунд.
            hover_at_current_position(ARUCO_HOVER_TIME, 5)
            return True

        if pioneer.point_reached():
            return marker_is_visible()

        time.sleep(LOOP_DELAY)


def get_landing_point(number):
    if number not in LANDING_POINTS_CM:
        raise ValueError(f"Нет посадочной зоны №{number}")

    x_cm, y_cm = LANDING_POINTS_CM[number]
    return x_cm / 100.0, y_cm / 100.0


def fly_to_landing_zone_and_land(number):
    """Подлёт по координатам и немедленная посадка без центровки и зависания."""
    landing_x, landing_y = get_landing_point(number)

    print(
        f"[LANDING] Полёт в зону №{number}: "
        f"X={landing_x:.2f}, Y={landing_y:.2f}"
    )

    go_to_point_with_speed(
        landing_x,
        landing_y,
        LANDING_APPROACH_HEIGHT,
        LANDING_YAW,
        FLIGHT_SPEED,
    )
    wait_point()

    print(f"[LANDING] Зона №{number} достигнута")

    # Перед посадкой зависаем над посадочной зоной 5 секунд.
    hover_at_current_position(LANDING_HOVER_TIME, 5)

    print("[LAND] Начало посадки после зависания")
    pioneer.land()
    wait_point()
    print("[LAND] Посадка завершена")


def land_here():
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


# ============================================================
# ОСНОВНОЙ МАРШРУТ
# ============================================================


def run_flight():
    print("[FLIGHT] ARM")
    pioneer.arm()

    print("[FLIGHT] Взлёт")
    pioneer.takeoff()
    print("[FLIGHT] Команда взлёта отправлена")

    go_to_point_with_speed(*ENTRY_POINT, speed=FLIGHT_SPEED)
    wait_point()

    go_to_point_with_speed(*SEARCH_ENTRY_POINT, speed=FLIGHT_SPEED)
    wait_point()

    x_columns = generate_x_columns()
    print("[SEARCH] Полосы X:", x_columns)

    target_found = False

    for column_index, x in enumerate(x_columns):
        if column_index % 2 == 0:
            y_start, y_end = Y_MIN, Y_MAX
        else:
            y_start, y_end = Y_MAX, Y_MIN

        target_found = fly_and_search(x, y_start, SEARCH_HEIGHT, YAW)
        if target_found:
            break

        target_found = fly_and_search(x, y_end, SEARCH_HEIGHT, YAW)
        if target_found:
            break

    if target_found:
        print(
            "[SEARCH] ArUco найден. Зависание над меткой завершено. "
            "Перелёт в посадочную зону №7."
        )
        fly_to_landing_zone_and_land(7)
    else:
        print(
            f"[SEARCH] ArUco ID {TARGET_MARKER_ID} не найден "
            "за один проход. Перелёт в посадочную зону №1."
        )
        fly_to_landing_zone_and_land(LANDING_POINT_NUMBER)


# ============================================================
# MAIN
# ============================================================


try:
    # В рабочем варианте Pioneer создаётся до Camera/ImageViewer.
    print("[PIONEER] Подключение...")
    pioneer = Pioneer()
    print("[PIONEER] Подключение создано")

    print("[CAMERA] Установка угла...")
    servo_camera = ServoCamera()

    if servo_camera.set_angle(CAMERA_ANGLE):
        print(f"[CAMERA] Угол установлен: {CAMERA_ANGLE}")
    else:
        print("[CAMERA] Не удалось установить угол")

    video_thread = threading.Thread(
        target=video_worker,
        daemon=False,
    )
    video_thread.start()

    # Даём камере создать поток до начала полёта.
    time.sleep(1.0)

    run_flight()

except KeyboardInterrupt:
    print("\n[STOP] Остановка оператором")
    print("[SAFETY] Посадка в текущем месте")

    if pioneer is not None:
        try:
            land_here()
        except Exception as error:
            print("[LAND ERROR]", repr(error))

except Exception as error:
    print()
    print("[ERROR]", repr(error))
    print(
        f"[PANIC] Попытка перелёта в посадочную зону "
        f"№{PANIC_LANDING_POINT_NUMBER}"
    )

    if pioneer is not None:
        try:
            fly_to_landing_zone_and_land(PANIC_LANDING_POINT_NUMBER)
        except Exception as panic_error:
            print("[PANIC ERROR]", repr(panic_error))
            print("[SAFETY] Перелёт в зону №7 не удался. Посадка на месте.")

            try:
                land_here()
            except Exception:
                try:
                    pioneer.land()
                except Exception:
                    pass

finally:
    running = False

    if video_thread is not None and video_thread.is_alive():
        video_thread.join(timeout=3.0)

    if pioneer is not None:
        try:
            pioneer.close_connection()
        except Exception as error:
            print("[PIONEER] close_connection error:", repr(error))

    print("[END] Программа завершена")
