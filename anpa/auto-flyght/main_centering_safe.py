import time
import math
import threading
import cv2

from pioneer_sdk2 import Pioneer
from pioneer_sdk2 import Camera
from pioneer_sdk2 import ImageViewer
from pioneer_sdk2 import ServoCamera


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

HOVER_TIME = 5.0

# Частота основных циклов
LOOP_DELAY = 0.2

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
# ЦЕНТРИРОВАНИЕ ПО ARUCO
# ============================================================

# Допустимая погрешность центрирования, пикселей
CENTER_TOLERANCE_PX = 50

# P-регулятор: скорость = ошибка_в_пикселях * CENTER_KP
CENTER_KP = 0.0010

# Ограничения скорости коррекции, м/с
CENTER_MIN_SPEED = 0.04
CENTER_MAX_SPEED = 0.16

# Сглаживание координат ArUco:
# 0.0 = сильное сглаживание, 1.0 = без сглаживания
CENTER_FILTER_ALPHA = 0.30

# Время удержания маркера в центре
CENTER_STABLE_TIME = 1.5

# Допустимое время временной потери маркера
MARKER_LOST_TIMEOUT = 2.0

# Максимальное время центрирования
CENTER_TIMEOUT = 25.0


# ============================================================
# ОБЩЕЕ СОСТОЯНИЕ
# ============================================================

running = True

marker_lock = threading.Lock()

target_marker_visible = False
target_marker_center = None
target_frame_size = None

camera = None
viewer = None


# ============================================================
# ARUCO
# ============================================================

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

aruco_parameters = cv2.aruco.DetectorParameters()

aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary,
    aruco_parameters
)


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

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            corners, ids, rejected = aruco_detector.detectMarkers(gray)

            current_target_visible = False
            current_target_center = None

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

            # Информация поверх картинки
            cv2.rectangle(
                frame,
                (10, 10),
                (410, 105),
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

    if not pos or len(pos) < 3:
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

    if not pos or len(pos) < 3:
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

    Чем ближе маркер к центру кадра, тем меньше скорость.
    Координаты маркера дополнительно сглаживаются, чтобы
    уменьшить дрожание от шума детектора.
    """

    print("[CENTER] Начинаю плавное центрирование")

    stop_at_current_position()

    started = time.time()
    stable_since = None
    marker_last_seen = time.time()

    filtered_x = None
    filtered_y = None

    def clamp(value, low, high):
        return max(low, min(high, value))

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

        if now - started > CENTER_TIMEOUT:
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

            pioneer.set_manual_speed_body_fixed(
                vx=0.0,
                vy=0.0,
                vz=0.0,
                yaw_rate=0.0
            )

            if now - marker_last_seen > MARKER_LOST_TIMEOUT:
                raise CenteringError(
                    "ArUco потерян во время центрирования"
                )

            time.sleep(0.05)
            continue

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
                return

            time.sleep(0.05)
            continue

        stable_since = None

        # По горизонтали кадра: маркер справа -> летим вправо.
        vx = speed_from_error(error_x)

        # По вертикали кадра оставляем знак как в исходной версии:
        # маркер ниже центра -> движение назад.
        vy = -speed_from_error(error_y)

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
    wait_point()

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
        return_home_and_land()

    else:
        # ----------------------------------------------------
        # 5. Маркер найден — центрирование
        # ----------------------------------------------------

        print("[SEARCH] Маршрут поиска остановлен")

        try:
            center_over_target()

        except CenteringError as e:
            print()
            print("[CENTER ERROR]", e)
            print("[SAFETY] Центрирование не удалось. Возврат на старт.")
            return_home_and_land()

        else:
            # ------------------------------------------------
            # 6. Зависание
            # ------------------------------------------------

            print(
                f"[HOVER] Зависание над меткой "
                f"{HOVER_TIME:.1f} секунд"
            )

            hover_pos = pioneer.get_local_position_lps()

            if hover_pos:
                pioneer.go_to_local_point(
                    x=hover_pos[0],
                    y=hover_pos[1],
                    z=SEARCH_HEIGHT,
                    yaw=YAW,
                    time=MIN_FLIGHT_TIME
                )

            time.sleep(HOVER_TIME)

            # --------------------------------------------
            # 7. Домой
            # --------------------------------------------

            return_home_and_land()


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

    running = False

    if video_thread is not None:
        video_thread.join(timeout=3.0)

    try:
        pioneer.close_connection()
    except Exception:
        pass

    print("[END] Программа завершена")
