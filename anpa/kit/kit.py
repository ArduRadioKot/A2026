#!/usr/bin/env python3

import time
import cv2
import numpy as np
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from user.library import DroneLibrary


# ============================================================
# НАСТРОЙКИ ЦВЕТА
# ============================================================
LOWER_HSV = np.array([15, 80, 80])
UPPER_HSV = np.array([40, 255, 255])


# ============================================================
# КАМЕРА 320x240
# ============================================================
IMG_W = 320
IMG_H = 240
IMG_CX = IMG_W // 2       # 160
IMG_CY = IMG_H // 2       # 120


# ============================================================
# ПОИСК
# ============================================================
SEARCH_DEPTH = 0.35
SEARCH_PITCH = -15
SEARCH_SPEED = 25
LANES = 5
LONG_TIME = 12.0
SHIFT_TIME = 2.0


# ============================================================
# ДЕТЕКТОР
# ============================================================
MIN_AREA = 40
STOP_BOX_SIZE = 70
LOCK_FRAMES = 3


# ============================================================
# РЕГУЛЯТОР
# ============================================================
YAW_KP = 20.0
MAX_YAW_CORRECTION = 15.0


# ============================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
# ============================================================
bridge = CvBridge()
latest_frame = None


# ============================================================
# ФУНКЦИИ ЛАЗЕРА (КОД МОРЗЕ)
# ============================================================

def blink_laser(drone, duration):
    """Включает лазер на указанное время и выключает"""
    drone.set_laser(1)
    time.sleep(duration)
    drone.set_laser(0)

def send_dot(drone):
    """Точка (короткий сигнал)"""
    blink_laser(drone, 0.3)
    time.sleep(0.2)

def send_dash(drone):
    """Тире (длинный сигнал)"""
    blink_laser(drone, 0.9)
    time.sleep(0.2)

def send_number(drone, number):
    """Отправляет цифру через лазер (код Морзе)"""
    morse_numbers = {
        '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
        '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.'
    }
    
    rospy.loginfo(f'Отправка цифры {number} лазером...')
    code = morse_numbers.get(str(number), '')
    
    for symbol in code:
        if rospy.is_shutdown():
            break
        if symbol == '.':
            send_dot(drone)
        elif symbol == '-':
            send_dash(drone)
        time.sleep(0.3)
    
    time.sleep(1)


# ============================================================
# ROS CAMERA
# ============================================================
def camera_callback(msg):
    global latest_frame
    try:
        latest_frame = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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

    best = None
    best_area = 0

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
        "cx": x + w / 2, "cy": y + h / 2, "area": best_area,
        "x": x, "y": y, "w": w, "h": h,
    }


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================
def normalize_course(course):
    while course >= 360: course -= 360
    while course < 0: course += 360
    return course

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))

def wait_lock(seconds=0.4):
    start = time.time()
    found_frames = 0
    last_detection = None
    while time.time() - start < seconds:
        detection = detect_cube(latest_frame)
        if detection is not None:
            found_frames += 1
            last_detection = detection
            if found_frames >= LOCK_FRAMES:
                return last_detection
        else:
            found_frames = 0
        time.sleep(0.05)
    return None


# ============================================================
# ПОИСК И ПРИБЛИЖЕНИЕ
# ============================================================
def search_segment(drone, heading, duration):
    print(f"\n[SEARCH] курс: {heading}, время: {duration}")
    drone.set_course(heading)
    time.sleep(1.0)
    drone.set_speed(SEARCH_SPEED)

    start = time.time()
    detected_count = 0

    while time.time() - start < duration:
        detection = detect_cube(latest_frame)
        if detection is not None:
            detected_count += 1
            if detected_count >= LOCK_FRAMES:
                drone.set_speed(0)
                print("\n================================\n ЦЕЛЬ ОБНАРУЖЕНА\n================================")
                return detection
        else:
            detected_count = 0
        time.sleep(0.08)

    drone.set_speed(0)
    return None


def shift_lane(drone):
    print("[SEARCH] переход на соседнюю полосу")
    drone.set_course(90)
    time.sleep(1)
    drone.set_speed(SEARCH_SPEED)
    
    start = time.time()
    while time.time() - start < SHIFT_TIME:
        detection = detect_cube(latest_frame)
        if detection is not None:
            drone.set_speed(0)
            detection = wait_lock()
            if detection is not None:
                return detection
        time.sleep(0.08)
    drone.set_speed(0)
    return None


def reacquire(drone, base_course):
    print("[VISION] цель потеряна, выполняю сканирование")
    drone.set_speed(0)
    for offset in [-10, -20, -30, -15, 0, 15, 30, 20, 10, 0]:
        course = normalize_course(base_course + offset)
        drone.set_course(course)
        time.sleep(0.4)
        detection = detect_cube(latest_frame)
        if detection is not None:
            detection = wait_lock(0.3)
            if detection is not None:
                print(f"[VISION] цель найдена снова: {course}")
                return detection
    print("[VISION] повторно найти цель не удалось")
    return None


def approach_cube(drone):
    print("\n================================\n ПРИБЛИЖЕНИЕ К КУБИКУ\n================================")
    lost_since = None

    while True:
        detection = detect_cube(latest_frame)

        if detection is None:
            if lost_since is None:
                lost_since = time.time()
            drone.set_speed(0)
            if time.time() - lost_since > 0.7:
                result = reacquire(drone, drone.get_course())
                if result is None:
                    return False
                lost_since = None
            time.sleep(0.05)
            continue

        lost_since = None
        cx, cy = detection["cx"], detection["cy"]
        w, h = detection["w"], detection["h"]

        print(f"[TARGET] cx={round(cx)} cy={round(cy)} size={w}x{h}")

        if max(w, h) >= STOP_BOX_SIZE:
            drone.set_speed(0)
            print("\n################################\n#       КУБИК НАЙДЕН           #\n################################")
            
            # >>> ЗДЕСЬ МЫ МИГАЕМ ЛАЗЕРОМ ПРИ УСПЕХЕ! <<<
            # Можете поменять цифру 1 на любую другую (0-9)
            send_number(drone, 1) 
            
            return True

        error_x = (cx - IMG_CX) / IMG_CX
        correction = clamp(error_x * YAW_KP, -MAX_YAW_CORRECTION, MAX_YAW_CORRECTION)
        drone.set_course(normalize_course(drone.get_course() + correction))

        size = max(w, h)
        speed = 20 if size < 25 else (14 if size < 45 else 8)
        if abs(error_x) > 0.45:
            speed = 0

        drone.set_speed(speed)
        time.sleep(0.08)


def search_pool(drone):
    print("\n================================\n ПОИСК КУБИКА\n================================")
    drone.set_speed(0)
    drone.set_depth(SEARCH_DEPTH)
    drone.set_pitch(SEARCH_PITCH)
    time.sleep(2)

    for lane in range(LANES):
        print(f"\n========== ПОЛОСА {lane + 1} ИЗ {LANES} ==========")
        heading = 0 if lane % 2 == 0 else 180

        detection = search_segment(drone, heading, LONG_TIME)
        if detection is not None and approach_cube(drone):
            return True

        if lane == LANES - 1:
            break

        detection = shift_lane(drone)
        if detection is not None and approach_cube(drone):
            return True

    drone.set_speed(0)
    print("\nКубик не найден.")
    return False


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Запуск поиска кубика...")

    # 1. Инициализация библиотеки (она сама создаст ROS узел)
    drone = DroneLibrary()

    # 2. Подписка на камеру
    rospy.Subscriber("/raspicam_node/image", Image, camera_callback, queue_size=1)

    # 3. Запуск систем
    drone.start()

    try:
        search_pool(drone)
    except KeyboardInterrupt:
        print("\nОстановка пользователем")
    except Exception as exc:
        print(f"ОШИБКА: {type(exc).__name__} {exc}")
    finally:
        print("\nОстановка дрона...")
        drone.set_speed(0)
        try:
            drone.set_pitch(0)
        except Exception:
            pass
        
        time.sleep(0.5)  # Даём дрону время получить команду на остановку
        drone.stop()
        drone.set_offline_mode()
        
        print("Программа завершена")
        
        # 4. КОРРЕКТНОЕ ЗАВЕРШЕНИЕ ROS (убирает ошибки 'NoneType' в конце)
        rospy.signal_shutdown("Mission complete")
        time.sleep(1.0)  # Даём фоновым потокам ROS секунду на закрытие сокетов
