import time
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Сообщения ROS 2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import String, ColorRGBA

# Сообщения PX4 (px4_msgs)
from px4_msgs.msg import VehicleCommand, TrajectorySetpoint, OffboardControlMode

from cv_bridge import CvBridge
import cv2
import numpy as np


class CheburashkaFinderPX4(Node):
    def __init__(self):
        super().__init__('cheburashka_finder_node')

        # --- Константы регламента и поля ---
        self.CELL_SIZE = 0.8        # Размер клетки (80x80 см)
        self.TARGET_ALTITUDE = 1.2  # Высота полёта (м)
        self.TAKEOFF_ALT = -1.2     # PX4 NED Z = -1.2 м

        # Состояния автомата
        self.state = "INIT"

        # OpenCV
        self.bridge = CvBridge()
        self.latest_image = None

        # Профиль QoS для PX4
        qos_profile_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- ПОДПИСЧИКИ И ИЗДАТЕЛИ ---
        self.pose_sub = self.create_subscription(
            PoseStamped, '/aruco/world_pose', self.pose_cb, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera_1/image_raw', self.image_cb, 10)

        self.cmd_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', 10)
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', 10)
        self.trajectory_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10)

        self.led_pub = self.create_publisher(ColorRGBA, '/led_control/state', 10)
        self.rover_target_pub = self.create_publisher(Point, '/sverk/rover_target_coords', 10)
        self.log_pub = self.create_publisher(String, '/sverk/logs', 10)

        # Переменные позиционирования (Старт из центра A6)
        self.current_pose = PoseStamped()
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = self.TAKEOFF_ALT

        # Центры ячеек относительно центра A6 (0.0, 0.0):
        # A1: центр ячейки A1 (-4.0, 0.0)
        # E2: центр ячейки E2 (-3.2, -3.2)
        self.search_waypoints = [
            {"name": "Центр A1", "x": -4.0, "y": 0.0},
            {"name": "Центр E2", "x": -3.2, "y": -3.2}
        ]
        self.current_waypoint_idx = 0

        # Пороги HSV для коричневого цвета
        self.lower_brown = np.array([5, 50, 20])
        self.upper_brown = np.array([20, 255, 200])

        # Таймеры
        self.offboard_counter = 0
        self.hover_start_time = None
        self.led_start_time = None
        self.found_coords = None

        self.timer = self.create_timer(0.05, self.control_loop)
        self.log("Нода запущена! Ожидание инициализации ArUco и PX4...")

    def log(self, message):
        """Логирование событий"""
        msg = String()
        timestamp = self.get_clock().now().to_msg().sec
        msg.data = f"[{timestamp}] {message}"
        self.log_pub.publish(msg)
        self.get_logger().info(message)

    def pose_cb(self, msg):
        self.current_pose = msg

    def image_cb(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            pass

    def set_led_color(self, r, g, b, a=1.0):
        """Светодиодная индикация"""
        color = ColorRGBA()
        color.r = float(r)
        color.g = float(g)
        color.b = float(b)
        color.a = float(a)
        self.led_pub.publish(color)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        self.offboard_mode_pub.publish(msg)

    def publish_trajectory_setpoint(self, x, y, z):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float(x), float(y), float(z)]
        msg.yaw = 0.0
        self.trajectory_pub.publish(msg)

    def send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def arm(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.log("Команда ARMING отправлена.")

    def set_offboard_mode(self):
        self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self.log("Запрос на режим OFFBOARD отправлен.")

    def process_hsv_detection(self):
        if self.latest_image is None:
            return False, 0, 0

        hsv = cv2.cvtColor(self.latest_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_brown, self.upper_brown)
        
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width, _ = self.latest_image.shape
        center_x, center_y = width // 2, height // 2

        for cnt in contours:
            if cv2.contourArea(cnt) > 400:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return True, cx - center_x, cy - center_y

        return False, 0, 0

    def control_loop(self):
        # Постоянная отправка позиционных установок в PX4
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint(self.target_x, self.target_y, self.target_z)

        # 1. ИНИЦИАЛИЗАЦИЯ
        if self.state == "INIT":
            self.offboard_counter += 1
            if self.offboard_counter > 20:
                self.set_offboard_mode()
                self.arm()
                self.log("Старт взлета из центра ячейки A6...")
                self.state = "TAKEOFF"

        # 2. ВЗЛЕТ НА 1.2M
        elif self.state == "TAKEOFF":
            self.target_z = self.TAKEOFF_ALT
            time.sleep(5)
            """
            curr_z = abs(self.current_pose.pose.position.z)
            
            if abs(curr_z - self.TARGET_ALTITUDE) < 0.3:
                self.log("Высота достигнута. Зависание над центром A6 на 3 секунды...")
                self.hover_start_time = time.time()
                self.state = "HOVER_AFTER_TAKEOFF"
            """

        # 3. ЗАВИСАНИЕ ПОСЛЕ ВЗЛЁТА (3 СЕКУНДЫ)
        elif self.state == "HOVER_AFTER_TAKEOFF":
            self.target_x = 0.0
            self.target_y = 0.0
            
            if time.time() - self.hover_start_time >= 3.0:
                self.log("Пауза завершена. Начало движения по маршруту: Центр A1 -> Центр E2...")
                self.state = "SEARCH"

        # 4. ПОИСК ЧЕБУРАШКИ ПО МАРШРУТУ (A1 -> E2)
        elif self.state == "SEARCH":
            found, err_x, err_y = self.process_hsv_detection()

            if found:
                kP = 0.001
                self.target_x += -err_y * kP
                self.target_y += -err_x * kP

                if abs(err_x) < 20 and abs(err_y) < 20:
                    self.log("Чебурашка обнаружен! Дрон отцентрирован точно над объектом.")
                    self.found_coords = Point(
                        x=self.current_pose.pose.position.x,
                        y=self.current_pose.pose.position.y,
                        z=0.0
                    )
                    self.rover_target_pub.publish(self.found_coords)
                    self.led_start_time = time.time()
                    self.state = "INDICATE_BLUE"

            else:
                wp = self.search_waypoints[self.current_waypoint_idx]
                self.target_x = wp["x"]
                self.target_y = wp["y"]

                # Вычисление дистанции до центра целевой клетки
                dx = self.current_pose.pose.position.x - wp["x"]
                dy = self.current_pose.pose.position.y - wp["y"]
                dist = math.hypot(dx, dy)

                if int(time.time() * 2) % 2 == 0:
                    self.get_logger().info(f"Движение к {wp['name']} ({wp['x']}, {wp['y']}). Дистанция: {dist:.2f}м")

                # При достижении центра ячейки переходим к следующей
                if dist < 0.35:
                    self.log(f"Достигнут {wp['name']}.")
                    if self.current_waypoint_idx < len(self.search_waypoints) - 1:
                        self.current_waypoint_idx += 1
                    else:
                        self.log("Все ключевые точки пройдены, объект не обнаружен. Возврат домой.")
                        self.state = "RETURN_HOME"

        # 5. СИНЯЯ ИНДИКАЦИЯ НА 3 СЕКУНДЫ ПО РЕГЛАМЕНТУ
        elif self.state == "INDICATE_BLUE":
            self.set_led_color(0.0, 0.0, 1.0)
            if time.time() - self.led_start_time >= 3.5:
                self.set_led_color(0.0, 0.0, 0.0)
                self.log("Индикация завершена. Возврат на стартовую позицию в центр A6...")
                self.state = "RETURN_HOME"

        # 6. ВОЗВРАТ НА СТАРТОВУЮ ПЛОЩАДКУ (Центр A6)
        elif self.state == "RETURN_HOME":
            self.target_x = 0.0
            self.target_y = 0.0
            dist = math.hypot(self.current_pose.pose.position.x, self.current_pose.pose.position.y)
            
            if dist < 0.3:
                self.log("Дрон вернулся в центр ячейки A6. Посадка...")
                self.state = "LAND"

        # 7. ПОСАДКА
        elif self.state == "LAND":
            self.send_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.log("Команда LAND отправлена. Посадка завершена.")
            self.state = "FINISHED"


def main(args=None):
    rclpy.init(args=args)
    node = CheburashkaFinderPX4()
    
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()