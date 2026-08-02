#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (
    PoseStamped,
    PoseWithCovarianceStamped,
    Twist,
)
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Int32, String, UInt8
from tf2_ros import Buffer, TransformException, TransformListener


# ============================================================
# СИСТЕМА КООРДИНАТ РЕАЛЬНОГО ПОЛЯ
# ============================================================
#
# Начало координат находится в стартовой позиции ровера.
#
#   x > 0 — вниз по фотографии поля
#   x < 0 — вверх
#   y > 0 — вправо
#   y < 0 — влево
#
# Маршруты ниже специально проходят через центры коридоров,
# а не отправляют ровер напрямую к конечной координате.
# ============================================================


@dataclass(frozen=True)
class Waypoint:
    name: str
    x: float
    y: float
    final_yaw: float | None = None


@dataclass(frozen=True)
class Route:
    name: str
    points: tuple[Waypoint, ...]


INTEREST_POINT_X_OFFSET = 0.15


ROUTES: dict[str, Route] = {
    "1": Route(
        name="Маршрут к ближней жёлтой метке",
        points=(
            Waypoint(
                name="Спуск по левому коридору",
                x=1.50,
                y=0.00,
            ),
            Waypoint(
                name="Обход нижнего края перегородки",
                x=1.50,
                y=0.80,
            ),
            Waypoint(
                name="Ближняя жёлтая метка + 15 см по X",
                x=0.80 + INTEREST_POINT_X_OFFSET,
                y=0.80,
                final_yaw=0.0,
            ),
        ),
    ),
    "2": Route(
        name="Маршрут к верхней левой жёлтой метке",
        points=(
            Waypoint(
                name="Спуск по левому коридору",
                x=1.50,
                y=0.00,
            ),
            Waypoint(
                name="Проход вправо под перегородками",
                x=1.50,
                y=2.25,
            ),
            Waypoint(
                name="Подъём за длинной перегородкой",
                x=0.80,
                y=2.25,
            ),
            Waypoint(
                name="Диагональный проход через центр",
                x=-0.80,
                y=1.30,
            ),
            Waypoint(
                name="Выход к левому верхнему карману",
                x=-1.80,
                y=0.20,
            ),
            Waypoint(
                name="Верхняя левая жёлтая метка + 15 см по X",
                x=-2.40 + INTEREST_POINT_X_OFFSET,
                y=0.00,
                final_yaw=0.0,
            ),
        ),
    ),
}


def yaw_to_quaternion(
    yaw: float,
) -> tuple[float, float, float, float]:
    half = yaw / 2.0

    return (
        0.0,
        0.0,
        math.sin(half),
        math.cos(half),
    )


def quaternion_to_yaw(
    x: float,
    y: float,
    z: float,
    w: float,
) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(
        sin_yaw,
        cos_yaw,
    )


def distance_2d(
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    return math.hypot(
        bx - ax,
        by - ay,
    )


def path_length(
    path: Path,
) -> float:
    total = 0.0

    for previous, current in zip(
        path.poses,
        path.poses[1:],
    ):
        total += distance_2d(
            previous.pose.position.x,
            previous.pose.position.y,
            current.pose.position.x,
            current.pose.position.y,
        )

    return total


class LabyrinthWaypointMission(Node):
    def __init__(
        self,
        planner_id: str,
        waypoint_timeout: float,
        retries: int,
        waypoint_tolerance: float,
        hold_time: float,
        position_std_limit: float,
        yaw_std_limit_deg: float,
        require_amcl: bool,
        spin_speed: float,
        stop_angle_deg: float,
        recognition_pause: float,
        required_confirmations: int,
        light_topic: str,
        light_type: str,
        keep_light_on: bool,
        light_settle_time: float,
    ) -> None:
        super().__init__("labyrinth_waypoint_mission")

        self.planner_id = planner_id
        self.waypoint_timeout = waypoint_timeout
        self.retries = retries
        self.waypoint_tolerance = waypoint_tolerance
        self.hold_time = hold_time
        self.position_std_limit = position_std_limit
        self.yaw_std_limit = math.radians(
            yaw_std_limit_deg
        )
        self.require_amcl = require_amcl

        self.spin_speed = max(abs(spin_speed), 0.05)
        self.stop_angle = math.radians(
            max(stop_angle_deg, 10.0)
        )
        self.recognition_pause = max(
            recognition_pause,
            0.5,
        )
        self.required_confirmations = max(
            required_confirmations,
            2,
        )

        self.light_topic_override = light_topic.strip()
        self.light_type_override = light_type
        self.keep_light_on = keep_light_on
        self.light_settle_time = max(
            light_settle_time,
            0.0,
        )

        self.light_publisher = None
        self.light_message_type = ""
        self.light_topic = ""

        self.led_message_class = None
        self.led_state_subscription = None
        self.latest_led_state = None
        self.latest_led_state_time = 0.0
        self.led_fields_logged = False

        self.compute_path_client = ActionClient(
            self,
            ComputePathToPose,
            "/compute_path_to_pose",
        )

        self.navigate_client = ActionClient(
            self,
            NavigateToPose,
            "/navigate_to_pose",
        )

        self.clear_global_client = self.create_client(
            ClearEntireCostmap,
            "/global_costmap/clear_entirely_global_costmap",
        )

        self.clear_local_client = self.create_client(
            ClearEntireCostmap,
            "/local_costmap/clear_entirely_local_costmap",
        )

        self.velocity_publisher = self.create_publisher(
            Twist,
            "/cmd_vel_nav",
            10,
        )

        path_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.route_path_publisher = self.create_publisher(
            Path,
            "/mission/planned_path",
            path_qos,
        )

        self.segment_path_publisher = self.create_publisher(
            Path,
            "/mission/current_segment",
            path_qos,
        )

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.on_map,
            map_qos,
        )

        amcl_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.amcl_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.on_amcl_pose,
            amcl_qos,
        )

        code_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.code_subscription = self.create_subscription(
            String,
            "/recognized_code",
            self.on_code,
            code_qos,
        )

        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.latest_map: OccupancyGrid | None = None
        self.latest_amcl: PoseWithCovarianceStamped | None = None
        self.latest_amcl_received_at = 0.0

        self.active_navigation_goal = None
        self.search_active = False
        self.recognized_code: str | None = None
        self.selected_code: str | None = None
        self.recognition_counts: dict[str, int] = {}

        self.last_feedback_print = 0.0
        self.current_waypoint_index = 0
        self.total_waypoints = 0

    # ========================================================
    # ПОДПИСКИ
    # ========================================================

    def on_map(
        self,
        message: OccupancyGrid,
    ) -> None:
        self.latest_map = message

    def on_amcl_pose(
        self,
        message: PoseWithCovarianceStamped,
    ) -> None:
        self.latest_amcl = message
        self.latest_amcl_received_at = time.monotonic()

    def on_code(
        self,
        message: String,
    ) -> None:
        # Результаты принимаются только тогда, когда ровер
        # полностью остановлен и ждёт распознавание.
        if not self.search_active:
            return

        code = "".join(
            message.data.strip().upper().split()
        )

        if not code:
            return

        if code in {
            "NONE",
            "UNKNOWN",
            "NO_TEXT",
            "NOT_FOUND",
            "-",
        }:
            return

        count = (
            self.recognition_counts.get(code, 0)
            + 1
        )

        self.recognition_counts[code] = count

        self.get_logger().info(
            f"Кандидат '{code}': "
            f"{count}/{self.required_confirmations}"
        )

        if (
            count >= self.required_confirmations
            and self.selected_code is None
        ):
            self.selected_code = code
            self.recognized_code = code

            self.get_logger().info(
                "КОД ПОДТВЕРЖДЁН ДВАЖДЫ: "
                f"{code}"
            )

    # ========================================================
    # TF И ЛОКАЛИЗАЦИЯ
    # ========================================================

    def current_pose(
        self,
    ) -> tuple[float, float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                Time(),
                timeout=Duration(seconds=1.0),
            )
        except TransformException as error:
            self.get_logger().error(
                "Не удалось получить TF map → base_link: "
                f"{error}"
            )
            return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation

        return (
            float(translation.x),
            float(translation.y),
            quaternion_to_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            ),
        )

    def wait_for_system(
        self,
        timeout: float = 60.0,
    ) -> bool:
        self.get_logger().info(
            "Ожидаю карту, локализацию и Nav2..."
        )

        deadline = time.monotonic() + timeout
        last_print = 0.0

        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(
                self,
                timeout_sec=0.2,
            )

            map_ready = (
                self.latest_map is not None
                and self.latest_map.info.width > 0
                and self.latest_map.info.height > 0
                and self.latest_map.info.resolution > 0.0
            )

            tf_ready = self.tf_buffer.can_transform(
                "map",
                "base_link",
                Time(),
                timeout=Duration(seconds=0.1),
            )

            planner_ready = (
                self.compute_path_client.server_is_ready()
            )

            navigation_ready = (
                self.navigate_client.server_is_ready()
            )

            if (
                map_ready
                and tf_ready
                and planner_ready
                and navigation_ready
            ):
                pose = self.current_pose()

                if pose is None:
                    continue

                x, y, yaw = pose

                self.get_logger().info(
                    "Система готова. "
                    f"Поза: x={x:.3f}, y={y:.3f}, "
                    f"yaw={math.degrees(yaw):.1f}°"
                )

                return True

            now = time.monotonic()

            if now - last_print >= 3.0:
                last_print = now

                self.get_logger().info(
                    "Готовность: "
                    f"map={map_ready}, "
                    f"tf={tf_ready}, "
                    f"planner={planner_ready}, "
                    f"navigate={navigation_ready}"
                )

        self.get_logger().error(
            "Не дождался готовности карты, TF или Nav2"
        )
        return False

    def localization_metrics(
        self,
    ) -> tuple[float, float] | None:
        if self.latest_amcl is None:
            return None

        if (
            time.monotonic()
            - self.latest_amcl_received_at
            > 3.0
        ):
            return None

        covariance = self.latest_amcl.pose.covariance

        position_variance = max(
            float(covariance[0]),
            float(covariance[7]),
            0.0,
        )

        yaw_variance = max(
            float(covariance[35]),
            0.0,
        )

        return (
            math.sqrt(position_variance),
            math.sqrt(yaw_variance),
        )

    def wait_for_good_localization(
        self,
        timeout: float = 12.0,
    ) -> bool:
        """
        Проверяет AMCL только когда это действительно возможно.

        В текущем запуске TF map -> base_link работает, но /amcl_pose
        может не публиковаться. Без --require-amcl это не должно
        задерживать или блокировать движение.
        """
        publisher_count = self.count_publishers(
            "/amcl_pose"
        )

        if publisher_count == 0:
            if self.require_amcl:
                self.get_logger().error(
                    "У /amcl_pose нет издателей, "
                    "а задан параметр --require-amcl"
                )
                return False

            self.get_logger().info(
                "/amcl_pose не публикуется — "
                "использую рабочий TF map → base_link"
            )
            return self.current_pose() is not None

        # В необязательном режиме не задерживаем запуск на 12 секунд.
        effective_timeout = (
            timeout
            if self.require_amcl
            else min(timeout, 2.0)
        )

        deadline = (
            time.monotonic()
            + effective_timeout
        )
        last_print = 0.0

        while (
            rclpy.ok()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.2,
            )

            metrics = self.localization_metrics()

            if metrics is None:
                continue

            position_std, yaw_std = metrics

            if (
                position_std
                <= self.position_std_limit
                and yaw_std
                <= self.yaw_std_limit
            ):
                self.get_logger().info(
                    "AMCL в норме: "
                    f"σxy={position_std:.3f} м, "
                    f"σyaw="
                    f"{math.degrees(yaw_std):.1f}°"
                )
                return True

            now = time.monotonic()

            if now - last_print >= 1.0:
                last_print = now

                self.get_logger().warning(
                    "Локализация неточная: "
                    f"σxy={position_std:.3f} м, "
                    f"σyaw="
                    f"{math.degrees(yaw_std):.1f}°"
                )

        metrics = self.localization_metrics()

        if metrics is None:
            if self.require_amcl:
                self.get_logger().error(
                    "Нет свежего сообщения /amcl_pose"
                )
                return False

            self.get_logger().info(
                "/amcl_pose пока не получен — "
                "продолжаю по TF map → base_link"
            )
            return self.current_pose() is not None

        position_std, yaw_std = metrics

        if self.require_amcl:
            self.get_logger().error(
                "Локализация слишком неточная: "
                f"σxy={position_std:.3f} м, "
                f"σyaw="
                f"{math.degrees(yaw_std):.1f}°"
            )
            return False

        self.get_logger().warning(
            "AMCL сообщает большую погрешность, "
            "но строгий режим не включён. "
            "Продолжаю по TF."
        )

        return self.current_pose() is not None

    # ========================================================
    # КАРТА
    # ========================================================

    def map_cell_value(
        self,
        x: float,
        y: float,
    ) -> int | None:
        grid = self.latest_map

        if grid is None:
            return None

        resolution = float(grid.info.resolution)

        if resolution <= 0.0:
            return None

        origin = grid.info.origin

        origin_yaw = quaternion_to_yaw(
            origin.orientation.x,
            origin.orientation.y,
            origin.orientation.z,
            origin.orientation.w,
        )

        dx = x - float(origin.position.x)
        dy = y - float(origin.position.y)

        cos_yaw = math.cos(origin_yaw)
        sin_yaw = math.sin(origin_yaw)

        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy

        cell_x = int(
            math.floor(local_x / resolution)
        )

        cell_y = int(
            math.floor(local_y / resolution)
        )

        if (
            cell_x < 0
            or cell_y < 0
            or cell_x >= grid.info.width
            or cell_y >= grid.info.height
        ):
            return None

        index = (
            cell_y * grid.info.width
            + cell_x
        )

        if index < 0 or index >= len(grid.data):
            return None

        return int(grid.data[index])

    def validate_waypoints(
        self,
        route: Route,
    ) -> bool:
        for index, waypoint in enumerate(
            route.points,
            start=1,
        ):
            value = self.map_cell_value(
                waypoint.x,
                waypoint.y,
            )

            if value is None:
                self.get_logger().error(
                    f"Точка {index} находится "
                    "за границами карты: "
                    f"({waypoint.x:.2f}, {waypoint.y:.2f})"
                )
                return False

            if value < 0:
                self.get_logger().error(
                    f"Точка {index} попала "
                    "в неизвестную область карты"
                )
                return False

            if value >= 65:
                self.get_logger().error(
                    f"Точка {index} попала в стену: "
                    f"occupancy={value}"
                )
                return False

        return True

    # ========================================================
    # ПОЗЫ И ПЛАНИРОВАНИЕ
    # ========================================================

    def make_pose(
        self,
        x: float,
        y: float,
        yaw: float,
    ) -> PoseStamped:
        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = (
            self.get_clock().now().to_msg()
        )

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quaternion(yaw)

        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        return pose

    def route_poses(
        self,
        route: Route,
    ) -> list[PoseStamped]:
        poses: list[PoseStamped] = []

        for index, waypoint in enumerate(route.points):
            if waypoint.final_yaw is not None:
                yaw = waypoint.final_yaw
            elif index + 1 < len(route.points):
                next_waypoint = route.points[index + 1]

                yaw = math.atan2(
                    next_waypoint.y - waypoint.y,
                    next_waypoint.x - waypoint.x,
                )
            else:
                yaw = 0.0

            poses.append(
                self.make_pose(
                    waypoint.x,
                    waypoint.y,
                    yaw,
                )
            )

        return poses

    def compute_segment(
        self,
        start: PoseStamped,
        goal: PoseStamped,
        timeout: float = 20.0,
    ) -> Path | None:
        action_goal = ComputePathToPose.Goal()

        action_goal.start = start
        action_goal.goal = goal
        action_goal.use_start = True
        action_goal.planner_id = self.planner_id

        send_future = (
            self.compute_path_client.send_goal_async(
                action_goal
            )
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done():
            self.get_logger().error(
                "planner_server не ответил"
            )
            return None

        goal_handle = send_future.result()

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):
            self.get_logger().error(
                "planner_server отклонил сегмент"
            )
            return None

        result_future = (
            goal_handle.get_result_async()
        )

        rclpy.spin_until_future_complete(
            self,
            result_future,
            timeout_sec=timeout,
        )

        if not result_future.done():
            cancel_future = (
                goal_handle.cancel_goal_async()
            )

            rclpy.spin_until_future_complete(
                self,
                cancel_future,
                timeout_sec=5.0,
            )

            self.get_logger().error(
                "Планирование сегмента превысило таймаут"
            )
            return None

        wrapped = result_future.result()

        if wrapped is None:
            return None

        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            result = wrapped.result

            self.get_logger().error(
                "Не удалось построить сегмент: "
                f"status={wrapped.status}, "
                f"error_code="
                f"{getattr(result, 'error_code', 'unknown')}, "
                f"error_msg="
                f"{getattr(result, 'error_msg', '')}"
            )
            return None

        path = wrapped.result.path

        if path is None or len(path.poses) < 2:
            return None

        return path

    def preplan_route(
        self,
        route: Route,
    ) -> Path | None:
        current = self.current_pose()

        if current is None:
            return None

        current_x, current_y, current_yaw = current

        start_pose = self.make_pose(
            current_x,
            current_y,
            current_yaw,
        )

        waypoint_poses = self.route_poses(route)

        combined = Path()
        combined.header.frame_id = "map"
        combined.header.stamp = (
            self.get_clock().now().to_msg()
        )

        total_length = 0.0

        for index, goal_pose in enumerate(
            waypoint_poses,
            start=1,
        ):
            self.get_logger().info(
                f"Проверяю сегмент "
                f"{index}/{len(waypoint_poses)}..."
            )

            segment = self.compute_segment(
                start_pose,
                goal_pose,
            )

            if segment is None:
                self.get_logger().error(
                    f"Маршрут невозможен на сегменте {index}"
                )
                return None

            segment_length = path_length(segment)
            total_length += segment_length

            self.get_logger().info(
                f"Сегмент {index}: "
                f"{segment_length:.2f} м, "
                f"{len(segment.poses)} поз"
            )

            if combined.poses:
                combined.poses.extend(
                    segment.poses[1:]
                )
            else:
                combined.poses.extend(
                    segment.poses
                )

            start_pose = goal_pose

        self.route_path_publisher.publish(combined)

        self.get_logger().info(
            "Полный маршрут проверен: "
            f"длина={total_length:.2f} м, "
            f"промежуточных точек={len(route.points)}"
        )

        return combined

    # ========================================================
    # COSTMAP
    # ========================================================

    def clear_costmaps(self) -> None:
        request = ClearEntireCostmap.Request()

        clients = (
            (
                "global",
                self.clear_global_client,
            ),
            (
                "local",
                self.clear_local_client,
            ),
        )

        for name, client in clients:
            if not client.wait_for_service(
                timeout_sec=2.0,
            ):
                self.get_logger().warning(
                    f"Сервис очистки {name} costmap недоступен"
                )
                continue

            future = client.call_async(request)

            rclpy.spin_until_future_complete(
                self,
                future,
                timeout_sec=5.0,
            )

            if not future.done():
                self.get_logger().warning(
                    f"Не удалось очистить {name} costmap"
                )

        time.sleep(0.5)

    # ========================================================
    # ДВИЖЕНИЕ ПО ТОЧКАМ
    # ========================================================

    def navigation_feedback(
        self,
        feedback_message,
    ) -> None:
        now = time.monotonic()

        if now - self.last_feedback_print < 1.0:
            return

        self.last_feedback_print = now
        feedback = feedback_message.feedback

        self.get_logger().info(
            f"Точка "
            f"{self.current_waypoint_index}/"
            f"{self.total_waypoints}: "
            f"осталось="
            f"{float(feedback.distance_remaining):.2f} м, "
            f"recoveries="
            f"{int(feedback.number_of_recoveries)}"
        )

    def navigate_to_pose(
        self,
        pose: PoseStamped,
    ) -> bool:
        action_goal = NavigateToPose.Goal()
        action_goal.pose = pose

        send_future = (
            self.navigate_client.send_goal_async(
                action_goal,
                feedback_callback=(
                    self.navigation_feedback
                ),
            )
        )

        rclpy.spin_until_future_complete(
            self,
            send_future,
            timeout_sec=10.0,
        )

        if not send_future.done():
            self.get_logger().error(
                "Nav2 не ответил на цель"
            )
            return False

        goal_handle = send_future.result()

        if (
            goal_handle is None
            or not goal_handle.accepted
        ):
            self.get_logger().error(
                "Nav2 отклонил промежуточную точку"
            )
            return False

        self.active_navigation_goal = goal_handle

        result_future = (
            goal_handle.get_result_async()
        )

        deadline = (
            time.monotonic()
            + self.waypoint_timeout
        )

        while (
            rclpy.ok()
            and not result_future.done()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.1,
            )

        self.active_navigation_goal = None

        if not result_future.done():
            self.get_logger().error(
                "Движение к точке превысило таймаут"
            )

            cancel_future = (
                goal_handle.cancel_goal_async()
            )

            rclpy.spin_until_future_complete(
                self,
                cancel_future,
                timeout_sec=5.0,
            )

            self.stop_robot()
            return False

        wrapped = result_future.result()

        if wrapped is None:
            self.stop_robot()
            return False

        if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
            result = wrapped.result

            self.get_logger().error(
                "Nav2 не достиг точки: "
                f"status={wrapped.status}, "
                f"error_code="
                f"{getattr(result, 'error_code', 'unknown')}, "
                f"error_msg="
                f"{getattr(result, 'error_msg', '')}"
            )

            self.stop_robot()
            return False

        self.stop_robot()
        return True

    def navigate_route(
        self,
        route: Route,
    ) -> bool:
        poses = self.route_poses(route)

        self.total_waypoints = len(poses)

        for index, (
            waypoint,
            target_pose,
        ) in enumerate(
            zip(route.points, poses),
            start=1,
        ):
            self.current_waypoint_index = index

            current = self.current_pose()

            if current is None:
                return False

            current_x, current_y, current_yaw = current

            remaining = distance_2d(
                current_x,
                current_y,
                waypoint.x,
                waypoint.y,
            )

            if remaining <= self.waypoint_tolerance:
                self.get_logger().info(
                    f"Точка {index} уже достигнута "
                    f"(расстояние {remaining:.2f} м), "
                    "пропускаю"
                )
                continue

            if not self.wait_for_good_localization():
                self.get_logger().error(
                    "Останавливаю маршрут: "
                    "AMCL потерял локализацию"
                )
                self.stop_robot()
                return False

            self.get_logger().info(
                "=" * 52
            )

            self.get_logger().info(
                f"Точка {index}/{len(poses)}: "
                f"{waypoint.name}"
            )

            self.get_logger().info(
                f"Цель: x={waypoint.x:.2f}, "
                f"y={waypoint.y:.2f}; "
                f"до неё {remaining:.2f} м"
            )

            success = False

            for attempt in range(
                1,
                self.retries + 2,
            ):
                current = self.current_pose()

                if current is None:
                    break

                current_x, current_y, current_yaw = current

                start_pose = self.make_pose(
                    current_x,
                    current_y,
                    current_yaw,
                )

                segment = self.compute_segment(
                    start_pose,
                    target_pose,
                )

                if segment is None:
                    self.get_logger().warning(
                        f"Попытка {attempt}: "
                        "сегмент не построен"
                    )
                else:
                    segment.header.frame_id = "map"
                    segment.header.stamp = (
                        self.get_clock().now().to_msg()
                    )

                    self.segment_path_publisher.publish(
                        segment
                    )

                    self.get_logger().info(
                        f"Попытка {attempt}: "
                        f"сегмент {path_length(segment):.2f} м"
                    )

                    if self.navigate_to_pose(
                        target_pose
                    ):
                        success = True
                        break

                if attempt <= self.retries:
                    self.get_logger().warning(
                        "Очищаю costmap и перепланирую..."
                    )

                    self.clear_costmaps()

                    time.sleep(1.0)

            if not success:
                self.get_logger().error(
                    f"Не удалось достичь точки {index}"
                )
                self.stop_robot()
                return False

            actual = self.current_pose()

            if actual is not None:
                actual_x, actual_y, actual_yaw = actual

                error = distance_2d(
                    actual_x,
                    actual_y,
                    waypoint.x,
                    waypoint.y,
                )

                self.get_logger().info(
                    f"Точка {index} достигнута. "
                    f"Погрешность={error:.2f} м, "
                    f"поза=({actual_x:.2f}, {actual_y:.2f})"
                )

            if self.hold_time > 0.0:
                end = (
                    time.monotonic()
                    + self.hold_time
                )

                while (
                    rclpy.ok()
                    and time.monotonic() < end
                ):
                    self.stop_robot()

        return True

    # ========================================================
    # ПОИСК КОДА
    # ========================================================

    # ========================================================
    # ПОДСВЕТКА
    # ========================================================

    @staticmethod
    def supported_light_types() -> dict[str, type]:
        supported: dict[str, type] = {
            "std_msgs/msg/Bool": Bool,
            "std_msgs/msg/Int32": Int32,
            "std_msgs/msg/UInt8": UInt8,
            "std_msgs/msg/String": String,
        }

        try:
            supported[
                "rover_interfaces/msg/LedStripState"
            ] = get_message(
                "rover_interfaces/msg/LedStripState"
            )
        except Exception:
            # Сообщение может стать доступным после source
            # рабочего пространства; повторная попытка делается
            # в resolve_light_interface().
            pass

        return supported

    def on_led_state(
        self,
        message,
    ) -> None:
        self.latest_led_state = message
        self.latest_led_state_time = time.monotonic()

    @staticmethod
    def value_for_ros_type(
        ros_type: str,
        enabled: bool,
        maximum: int = 255,
    ):
        lowered = ros_type.lower()

        if "bool" in lowered:
            return enabled

        if "float" in lowered or "double" in lowered:
            return 1.0 if enabled else 0.0

        if "uint8" in lowered or "int8" in lowered:
            return maximum if enabled else 0

        if "uint16" in lowered or "int16" in lowered:
            return maximum if enabled else 0

        if (
            "uint32" in lowered
            or "int32" in lowered
            or "uint64" in lowered
            or "int64" in lowered
        ):
            return 1 if enabled else 0

        return maximum if enabled else 0

    def select_constant(
        self,
        message_class,
        enabled: bool,
    ):
        names = (
            (
                "SOLID",
                "STATIC",
                "ON",
                "ENABLED",
                "MODE_SOLID",
                "EFFECT_SOLID",
            )
            if enabled
            else (
                "OFF",
                "DISABLED",
                "NONE",
                "MODE_OFF",
                "EFFECT_OFF",
            )
        )

        for name in names:
            if hasattr(message_class, name):
                return getattr(message_class, name)

        return None

    def set_nested_color(
        self,
        value,
        enabled: bool,
    ) -> bool:
        if not hasattr(
            value,
            "get_fields_and_field_types",
        ):
            return False

        fields = value.get_fields_and_field_types()
        changed = False

        for field_name, ros_type in fields.items():
            lowered = field_name.lower()

            if lowered in ("r", "red"):
                setattr(
                    value,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True

            elif lowered in ("g", "green"):
                setattr(
                    value,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True

            elif lowered in ("b", "blue"):
                setattr(
                    value,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True

            elif lowered in ("a", "alpha"):
                setattr(
                    value,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True

            elif lowered in (
                "brightness",
                "intensity",
                "value",
            ):
                setattr(
                    value,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True

        return changed

    def configure_ledstrip_message(
        self,
        message,
        enabled: bool,
    ) -> bool:
        if not hasattr(
            message,
            "get_fields_and_field_types",
        ):
            return False

        fields = message.get_fields_and_field_types()

        if not self.led_fields_logged:
            self.led_fields_logged = True

            self.get_logger().info(
                "Поля LedStripState: "
                + ", ".join(
                    f"{name}:{field_type}"
                    for name, field_type in fields.items()
                )
            )

        changed = False
        message_class = type(message)

        for field_name, ros_type in fields.items():
            lowered = field_name.lower()
            current_value = getattr(
                message,
                field_name,
            )

            if lowered in (
                "enabled",
                "enable",
                "is_enabled",
                "on",
                "is_on",
                "active",
                "power",
            ):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        1,
                    ),
                )
                changed = True
                continue

            if lowered in ("r", "red"):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True
                continue

            if lowered in ("g", "green"):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True
                continue

            if lowered in ("b", "blue"):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True
                continue

            if lowered in ("a", "alpha"):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True
                continue

            if lowered in (
                "brightness",
                "intensity",
                "level",
                "value",
            ):
                setattr(
                    message,
                    field_name,
                    self.value_for_ros_type(
                        ros_type,
                        enabled,
                        255,
                    ),
                )
                changed = True
                continue

            if lowered in (
                "effect",
                "mode",
                "pattern",
                "state",
            ):
                if "string" in ros_type.lower():
                    setattr(
                        message,
                        field_name,
                        "solid" if enabled else "off",
                    )
                    changed = True
                else:
                    constant = self.select_constant(
                        message_class,
                        enabled,
                    )

                    if constant is not None:
                        setattr(
                            message,
                            field_name,
                            constant,
                        )
                    else:
                        setattr(
                            message,
                            field_name,
                            1 if enabled else 0,
                        )

                    changed = True

                continue

            if lowered in (
                "color",
                "colour",
                "rgb",
                "rgba",
                "primary_color",
            ):
                if self.set_nested_color(
                    current_value,
                    enabled,
                ):
                    changed = True

                continue

            if lowered in (
                "colors",
                "colours",
                "pixels",
                "leds",
            ):
                try:
                    for item in current_value:
                        if self.set_nested_color(
                            item,
                            enabled,
                        ):
                            changed = True
                except TypeError:
                    pass

                continue

            if lowered == "data" and "string" in ros_type.lower():
                setattr(
                    message,
                    field_name,
                    json.dumps(
                        {
                            "enabled": enabled,
                            "effect": (
                                "solid"
                                if enabled
                                else "off"
                            ),
                            "r": 255 if enabled else 0,
                            "g": 255 if enabled else 0,
                            "b": 255 if enabled else 0,
                            "brightness": (
                                255 if enabled else 0
                            ),
                        },
                        ensure_ascii=False,
                    ),
                )
                changed = True

        return changed

    def resolve_light_interface(self) -> bool:
        if self.light_publisher is not None:
            return True

        graph = {
            topic: types
            for topic, types
            in self.get_topic_names_and_types()
        }

        supported = self.supported_light_types()

        custom_type = (
            "rover_interfaces/msg/LedStripState"
        )

        if custom_type not in supported:
            try:
                supported[custom_type] = get_message(
                    custom_type
                )
            except Exception as error:
                self.get_logger().error(
                    "Не удалось загрузить "
                    "rover_interfaces/msg/LedStripState. "
                    "Проверь source install/setup.bash: "
                    f"{error}"
                )
                return False

        # /led_strip/state публикует фактическое состояние.
        # Для управления используется /led_strip/set_state.
        requested_topic = (
            self.light_topic_override
            or "/led_strip/set_state"
        )

        if not requested_topic.startswith("/"):
            requested_topic = (
                f"/{requested_topic}"
            )

        if requested_topic == "/led_strip/state":
            self.get_logger().warning(
                "/led_strip/state — выход состояния, "
                "переключаю управление на "
                "/led_strip/set_state"
            )

            requested_topic = (
                "/led_strip/set_state"
            )

        available_types = graph.get(
            requested_topic,
            [],
        )

        state_types = graph.get(
            "/led_strip/state",
            [],
        )

        if self.light_type_override == "auto":
            if custom_type in available_types:
                selected_type = custom_type
            elif custom_type in state_types:
                selected_type = custom_type
            else:
                selected_type = next(
                    (
                        type_name
                        for type_name in available_types
                        if type_name in supported
                    ),
                    custom_type,
                )

        elif self.light_type_override == "ledstrip":
            selected_type = custom_type

        else:
            selected_type = {
                "bool": "std_msgs/msg/Bool",
                "int": "std_msgs/msg/Int32",
                "uint8": "std_msgs/msg/UInt8",
                "string": "std_msgs/msg/String",
            }[self.light_type_override]

        message_class = supported.get(
            selected_type
        )

        if message_class is None:
            self.get_logger().error(
                "Неподдерживаемый тип подсветки: "
                f"{selected_type}"
            )
            return False

        self.light_topic = requested_topic
        self.light_message_type = selected_type
        self.led_message_class = message_class

        self.light_publisher = self.create_publisher(
            message_class,
            requested_topic,
            10,
        )

        if selected_type == custom_type:
            state_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=(
                    ReliabilityPolicy.BEST_EFFORT
                ),
                durability=(
                    DurabilityPolicy.VOLATILE
                ),
            )

            self.led_state_subscription = (
                self.create_subscription(
                    message_class,
                    "/led_strip/state",
                    self.on_led_state,
                    state_qos,
                )
            )

            # Коротко ждём первое состояние, чтобы сохранить
            # неизвестные поля сообщения при отправке команды.
            deadline = time.monotonic() + 1.0

            while (
                rclpy.ok()
                and self.latest_led_state is None
                and time.monotonic() < deadline
            ):
                rclpy.spin_once(
                    self,
                    timeout_sec=0.10,
                )

        subscribers = len(
            self.get_subscriptions_info_by_topic(
                requested_topic
            )
        )

        self.get_logger().info(
            "Управление подсветкой: "
            f"{requested_topic} "
            f"[{selected_type}], "
            f"подписчиков={subscribers}"
        )

        if subscribers == 0:
            self.get_logger().warning(
                "У /led_strip/set_state пока нет "
                "подписчиков. Команда будет отправлена, "
                "но аппаратный LED-узел может быть "
                "не запущен."
            )

        return True

    def make_light_message(
        self,
        enabled: bool,
    ):
        if (
            self.light_message_type
            == "std_msgs/msg/Bool"
        ):
            return Bool(data=enabled)

        if (
            self.light_message_type
            == "std_msgs/msg/Int32"
        ):
            return Int32(
                data=1 if enabled else 0
            )

        if (
            self.light_message_type
            == "std_msgs/msg/UInt8"
        ):
            return UInt8(
                data=1 if enabled else 0
            )

        if (
            self.light_message_type
            == "std_msgs/msg/String"
        ):
            return String(
                data="on" if enabled else "off"
            )

        if (
            self.light_message_type
            == "rover_interfaces/msg/LedStripState"
        ):
            if self.latest_led_state is not None:
                message = copy.deepcopy(
                    self.latest_led_state
                )
            else:
                message = self.led_message_class()

            changed = self.configure_ledstrip_message(
                message,
                enabled,
            )

            if not changed:
                fields = (
                    message.get_fields_and_field_types()
                )

                self.get_logger().error(
                    "Не удалось определить управляющие "
                    "поля LedStripState. Поля сообщения: "
                    + ", ".join(
                        f"{name}:{field_type}"
                        for name, field_type
                        in fields.items()
                    )
                )

                return None

            return message

        return None

    def set_light(
        self,
        enabled: bool,
    ) -> bool:
        if not self.resolve_light_interface():
            return False

        message = self.make_light_message(
            enabled
        )

        if message is None:
            return False

        before_state_time = (
            self.latest_led_state_time
        )

        # LED-контроллер может стартовать позднее, поэтому
        # команда отправляется несколько раз в течение секунды.
        for _ in range(10):
            if not rclpy.ok():
                return False

            try:
                self.light_publisher.publish(
                    message
                )

                rclpy.spin_once(
                    self,
                    timeout_sec=0.10,
                )

            except Exception as error:
                self.get_logger().warning(
                    "Ошибка управления подсветкой: "
                    f"{error}"
                )
                return False

        # Ждём обновление /led_strip/state. Сам факт нового
        # сообщения подтверждает, что LED-узел жив.
        deadline = time.monotonic() + 1.5

        while (
            rclpy.ok()
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(
                self,
                timeout_sec=0.10,
            )

            if (
                self.latest_led_state_time
                > before_state_time
            ):
                break

        if (
            self.light_message_type
            == "rover_interfaces/msg/LedStripState"
            and self.latest_led_state_time
            <= before_state_time
        ):
            self.get_logger().warning(
                "Команда отправлена в "
                "/led_strip/set_state, но новое состояние "
                "в /led_strip/state не пришло."
            )

        self.get_logger().info(
            "Подсветка включена"
            if enabled
            else "Подсветка выключена"
        )

        return True

    # ========================================================
    # МЕДЛЕННОЕ ВРАЩЕНИЕ С ОСТАНОВКАМИ
    # ========================================================

    def current_rotation_yaw(
        self,
    ) -> float | None:
        # odom предпочтительнее для измерения поворота:
        # корректировки AMCL не должны искажать накопленный угол.
        for target_frame in ("odom", "map"):
            try:
                transform = (
                    self.tf_buffer.lookup_transform(
                        target_frame,
                        "base_link",
                        Time(),
                        timeout=Duration(
                            seconds=0.4
                        ),
                    )
                )
            except TransformException:
                continue

            rotation = transform.transform.rotation

            return quaternion_to_yaw(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )

        return None

    def rotate_one_step(
        self,
        target_angle: float,
        timeout: float,
    ) -> float:
        previous_yaw = self.current_rotation_yaw()

        if previous_yaw is None:
            self.get_logger().error(
                "Не удалось получить yaw перед поворотом"
            )
            return 0.0

        accumulated = 0.0
        deadline = time.monotonic() + timeout

        command = Twist()
        command.angular.z = self.spin_speed

        try:
            while (
                rclpy.ok()
                and accumulated < target_angle
                and time.monotonic() < deadline
            ):
                self.velocity_publisher.publish(command)

                rclpy.spin_once(
                    self,
                    timeout_sec=0.05,
                )

                current_yaw = self.current_rotation_yaw()

                if current_yaw is None:
                    continue

                delta = math.atan2(
                    math.sin(
                        current_yaw - previous_yaw
                    ),
                    math.cos(
                        current_yaw - previous_yaw
                    ),
                )

                # Защита от единичного скачка TF.
                if abs(delta) <= 0.50:
                    accumulated += abs(delta)

                previous_yaw = current_yaw

        finally:
            self.stop_robot()

        return accumulated

    def wait_for_recognition(
        self,
        pause_number: int,
        total_pauses: int,
        recognize_code: bool,
    ) -> None:
        self.stop_robot()

        if not recognize_code:
            self.get_logger().info(
                f"Остановка {pause_number}/{total_pauses}: "
                f"жду {self.recognition_pause:.1f} с"
            )
        elif self.selected_code is not None:
            self.get_logger().info(
                f"Остановка {pause_number}/{total_pauses}: "
                f"код уже подтверждён: "
                f"{self.selected_code}"
            )
        else:
            self.get_logger().info(
                f"Остановка {pause_number}/{total_pauses}: "
                f"жду распознавание "
                f"{self.recognition_pause:.1f} с"
            )

        # Принимаем OCR только во время полной остановки.
        self.search_active = (
            recognize_code
            and self.selected_code is None
        )

        # После подтверждения всё равно выполняются два оборота,
        # но длинное ожидание больше не требуется.
        pause_duration = (
            0.5
            if self.selected_code is not None
            else self.recognition_pause
        )

        deadline = (
            time.monotonic()
            + pause_duration
        )

        while (
            rclpy.ok()
            and time.monotonic() < deadline
        ):
            self.stop_robot()

            rclpy.spin_once(
                self,
                timeout_sec=0.10,
            )

            if self.selected_code is not None:
                break

        self.search_active = False

    def perform_interest_scan(
        self,
        rotations: float,
        timeout: float,
        recognize_code: bool,
    ) -> str | None:
        self.recognized_code = None
        self.selected_code = None
        self.recognition_counts.clear()
        self.search_active = False

        light_enabled = self.set_light(True)

        if (
            light_enabled
            and self.light_settle_time > 0.0
        ):
            self.get_logger().info(
                "Жду стабилизацию подсветки..."
            )

            deadline = (
                time.monotonic()
                + self.light_settle_time
            )

            while (
                rclpy.ok()
                and time.monotonic() < deadline
            ):
                self.stop_robot()

                rclpy.spin_once(
                    self,
                    timeout_sec=0.10,
                )

        total_angle = (
            2.0
            * math.pi
            * rotations
        )

        total_pauses = max(
            1,
            math.ceil(
                total_angle / self.stop_angle
            ),
        )

        expected_rotation_time = (
            total_angle / self.spin_speed
        )

        expected_pause_time = (
            total_pauses
            * self.recognition_pause
        )

        actual_timeout = max(
            timeout,
            expected_rotation_time
            + expected_pause_time
            + 15.0,
        )

        scan_deadline = (
            time.monotonic()
            + actual_timeout
        )

        accumulated_total = 0.0

        self.get_logger().info(
            "Начинаю сканирование точки интереса:"
        )

        self.get_logger().info(
            f"  оборотов: {rotations:.1f}"
        )

        self.get_logger().info(
            f"  скорость: "
            f"{self.spin_speed:.2f} рад/с"
        )

        self.get_logger().info(
            f"  остановка каждые: "
            f"{math.degrees(self.stop_angle):.0f}°"
        )

        self.get_logger().info(
            f"  ожидание на остановке: "
            f"{self.recognition_pause:.1f} с"
        )

        self.get_logger().info(
            f"  подтверждений для выбора: "
            f"{self.required_confirmations}"
        )

        try:
            # Первая проверка до начала вращения.
            self.wait_for_recognition(
                pause_number=0,
                total_pauses=total_pauses,
                recognize_code=recognize_code,
            )

            for pause_index in range(
                1,
                total_pauses + 1,
            ):
                if (
                    not rclpy.ok()
                    or time.monotonic() >= scan_deadline
                ):
                    break

                remaining = (
                    total_angle
                    - accumulated_total
                )

                if remaining <= 0.01:
                    break

                step_target = min(
                    self.stop_angle,
                    remaining,
                )

                step_timeout = max(
                    step_target / self.spin_speed * 2.0
                    + 3.0,
                    8.0,
                )

                rotated = self.rotate_one_step(
                    target_angle=step_target,
                    timeout=step_timeout,
                )

                accumulated_total += rotated

                self.get_logger().info(
                    "Поворот выполнен: "
                    f"{math.degrees(accumulated_total):.0f}°/"
                    f"{math.degrees(total_angle):.0f}°"
                )

                self.wait_for_recognition(
                    pause_number=pause_index,
                    total_pauses=total_pauses,
                    recognize_code=recognize_code,
                )

        finally:
            self.search_active = False
            self.stop_robot()

            if (
                light_enabled
                and not self.keep_light_on
            ):
                self.set_light(False)

        if accumulated_total + math.radians(5.0) < total_angle:
            self.get_logger().warning(
                "Сканирование завершилось раньше:"
                f" {math.degrees(accumulated_total):.0f}°/"
                f"{math.degrees(total_angle):.0f}°"
            )
        else:
            self.get_logger().info(
                "Два полных оборота завершены"
            )

        if recognize_code:
            if self.selected_code is not None:
                self.get_logger().info(
                    "Выбран подтверждённый код: "
                    f"{self.selected_code}"
                )
            else:
                self.get_logger().warning(
                    "Не получено двух одинаковых "
                    "распознаваний"
                )

        return self.selected_code

    # ========================================================
    # ОСТАНОВКА
    # ========================================================

    def stop_robot(self) -> None:
        """
        Безопасно отправляет нулевую скорость.

        При Ctrl+C контекст ROS может уже завершаться, поэтому
        публикация и spin защищены от RCLError.
        """
        if not rclpy.ok():
            return

        stop = Twist()

        for _ in range(6):
            if not rclpy.ok():
                break

            try:
                self.velocity_publisher.publish(stop)
                rclpy.spin_once(
                    self,
                    timeout_sec=0.03,
                )
            except Exception:
                # Контекст мог закрыться между rclpy.ok()
                # и вызовом publish().
                break

    def cancel_active_goals(self) -> None:
        if not rclpy.ok():
            return

        try:
            if self.active_navigation_goal is not None:
                future = (
                    self.active_navigation_goal
                    .cancel_goal_async()
                )

                rclpy.spin_until_future_complete(
                    self,
                    future,
                    timeout_sec=3.0,
                )

                self.active_navigation_goal = None

            self.stop_robot()

        except Exception as error:
            # Не создаём второй traceback при завершении ROS.
            try:
                self.get_logger().warning(
                    "Не удалось полностью отменить цель "
                    f"при завершении: {error}"
                )
            except Exception:
                pass

    # ========================================================
    # МИССИЯ
    # ========================================================

    def execute(
        self,
        route: Route,
        plan_only: bool,
        skip_code_search: bool,
        rotations: float,
        search_timeout: float,
    ) -> int:
        if not self.wait_for_system():
            return 1

        if not self.validate_waypoints(route):
            return 2

        self.get_logger().info(
            f"Выбран: {route.name}"
        )

        for index, point in enumerate(
            route.points,
            start=1,
        ):
            self.get_logger().info(
                f"  {index}: {point.name} — "
                f"({point.x:.2f}, {point.y:.2f})"
            )

        if not self.wait_for_good_localization():
            return 3

        full_path = self.preplan_route(route)

        if full_path is None:
            return 4

        if plan_only:
            print()
            print("=" * 60)
            print("МАРШРУТ ПО ЛАБИРИНТУ УСПЕШНО ПОСТРОЕН")
            print(
                f"Промежуточных точек: "
                f"{len(route.points)}"
            )
            print(
                f"Общая длина: "
                f"{path_length(full_path):.2f} м"
            )
            print(
                "Полный путь опубликован: "
                "/mission/planned_path"
            )
            print("=" * 60)
            return 0

        self.clear_costmaps()

        if not self.navigate_route(route):
            return 5

        code = self.perform_interest_scan(
            rotations=rotations,
            timeout=search_timeout,
            recognize_code=not skip_code_search,
        )

        print()
        print("=" * 60)

        if skip_code_search:
            print(
                "РЕЗУЛЬТАТ: два оборота "
                "с остановками выполнены"
            )
            print("=" * 60)
            return 0

        if code is not None:
            print(
                "РЕЗУЛЬТАТ: подтверждённый код "
                f"{code}"
            )
            print("=" * 60)
            return 0

        print(
            "РЕЗУЛЬТАТ: не получено двух "
            "одинаковых распознаваний"
        )
        print("=" * 60)
        return 6


def choose_route() -> str:
    print("Выбери маршрут:")
    print(
        "  1 — к ближней жёлтой метке "
        "через 3 точки"
    )
    print(
        "  2 — к верхней левой метке "
        "через 6 точек"
    )

    while True:
        value = input("Маршрут [1/2]: ").strip()

        if value in ROUTES:
            return value

        print("Нужно ввести 1 или 2.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Навигация SVERK Rover по известной карте "
            "лабиринта через безопасные промежуточные точки."
        )
    )

    parser.add_argument(
        "--point",
        choices=sorted(ROUTES),
        help=(
            "Конечная точка маршрута. "
            "Если не задана, появится выбор."
        ),
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help=(
            "Проверить все сегменты и построить полный путь "
            "без движения."
        ),
    )

    parser.add_argument(
        "--skip-code-search",
        action="store_true",
        help=(
            "Не распознавать код. Подсветка, остановки "
            "и два оборота всё равно выполняются."
        ),
    )

    parser.add_argument(
        "--planner-id",
        default="",
        help=(
            "ID глобального планировщика. "
            "Пустое значение использует планировщик "
            "из конфигурации Nav2."
        ),
    )

    parser.add_argument(
        "--waypoint-timeout",
        type=float,
        default=90.0,
        help=(
            "Таймаут одного участка маршрута, секунд."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help=(
            "Количество повторных попыток участка "
            "после очистки costmap."
        ),
    )

    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=0.22,
        help=(
            "Расстояние, при котором промежуточная точка "
            "считается уже достигнутой."
        ),
    )

    parser.add_argument(
        "--hold-time",
        type=float,
        default=0.6,
        help=(
            "Пауза в промежуточной точке для стабилизации AMCL."
        ),
    )

    parser.add_argument(
        "--position-std-limit",
        type=float,
        default=0.45,
        help=(
            "Максимальное стандартное отклонение позиции AMCL."
        ),
    )

    parser.add_argument(
        "--yaw-std-limit",
        type=float,
        default=35.0,
        help=(
            "Максимальное стандартное отклонение yaw AMCL, "
            "градусы."
        ),
    )

    parser.add_argument(
        "--require-amcl",
        action="store_true",
        help=(
            "Запретить движение, если /amcl_pose "
            "не публикуется."
        ),
    )

    parser.add_argument(
        "--rotations",
        type=float,
        default=2.0,
        help=(
            "Количество полных оборотов. "
            "По умолчанию: 2."
        ),
    )

    parser.add_argument(
        "--spin-speed",
        type=float,
        default=0.20,
        help=(
            "Медленная скорость вращения, рад/с. "
            "По умолчанию: 0.20."
        ),
    )

    parser.add_argument(
        "--stop-angle",
        type=float,
        default=90.0,
        help=(
            "Через сколько градусов останавливаться "
            "для распознавания. По умолчанию: 90."
        ),
    )

    parser.add_argument(
        "--recognition-pause",
        type=float,
        default=6.0,
        help=(
            "Сколько секунд стоять на каждой остановке. "
            "По умолчанию: 6."
        ),
    )

    parser.add_argument(
        "--required-confirmations",
        type=int,
        default=2,
        help=(
            "Сколько одинаковых распознаваний нужно "
            "для выбора кода. Минимум: 2."
        ),
    )

    parser.add_argument(
        "--search-timeout",
        type=float,
        default=180.0,
        help=(
            "Общий таймаут сканирования, секунд."
        ),
    )

    parser.add_argument(
        "--light-topic",
        default="/led_strip/set_state",
        help=(
            "Управляющий ROS-топик подсветки. "
            "По умолчанию: /led_strip/set_state. "
            "/led_strip/state является только "
            "выходом состояния."
        ),
    )

    parser.add_argument(
        "--light-type",
        choices=(
            "auto",
            "ledstrip",
            "bool",
            "int",
            "uint8",
            "string",
        ),
        default="ledstrip",
        help=(
            "Тип сообщения топика подсветки. "
            "По умолчанию: "
            "rover_interfaces/msg/LedStripState."
        ),
    )

    parser.add_argument(
        "--keep-light-on",
        action="store_true",
        help=(
            "Оставить подсветку включённой "
            "после завершения."
        ),
    )

    parser.add_argument(
        "--light-settle-time",
        type=float,
        default=0.8,
        help=(
            "Пауза после включения подсветки."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    route_key = (
        arguments.point
        if arguments.point is not None
        else choose_route()
    )

    route = ROUTES[route_key]

    print()
    print("Система координат:")
    print("  x > 0 — вниз")
    print("  y > 0 — вправо")
    print()
    print(
        "Конечная точка смещена по X на +0.15 м"
    )
    print(
        "После прибытия: подсветка, два оборота, "
        "остановки и двойное подтверждение OCR"
    )
    print()
    print(route.name)

    for index, waypoint in enumerate(
        route.points,
        start=1,
    ):
        print(
            f"  {index}. {waypoint.name}: "
            f"x={waypoint.x:.2f}, "
            f"y={waypoint.y:.2f}"
        )

    rclpy.init(
        signal_handler_options=SignalHandlerOptions.NO,
    )

    node = LabyrinthWaypointMission(
        planner_id=arguments.planner_id,
        waypoint_timeout=max(
            arguments.waypoint_timeout,
            15.0,
        ),
        retries=max(
            arguments.retries,
            0,
        ),
        waypoint_tolerance=max(
            arguments.waypoint_tolerance,
            0.05,
        ),
        hold_time=max(
            arguments.hold_time,
            0.0,
        ),
        position_std_limit=max(
            arguments.position_std_limit,
            0.05,
        ),
        yaw_std_limit_deg=max(
            arguments.yaw_std_limit,
            5.0,
        ),
        require_amcl=arguments.require_amcl,
        spin_speed=max(
            abs(arguments.spin_speed),
            0.05,
        ),
        stop_angle_deg=max(
            arguments.stop_angle,
            10.0,
        ),
        recognition_pause=max(
            arguments.recognition_pause,
            0.5,
        ),
        required_confirmations=max(
            arguments.required_confirmations,
            2,
        ),
        light_topic=arguments.light_topic,
        light_type=arguments.light_type,
        keep_light_on=arguments.keep_light_on,
        light_settle_time=max(
            arguments.light_settle_time,
            0.0,
        ),
    )

    try:
        exit_code = node.execute(
            route=route,
            plan_only=arguments.plan_only,
            skip_code_search=(
                arguments.skip_code_search
            ),
            rotations=max(
                arguments.rotations,
                0.25,
            ),
            search_timeout=max(
                arguments.search_timeout,
                10.0,
            ),
        )
    except KeyboardInterrupt:
        try:
            node.get_logger().warning(
                "Миссия прервана пользователем"
            )
        except Exception:
            pass

        node.cancel_active_goals()
        exit_code = 130

    finally:
        try:
            if rclpy.ok():
                node.stop_robot()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        if rclpy.ok():
            rclpy.shutdown()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
