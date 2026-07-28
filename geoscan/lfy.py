from pioneer_sdk import Pioneer
import time

pioneer = None

try:
    pioneer = Pioneer()

    print("Подключение к Pioneer...")
    pioneer.arm()
    print("ARM: OK")

    pioneer.takeoff()
    print("Взлёт...")

    route = [
    (0.0, 1.0, 1.7, 0.0),

    (-1.6, 0.5, 1.8, 0.0),
    (-1.6, 1.0, 1.8, 0.0),
    (-1.6, 2.0, 1.8, 0.0),
    (-1.6, 2.5, 1.8, 0.0),
    (-1.6, 4.0, 1.8, 0.0),

    (-2.0, 4.0, 1.8, 0.0),
    (-2.0, 2.5, 1.8, 0.0),
    (-2.0, 2.0, 1.8, 0.0),
    (-2.0, 1.0, 1.8, 0.0),
    (-2.0, 0.5, 1.8, 0.0),

    (-2.4, 0.5, 1.8, 0.0),
    (-2.4, 1.0, 1.8, 0.0),
    (-2.4, 2.0, 1.8, 0.0),
    (-2.4, 2.5, 1.8, 0.0),
    (-2.4, 4.0, 1.8, 0.0),


    (-2.8, 4.0, 1.8, 0.0),
    (-2.8, 2.5, 1.8, 0.0),
    (-2.8, 2.0, 1.8, 0.0),
    (-2.8, 1.0, 1.8, 0.0),
    (-2.8, 0.5, 1.8, 0.0),

    (-3.1, 0.5, 1.8, 0.0),
    (-3.1, 1.0, 1.8, 0.0),
    (-3.1, 2.0, 1.8, 0.0),
    (-3.1, 2.5, 1.8, 0.0),
    (-3.1, 4.0, 1.8, 0.0),


    (0.0, 1.0, 1.7, 0.0)
    ]


    for x, y, z, yaw in route:
        print(f"Точка: x={x}, y={y}, z={z}, yaw={yaw}")

        pioneer.go_to_local_point(
            x,
            y,
            z,
            yaw
        )

        while not pioneer.point_reached():

            time.sleep(0.12)


    print("Посадка...")
    pioneer.land()

    while not pioneer.point_reached():
        time.sleep(0.05)

    pioneer.disarm()

    print("Миссия завершена.")


except KeyboardInterrupt:
    print("\nОстановка оператором")

    if pioneer is not None:
        try:
            print("Аварийная посадка...")
            pioneer.land()

            time.sleep(3)

            pioneer.disarm()

        except Exception as e:
            print(
                "Ошибка при аварийной посадке:",
                e
            )


except Exception as e:
    print("\nКРИТИЧЕСКАЯ ОШИБКА:", e)

    if pioneer is not None:
        try:
            print("Выполняется аварийная посадка...")

            pioneer.land()

            time.sleep(3)

            pioneer.disarm()

            print("Аппарат посажен.")

        except Exception as land_error:
            print(
                "Ошибка аварийной посадки:",
                land_error
            )


finally:
    print("Программа завершена.")