import time
import threading
import cv2

from pioneer_sdk import Pioneer
from pioneer_sdk import Camera
from pioneer_sdk import ImageViewer


# ============================================================
# НАСТРОЙКИ
# ============================================================

STREAM_NAME = "pioneer"

running = True


route = [
    (-1.0, 1.0, 1.5, 0.0),
    (-1.0, 1.0, 1.5, 0.0),
    (-1.0, 2.5, 1.5, 0.0),
    (-1.0, 4.0, 1.5, 0.0),

    (-1.6, 4.0, 1.5, 0.0),
    (-1.6, 2.5, 1.5, 0.0),
    (-1.6, 1.0, 1.5, 0.0),

    (-2.2, 1.0, 1.5, 0.0),
    (-2.2, 2.5, 1.5, 0.0),
    (-2.2, 4.0, 1.5, 0.0),


    (-2.8, 4.0, 1.7, 0.0),
    (-2.8, 2.5, 1.7, 0.0),
    (-2.8, 1.0, 1.7, 0.0),

    (-3.1, 1.0, 1.7, 0.0),
    (-3.1, 2.5, 1.7, 0.0),
    (-3.1, 4.0, 1.7, 0.0),


    (0.0, 0.0, 1.7, 0.0)
]


# ============================================================
# ВИДЕО ПОТОК
# ============================================================

def video_worker():

    global running

    camera = None
    viewer = None

    try:

        print("[VIDEO] Запуск камеры")

        camera = Camera()

        viewer = ImageViewer()

        print("[VIDEO] Стрим:")
        print(
            f"http://10.42.0.1:8889/{STREAM_NAME}"
        )


        while running:

            frame = camera.get_cv_frame(
                timeout=2.0
            )


            if frame is None:
                continue


            # FPS текст

            cv2.putText(
                frame,
                "PIONEER STREAM",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )


            viewer.imshow(
                STREAM_NAME,
                frame,
                fps=20
            )


    except Exception as e:

        print(
            "[VIDEO ERROR]",
            e
        )


    finally:

        try:
            camera.stop()

        except:
            pass


        try:
            viewer.close()

        except:
            pass



# ============================================================
# ПОЛЁТ
# ============================================================

pioneer = None


try:

    pioneer = Pioneer()


    # запуск видео отдельно

    video_thread = threading.Thread(
        target=video_worker,
        daemon=True
    )

    video_thread.start()


    time.sleep(2)


    print("Подключение к Pioneer...")


    pioneer.arm()

    print("ARM: OK")


    pioneer.takeoff()

    print("Взлёт...")


    time.sleep(2)



    # -------------------------------
    # маршрут
    # -------------------------------

    for x,y,z,yaw in route:


        print(
            f"Точка x={x} y={y} z={z}"
        )


        pioneer.go_to_local_point(
            x,
            y,
            z,
            yaw
        )


        while not pioneer.point_reached():

            pos = pioneer.get_local_position_lps()

            if pos:
                print(
                    "Позиция:",
                    [
                        round(v,2)
                        for v in pos
                    ]
                )


            time.sleep(0.12)



    print("Маршрут завершён")


    print("Посадка...")


    pioneer.land()


    while not pioneer.point_reached():

        time.sleep(0.1)



    pioneer.disarm()


    print("Миссия завершена")



except KeyboardInterrupt:


    print(
        "\nОстановка оператором"
    )


    if pioneer:


        try:

            print(
                "Аварийная посадка..."
            )

            pioneer.land()

            time.sleep(3)

            pioneer.disarm()


        except Exception as e:

            print(
                "Ошибка посадки:",
                e
            )




except Exception as e:


    print(
        "КРИТИЧЕСКАЯ ОШИБКА:",
        e
    )


    if pioneer:


        try:

            print(
                "Аварийная посадка..."
            )

            pioneer.land()

            time.sleep(3)

            pioneer.disarm()


        except Exception as e2:

            print(
                "Ошибка аварийной посадки:",
                e2
            )



finally:


    running = False


    try:

        print(
            "Закрытие соединения"
        )

        pioneer.close_connection()

    except:

        pass


    print(
        "Программа завершена"
    )