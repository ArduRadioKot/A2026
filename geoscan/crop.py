import cv2
import os

# Входной файл
video_path = "input.mov"

# Папка для кадров
output_folder = "frames"

# Желаемая частота кадров
target_fps = 23

# Создаем папку
os.makedirs(output_folder, exist_ok=True)

# Открываем видео
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка: не удалось открыть видео")
    exit()

# Исходная частота видео
original_fps = cap.get(cv2.CAP_PROP_FPS)

# Количество кадров
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

duration = total_frames / original_fps

print(f"Исходный FPS: {original_fps}")
print(f"Длительность: {duration:.2f} секунд")
print(f"Будет сохранено примерно: {int(duration * target_fps)} кадров")


frame_number = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Время текущего кадра
    current_time = frame_number / original_fps

    # Номер кадра для 23 fps
    target_frame = int(current_time * target_fps)

    # Проверяем, нужен ли этот кадр
    if target_frame == saved_count:
        filename = os.path.join(
            output_folder,
            f"frame_{saved_count:06d}.jpg"
        )

        cv2.imwrite(filename, frame)

        saved_count += 1

    frame_number += 1


cap.release()

print("Готово!")
print(f"Сохранено кадров: {saved_count}")