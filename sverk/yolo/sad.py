from ultralytics import YOLO
from PIL import Image

model = YOLO("/home/nikita/Documents/GitHub/A2026/yolo26n_cheburashka.pt")
image_path = "/home/nikita/Documents/GitHub/A2026/sverk/yolo/photo_2026-07-31_17-45-39.jpg" 
results = model(image_path)  
results[0].show()  
results[0].save(filename="result.jpg") 