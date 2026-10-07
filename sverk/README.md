# Sverk Competition Project

Sverk is an independent robotics competition project. It contains separate experiments for a drone, a rover, and YOLO-based computer vision.

## Project structure

- `drone/` - drone-control programs and flight experiments.
- `roverr/` - rover-control programs, including LED-strip and movement experiments.
- `yolo/` - YOLO-related experiments and model files.
- `aruco_field_6x6_map.txt` - an ArUco 6x6 field map or marker-layout reference.

## Main technologies

- Python 3
- Robotics and vehicle-control integrations
- OpenCV and computer vision
- ArUco marker maps
- YOLO object detection

## Running the code

The folders contain independent competition scripts. There is no single project launcher. Before running a file, inspect its imports and hardware settings, then connect only the equipment required for that experiment.

Typical command:

```bash
python path/to/script.py
```

Run movement and flight programs only in a controlled test area with a manual emergency stop. Check model paths and camera configuration before starting YOLO experiments.

## Status

This project is preserved as competition and research material. Scripts may depend on local hardware, drivers, models, and environment-specific settings.
