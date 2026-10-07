# Geoscan Competition Project

Geoscan is an independent project from robotics and autonomous-flight competitions. It focuses on camera-based navigation, ArUco marker detection, YOLO experiments, video streaming, and flight-control prototypes.

## Project structure

- `aruco_flight.py` - flight logic using ArUco markers.
- `aruco-yolo-photo.py` and `aruco-yolo-photo copy.py` - photo-processing experiments combining ArUco and YOLO.
- `arucophoto.py` - ArUco photo processing.
- `pioneer_yolo_aruco_flight.py` - Pioneer flight prototype using YOLO and ArUco data.
- `flight.py` and `flights-final/` - flight experiments and final-stage variants.
- `cam-yolo-t.py`, `camera_reglament.py`, and `camera_reglament_terminal.py` - camera tests and calibration/inspection utilities.
- `stream.py` - video-streaming experiments.
- `crop.py` and `testyolo.py` - image preparation and YOLO tests.
- `ToDo` - unfinished work and planned improvements.

## Main technologies

- Python 3
- OpenCV and NumPy
- ArUco markers
- YOLO object detection
- Camera and video-stream processing
- Pioneer/robotics control integrations where required by a script

## Running the code

The scripts are separate competition experiments and do not share one universal launcher. Select a script according to the required task, then inspect its imports and configuration before running it. Some scripts require a camera, a trained YOLO model, a robot, or a local control library.

Typical command:

```bash
python path/to/script.py
```

Test camera and vision scripts with recorded images or a stationary robot first. Flight and movement scripts must be tested only in a controlled area with a manual emergency stop.

## Status

This repository preserves competition prototypes and working variants. Configuration, hardware interfaces, and model paths may need to be adapted to the local setup.
