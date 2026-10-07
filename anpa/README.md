# ANPA Competition Project

ANPA is an independent collection of experimental programs developed for robotics and autonomous-flight competitions. The project contains prototypes for drone control, camera streaming, ArUco marker processing, laser packet reception, and digit recognition.

## Project structure

- `auto-flyght/` - autonomous flight experiments, ArUco tracking, centering, and recovery routines.
- `ai/` - computer-vision and AI experiments, including digit detection and laser packet reception.
- `kit/` - integrated competition scripts and supporting computer-vision utilities.
- `anpa_polygon.json` - polygon or field configuration data.
- `avt.txt` - an experimental drone and video-stream implementation.
- `rnd.txt` - additional research and testing code.

## Main technologies

- Python 3
- OpenCV and NumPy
- ROS and `cv_bridge`
- Flask for MJPEG video streaming
- ArUco marker detection
- Optional YOLO/MNIST-related models and tools

## Running the code

This is a competition archive rather than a packaged application. The scripts have different hardware, ROS, camera, and model requirements. Read the selected script before running it and check its constants, imports, camera settings, and control limits.

Typical preparation:

1. Install Python dependencies used by the selected script.
2. Configure the required ROS environment when the script imports `rospy`.
3. Connect and test the camera, flight controller, or other hardware without propellers first.
4. Run the selected script directly, for example `python path/to/script.py`.

Do not run flight-control programs near people or obstacles without a safe test procedure and an independent manual override.

## Status

The code is preserved as competition and development material. Some files are prototypes, use local hardware libraries, or require environment-specific configuration.
