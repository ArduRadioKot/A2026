# A2026 Competition Projects

This repository contains three independent robotics and autonomous-systems projects developed for competitions:

- [`anpa/`](anpa/) - autonomous-flight, ArUco, laser communication, video streaming, and AI experiments.
- [`geoscan/`](geoscan/) - camera navigation, ArUco and YOLO detection, video streaming, and flight-control prototypes.
- [`sverk/`](sverk/) - separate drone, rover, and YOLO computer-vision experiments.

Each directory is a self-contained project archive. The projects do not share one common launcher or one universal dependency list. Hardware interfaces, camera settings, model paths, and runtime requirements may differ between scripts.

## Documentation

- [ANPA documentation](anpa/README.md) | [Russian version](anpa/README_RU.md)
- [Geoscan documentation](geoscan/README.md) | [Russian version](geoscan/README_RU.md)
- [Sverk documentation](sverk/README.md) | [Russian version](sverk/README_RU.md)

## General notes

The code is preserved as competition and research material. Before running a script, inspect its imports and configuration and connect only the required hardware. Flight and movement programs must be tested in a controlled area with a manual emergency stop.

Most scripts can be started directly with Python:

```bash
python path/to/script.py
```

The exact Python packages, ROS environment, camera, robot, drone, and trained models depend on the selected script.
