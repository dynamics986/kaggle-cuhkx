from __future__ import annotations

MODALITIES = ("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar")
VISUAL_MODALITIES = ("Depth_Color", "IR", "Thermal")
SENSOR_MODALITIES = ("Skeleton", "IMU", "Radar")
TRAIN_USERS = tuple([f"user{i}" for i in range(1, 10)] + [f"user{i}" for i in range(16, 25)])
NUM_CLASSES = 40

