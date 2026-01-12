import argparse
import time
import subprocess
from functools import lru_cache

import cv2
import numpy as np

from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

# -------------------- AUDIO FILE PATHS --------------------
AUDIO_STOP = "/home/pi/audio/stop.wav"
AUDIO_GO = "/home/pi/audio/go.wav"
AUDIO_IDLE = "/home/pi/audio/idle.wav"

AUDIO_COOLDOWN = 3.0  # seconds (prevents looping)
last_audio_time = 0

last_results = []


# -------------------- AUDIO PLAYER --------------------
def play_audio(path):
    """Play WAV file via Bluetooth speaker using aplay"""
    subprocess.Popen(
        ["aplay", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )


# -------------------- Detection Object --------------------
class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


# -------------------- Parse Detections --------------------
def parse_detections(metadata):
    global last_results

    threshold = args.threshold
    outputs = imx500.get_outputs(metadata, add_batch=True)

    if outputs is None:
        return last_results

    boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    boxes = np.array_split(boxes, 4, axis=1)
    boxes = zip(*boxes)

    last_results = [
        Detection(box, cls, score, metadata)
        for box, score, cls in zip(boxes, scores, classes)
        if score > threshold
    ]
    return last_results


# -------------------- Labels --------------------
@lru_cache
def get_labels():
    return intrinsics.labels


# -------------------- Decision Logic --------------------
def decide_state(detections, labels):
    detected = {labels[int(d.category)] for d in detections}

    if "pedestrian_on_zebra" in detected:
        return (
            "Please don't go, Pedestrian crossing zebra",
            (0, 0, 255),      # Red
            True,
            AUDIO_STOP
        )

    if "pedestrian_off_zebra" in detected:
        return (
            "Pedestrian waiting on zebra, You can Go",
            (0, 255, 0),      # Green
            True,
            AUDIO_GO
        )

    return (
        "No pedestrian, You can Go",
        (0, 0, 0),          # Black
        False,
        AUDIO_IDLE
    )


# -------------------- Draw Callback --------------------
def draw_detections(request, stream="main"):
    global last_audio_time

    detections = last_results
    labels = get_labels()

    with MappedArray(request, stream) as m:

        message, bg_color, show_boxes, audio_file = decide_state(detections, labels)

        # ---------- Draw pedestrian bounding boxes ----------
        if show_boxes:
            for det in detections:
                x, y, w, h = det.box
                cv2.rectangle(
                    m.array,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

        # ---------- Message box ----------
        overlay = m.array.copy()
        cv2.rectangle(
            overlay,
            (0, 0),
            (m.array.shape[1], 60),
            bg_color,
            -1
        )

        cv2.addWeighted(overlay, 0.85, m.array, 0.15, 0, m.array)

        cv2.putText(
            m.array,
            message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # ---------- Audio (rate-limited) ----------
        now = time.time()
        if now - last_audio_time > AUDIO_COOLDOWN:
            play_audio(audio_file)
            last_audio_time = now


# -------------------- Arguments --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--fps", type=int, default=25)
    return parser.parse_args()


# -------------------- Main --------------------
if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"

    with open(args.labels, "r") as f:
        intrinsics.labels = f.read().splitlines()

    intrinsics.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls={"FrameRate": args.fps},
        buffer_count=12
    )

    picam2.pre_callback = draw_detections

    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)

    while True:
        last_results = parse_detections(picam2.capture_metadata())
