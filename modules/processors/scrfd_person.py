# Download the SCRFD person detection ONNX model from the official InsightFace repo:
# https://github.com/deepinsight/insightface/tree/master/examples/person_detection
# Place the model in your models/ directory, e.g. models/scrfd_person_10g.onnx

import numpy as np
import cv2
import insightface

class SCRFDPersonDetector:
    def __init__(self, model_path="models/scrfd_person_2.5g.onnx", providers=["CPUExecutionProvider"]):
        # Use InsightFace's model zoo for proper SCRFD person detection
        self.detector = insightface.model_zoo.get_model(model_path)
        self.detector.prepare(0, nms_thresh=0.5, input_size=(640, 640))

    def detect(self, img):
        """
        Detect persons in image and return bounding boxes
        Returns: list of [x, y, width, height] boxes
        """
        try:
            bboxes, kpss = self.detector.detect(img)
            if len(bboxes) == 0:
                return []
            
            # Convert to [x, y, w, h] format
            boxes = []
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox[:4].astype(int)
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1) 
                x2 = min(img.shape[1], x2)
                y2 = min(img.shape[0], y2)
                w = x2 - x1
                h = y2 - y1
                if w > 0 and h > 0:
                    boxes.append([x1, y1, w, h])
            return boxes
        except Exception as e:
            print(f"[ERROR] SCRFD person detection failed: {e}")
            return []
