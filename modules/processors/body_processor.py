import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import onnxruntime as ort
from insightface.app import FaceAnalysis

class BodyProcessor:
    def __init__(self, execution_provider='CoreMLExecutionProvider'):
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Use 1 for speed; set to 2 for higher quality
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # RemBG for segmentation
        self.rembg_session = None  # Lazy init

        # InsightFace for person/face detection - with fallback providers
        providers = [execution_provider, 'CPUExecutionProvider']
        try:
            self.app = FaceAnalysis(providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[WARN] InsightFace init failed: {e}. Using CPU fallback.")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # ONNX session for person detection (optional)
        self.ort_session = None

    def detect_persons(self, frame):
        h, w = frame.shape[:2]
        scale = 640 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        if self.ort_session:
            input_blob = cv2.dnn.blobFromImage(resized, 1.0/255, (640, 640), (0.485, 0.456, 0.406), swapRB=True)
            outputs = self.ort_session.run(None, {self.ort_session.get_inputs()[0].name: input_blob})
            detections = outputs[0][0]
            boxes = []
            for det in detections:
                conf = det[4]
                if conf > 0.5:
                    x, y, bw, bh = det[0:4]
                    box = [int((x - bw/2) * w / 640), int((y - bh/2) * h / 640), int(bw * w / 640), int(bh * h / 640)]
                    boxes.append(box)
            return boxes
        else:
            faces = self.app.get(frame)
            boxes = [face.bbox.astype(int).tolist() for face in faces]
            return boxes

    def estimate_pose(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        if results.pose_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            mask = results.segmentation_mask if results.segmentation_mask is not None else None
            return keypoints, mask
        return None, None

    def segment_body(self, frame):
        if self.rembg_session is None:
            from rembg import remove, new_session
            self.rembg_session = new_session('u2net_human_seg')
        output = remove(frame, session=self.rembg_session)
        mask = output[:, :, 3] / 255.0 if output.shape[2] == 4 else np.zeros((frame.shape[0], frame.shape[1]))
        return mask

    def warp_body(self, source_img, source_keypoints, target_keypoints):
        if source_keypoints is None or target_keypoints is None:
            return source_img
        # Key body points: shoulders (11,12), hips (23,24)
        src_pts = np.array([source_keypoints[i][:2] for i in [11,12,23,24]], dtype=np.float32)
        tgt_pts = np.array([target_keypoints[i][:2] for i in [11,12,23,24]], dtype=np.float32)
        M, _ = cv2.estimateAffinePartial2D(src_pts, tgt_pts)
        warped = cv2.warpAffine(source_img, M, (source_img.shape[1], source_img.shape[0]))
        return warped
