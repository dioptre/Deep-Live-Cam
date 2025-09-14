
import cv2
import numpy as np
import mediapipe as mp
from rembg import remove
import onnxruntime as ort
from insightface.app import FaceAnalysis
from modules.processors.scrfd_person import SCRFDPersonDetector

class BodyProcessor:
    def __init__(self, execution_provider='CoreMLExecutionProvider', scrfd_model_path='models/scrfd_person_2.5g.onnx'):
        # MediaPipe Pose - use static mode to avoid state issues
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        # RemBG for segmentation
        self.rembg_session = None  # Lazy init

        # SCRFD Person Detector (using InsightFace model zoo)
        try:
            self.person_detector = SCRFDPersonDetector(scrfd_model_path, providers=[execution_provider, 'CPUExecutionProvider'])
            print("[INFO] SCRFD person detector initialized successfully")
        except Exception as e:
            print(f"[WARN] SCRFDPersonDetector init failed: {e}. Falling back to InsightFace face detection.")
            self.person_detector = None
        # InsightFace for face detection fallback
        providers = [execution_provider, 'CPUExecutionProvider']
        try:
            self.app = FaceAnalysis(providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"[WARN] InsightFace init failed: {e}. Using CPU fallback.")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))

    def detect_persons(self, frame):
        if self.person_detector is not None:
            try:
                return self.person_detector.detect(frame)
            except Exception as e:
                print(f"[WARN] SCRFD person detection failed: {e}. Falling back to face detection.")
        # Fallback: use InsightFace face detection
        faces = self.app.get(frame)
        boxes = [face.bbox.astype(int).tolist() for face in faces]
        return boxes

    def estimate_pose(self, frame):
        # Create a new pose instance for each call to avoid state issues
        with self.mp_pose.Pose(
            static_image_mode=True,  # Use static mode to avoid state issues
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5
        ) as pose:
            # Scale to consistent size (640x640) to avoid MediaPipe dimension errors
            target_size = 640
            h, w = frame.shape[:2]
            
            # Resize frame to square for consistent processing
            scaled_frame = cv2.resize(frame, (target_size, target_size))
            rgb_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            
            results = pose.process(rgb_frame)
            if results.pose_landmarks:
                # Get normalized keypoints (0-1 range)
                keypoints = np.array([[lm.x, lm.y, lm.visibility] for lm in results.pose_landmarks.landmark])
                
                # Draw landmarks on original frame (scale back for visualization)
                self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                mask = results.segmentation_mask if results.segmentation_mask is not None else None
                return keypoints, mask
        return None, None

    def segment_body(self, frame):
        from rembg import remove, new_session
        if self.rembg_session is None:
            self.rembg_session = new_session('u2net_human_seg')
        output = remove(frame, session=self.rembg_session)
        mask = output[:, :, 3] / 255.0 if output.shape[2] == 4 else np.zeros((frame.shape[0], frame.shape[1]))
        return mask

    def warp_body(self, source_img, source_keypoints, target_keypoints, target_shape):
        if source_keypoints is None or target_keypoints is None:
            return source_img
            
        # Key body points: shoulders (11,12), hips (23,24)
        # Convert normalized keypoints to pixel coordinates for both source and target
        src_h, src_w = source_img.shape[:2]
        tgt_h, tgt_w = target_shape[:2]
        
        src_pts = np.array([
            [source_keypoints[i][0] * src_w, source_keypoints[i][1] * src_h] 
            for i in [11,12,23,24]
        ], dtype=np.float32)
        
        tgt_pts = np.array([
            [target_keypoints[i][0] * tgt_w, target_keypoints[i][1] * tgt_h] 
            for i in [11,12,23,24]
        ], dtype=np.float32)
        
        # Check if we have valid points (not all zeros/NaN)
        if np.any(np.isnan(src_pts)) or np.any(np.isnan(tgt_pts)):
            return cv2.resize(source_img, (tgt_w, tgt_h))  # Fallback: just resize
            
        try:
            M, _ = cv2.estimateAffinePartial2D(src_pts, tgt_pts)
            if M is not None:
                warped = cv2.warpAffine(source_img, M, (tgt_w, tgt_h))
                return warped
        except Exception as e:
            print(f"[WARN] Body warping failed: {e}")
            
        return cv2.resize(source_img, (tgt_w, tgt_h))  # Fallback: just resize
