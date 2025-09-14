
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

    def get_body_part_keypoints(self):
        """Define keypoint groups for different body parts (excluding head)"""
        return {
            'torso': [11, 12, 23, 24],  # shoulders and hips
            'left_arm': [11, 13, 15],   # left shoulder, elbow, wrist
            'right_arm': [12, 14, 16],  # right shoulder, elbow, wrist
            'left_leg': [23, 25, 27],   # left hip, knee, ankle
            'right_leg': [24, 26, 28],  # right hip, knee, ankle
        }
    
    def create_body_part_mask(self, keypoints, part_name, image_shape):
        """Create a mask for a specific body part"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        part_keypoints = self.get_body_part_keypoints()[part_name]
        
        # Convert normalized keypoints to pixel coordinates
        points = []
        for idx in part_keypoints:
            if keypoints[idx][2] > 0.5:  # visibility threshold
                x = int(keypoints[idx][0] * w)
                y = int(keypoints[idx][1] * h)
                points.append([x, y])
        
        if len(points) >= 3:
            # Create a convex hull around the keypoints
            points = np.array(points)
            hull = cv2.convexHull(points)
            
            # Expand the hull slightly
            center = np.mean(hull, axis=0).astype(int)
            expanded_hull = []
            for point in hull:
                direction = point - center
                expanded_point = center + direction * 1.3  # Expand by 30%
                expanded_hull.append(expanded_point.astype(int))
            
            expanded_hull = np.array(expanded_hull)
            cv2.fillPoly(mask, [expanded_hull], 255)
        
        return mask / 255.0  # Normalize to 0-1
    
    def warp_body_part(self, source_img, source_keypoints, target_keypoints, part_name, target_shape):
        """Warp a specific body part"""
        if source_keypoints is None or target_keypoints is None:
            return cv2.resize(source_img, (target_shape[1], target_shape[0]))
            
        part_keypoint_indices = self.get_body_part_keypoints()[part_name]
        
        # Convert normalized keypoints to pixel coordinates
        src_h, src_w = source_img.shape[:2]
        tgt_h, tgt_w = target_shape[:2]
        
        src_pts = []
        tgt_pts = []
        
        for idx in part_keypoint_indices:
            if (source_keypoints[idx][2] > 0.5 and target_keypoints[idx][2] > 0.5):
                src_pts.append([source_keypoints[idx][0] * src_w, source_keypoints[idx][1] * src_h])
                tgt_pts.append([target_keypoints[idx][0] * tgt_w, target_keypoints[idx][1] * tgt_h])
        
        if len(src_pts) < 2:
            return cv2.resize(source_img, (tgt_w, tgt_h))
        
        src_pts = np.array(src_pts, dtype=np.float32)
        tgt_pts = np.array(tgt_pts, dtype=np.float32)
        
        try:
            if len(src_pts) >= 3:
                M, _ = cv2.estimateAffinePartial2D(src_pts, tgt_pts)
            else:
                # For 2 points, use similarity transform
                angle = np.arctan2(tgt_pts[1][1] - tgt_pts[0][1], tgt_pts[1][0] - tgt_pts[0][0]) - \
                       np.arctan2(src_pts[1][1] - src_pts[0][1], src_pts[1][0] - src_pts[0][0])
                scale = np.linalg.norm(tgt_pts[1] - tgt_pts[0]) / np.linalg.norm(src_pts[1] - src_pts[0])
                
                cos_a, sin_a = np.cos(angle) * scale, np.sin(angle) * scale
                M = np.array([[cos_a, -sin_a, tgt_pts[0][0] - (cos_a * src_pts[0][0] - sin_a * src_pts[0][1])],
                             [sin_a, cos_a, tgt_pts[0][1] - (sin_a * src_pts[0][0] + cos_a * src_pts[0][1])]], dtype=np.float32)
            
            if M is not None:
                warped = cv2.warpAffine(source_img, M, (tgt_w, tgt_h))
                return warped
        except Exception as e:
            print(f"[WARN] Body part {part_name} warping failed: {e}")
            
        return cv2.resize(source_img, (tgt_w, tgt_h))
    
    def process_body_parts(self, source_img, source_keypoints, target_frame, target_keypoints):
        """Process all body parts separately (excluding head)"""
        if source_keypoints is None or target_keypoints is None:
            return target_frame
        
        result = target_frame.copy()
        body_parts = self.get_body_part_keypoints()
        
        for part_name in body_parts.keys():
            # Create mask for this body part in target frame
            part_mask = self.create_body_part_mask(target_keypoints, part_name, target_frame.shape)
            
            if np.sum(part_mask) > 100:  # Only process if mask is substantial
                # Warp the source body part to match target
                warped_part = self.warp_body_part(source_img, source_keypoints, target_keypoints, part_name, target_frame.shape)
                
                # Blend the warped part into the result
                part_mask_3d = part_mask[..., np.newaxis]
                result = (warped_part * part_mask_3d + result * (1 - part_mask_3d)).astype(np.uint8)
        
        return result
