#!/usr/bin/env python
"""
Driver Drowsiness & Attention Monitoring System (Integrated)
Features:
 - Multi-Cascade Face Detection (Frontal + Alt2 + Profile for full head turning)
 - 68 Facial Landmarks (FacemarkLBF)
 - Eye Aspect Ratio (EAR) & Deep Learning CNN (bestModel.onnx) for Eye Tracking
 - Mouth Aspect Ratio (MAR) for Yawning Detection
 - 3D Head Pose (Yaw): Detects turning head Left / Right -> "Distracted"
 - Alert Logic:
     * When eyes are closed for > 2.0 seconds: "Not Awake"
     * When turning head Left or Right: "Distracted"
     * When yawning occurs: "Yawn Detected"
     * When eyes closed + yawning + head turned/tilt occur together: "Driver Drowsy"
"""

import sys
from pathlib import Path

# Add Driver-Drowsiness-Detection to sys.path so EAR and MAR can be imported directly
BASE_DIR = Path(__file__).resolve().parent.parent
CLONED_DIR = BASE_DIR / "Driver-Drowsiness-Detection"
if str(CLONED_DIR) not in sys.path:
    sys.path.insert(0, str(CLONED_DIR))

import cv2
import numpy as np
import math
import time
import onnxruntime as ort

from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio

# Landmark indices for 68-point model (0-indexed)
LEFT_EYE_IDXS = slice(42, 48)   # landmarks 43-48
RIGHT_EYE_IDXS = slice(36, 42)  # landmarks 37-42
MOUTH_IDXS = slice(48, 68)      # landmarks 49-68

# 3D Head Model Points (for Head Pose Estimation)
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose tip (index 33)
    (0.0, -330.0, -65.0),        # Chin (index 8)
    (-225.0, 170.0, -135.0),     # Left eye left corner (index 36)
    (225.0, 170.0, -135.0),      # Right eye right corner (index 45)
    (-150.0, -150.0, -125.0),    # Left mouth corner (index 48)
    (150.0, -150.0, -125.0)      # Right mouth corner (index 54)
], dtype="double")

# Thresholds:
# 1. Eye Closure: EAR < 0.18. At ~30 FPS, > 2 seconds is >= 50-60 consecutive frames.
EYE_AR_THRESH = 0.18
EYES_CLOSED_TIME_THRESH = 2.0  # seconds

# 2. Yawning: MAR > 0.65 for sustained duration
MOUTH_AR_THRESH = 0.65
YAWN_SUSTAINED_FRAMES = 12

# 3. Head Turning Thresholds (Yaw Degrees for left/right distraction)
YAW_DISTRACT_THRESH = 20.0     # Turn left/right > 20 deg -> Distracted
HEAD_TILT_THRESH = 25.0        # General tilt/turn angle

CNN_MODEL_PATH = BASE_DIR / "models" / "bestModel.onnx"
CASCADE_FRONTAL = BASE_DIR / "models" / "cascades" / "haarcascade_frontalface_default.xml"
CASCADE_ALT2 = BASE_DIR / "models" / "cascades" / "haarcascade_frontalface_alt2.xml"
CASCADE_PROFILE = BASE_DIR / "models" / "cascades" / "haarcascade_profileface.xml"
LBF_MODEL_PATH = BASE_DIR / "models" / "lbfmodel.yaml"


def compute_head_pose(shape, img_shape):
    """
    Computes accurate 3D Head Pose (Pitch, Yaw, Roll) using 6 key facial landmarks.
    """
    h, w = img_shape[:2]
    focal_length = w
    center = (w / 2.0, h / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1), dtype="double")

    image_points = np.array([
        shape[33],  # nose tip
        shape[8],   # chin
        shape[36],  # left eye corner
        shape[45],  # right eye corner
        shape[48],  # mouth left corner
        shape[54]   # mouth right corner
    ], dtype="double")

    ok, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not ok:
        return 0.0, 0.0, 0.0, None

    rmat, _ = cv2.Rodrigues(rvec)

    # Compute Euler angles
    sy = math.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    if sy >= 1e-6:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    pitch_deg = float(np.rad2deg(pitch))
    yaw_deg = float(np.rad2deg(yaw))
    roll_deg = float(np.rad2deg(roll))

    # Project a 3D pointer forward from the nose tip
    (nose_end_2d, _) = cv2.projectPoints(
        np.array([(0.0, 0.0, 500.0)]),
        rvec, tvec, camera_matrix, dist_coeffs
    )
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_2d[0][0][0]), int(nose_end_2d[0][0][1]))

    return pitch_deg, yaw_deg, roll_deg, (p1, p2)


def main():
    print("=" * 65)
    print("DRIVER DROWSINESS & ATTENTION MONITORING SYSTEM")
    print("=" * 65)

    # 1. Load Face Detectors (Frontal + Alt2 + Profile for wide angles)
    print("[INFO] Loading Face Detectors (Frontal + Profile)...")
    face_frontal = cv2.CascadeClassifier(str(CASCADE_FRONTAL))
    face_alt2 = cv2.CascadeClassifier(str(CASCADE_ALT2))
    face_profile = cv2.CascadeClassifier(str(CASCADE_PROFILE))

    # 2. Load 68-point Facemark model
    print("[INFO] Loading 68-point Facial Landmark Predictor...")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(str(LBF_MODEL_PATH))

    # 3. Load CNN Model
    session = None
    input_name = None
    output_name = None
    if CNN_MODEL_PATH.exists():
        print(f"[INFO] Loading Deep Learning CNN: {CNN_MODEL_PATH.name}")
        session = ort.InferenceSession(str(CNN_MODEL_PATH), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

    # 4. Open Camera
    print("[INFO] Initializing Camera (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open camera.")
            return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)

    print("\n[READY] Camera live! Press 'q' or 'ESC' to exit.")

    # Camera warm-up read
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.05)

    frame_width = 1024
    frame_height = 576
    
    # Timing & Event Tracking
    eye_closed_start_time = None
    eyes_closed_duration = 0.0
    
    yawn_counter = 0
    recent_yawn_timestamp = 0.0
    
    distracted_counter = 0
    recent_turn_timestamp = 0.0

    fps = 0.0
    prev_time = time.time()
    empty_frame_count = 0

    # Baseline head pose calibration (auto-adapts smoothly)
    baseline_yaw = None

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret or frame is None:
            empty_frame_count += 1
            if empty_frame_count > 30:
                print("[WARNING] Camera stream stopped responding.")
                break
            time.sleep(0.01)
            continue

        empty_frame_count = 0

        frame = cv2.resize(frame, (frame_width, frame_height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame.shape[:2]

        # Calculate FPS
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        # Multi-Cascade Face Detection
        faces = face_frontal.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            faces = face_alt2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
        
        is_profile_face = False
        profile_direction = "SIDE"
        if len(faces) == 0:
            # Detect right profile
            faces = face_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
            if len(faces) > 0:
                is_profile_face = True
                profile_direction = "RIGHT"
            else:
                # Flip horizontally to detect left profile
                flipped_gray = cv2.flip(gray, 1)
                p_faces = face_profile.detectMultiScale(flipped_gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
                if len(p_faces) > 0:
                    faces = np.array([[w - fx - fw, fy, fw, fh] for (fx, fy, fw, fh) in p_faces])
                    is_profile_face = True
                    profile_direction = "LEFT"

        # State flags for current frame
        is_eyes_closed_now = False
        is_yawning_now = False
        is_turned_now = False
        turned_direction = ""

        ear = 0.30
        mar = 0.20
        delta_yaw = 0.0
        cnn_awake_prob = 1.0

        if len(faces) > 0:
            # Sort by size and take the primary face
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[:1]
            (fx, fy, fw, fh) = faces[0]
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 2)

            if is_profile_face:
                # Steep turn away from camera
                is_turned_now = True
                turned_direction = profile_direction
                recent_turn_timestamp = curr_time
            else:
                # Fit 68 landmarks
                ok, landmarks_list = facemark.fit(gray, np.array(faces))
                if ok and len(landmarks_list) > 0:
                    shape = np.array(landmarks_list[0]).reshape(-1, 2)
                    if shape.shape[0] >= 68:
                        # 1. EYE TRACKING (EAR + CNN)
                        leftEye = shape[LEFT_EYE_IDXS].astype(int)
                        rightEye = shape[RIGHT_EYE_IDXS].astype(int)
                        leftEAR = eye_aspect_ratio(leftEye)
                        rightEAR = eye_aspect_ratio(rightEye)
                        ear = (leftEAR + rightEAR) / 2.0

                        # Draw eye contours
                        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
                        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

                        # CNN Eye Verification
                        if session is not None:
                            all_eye_pts = np.vstack([leftEye, rightEye])
                            ex, ey, ew, eh = cv2.boundingRect(all_eye_pts)
                            pad = 12
                            x1, y1 = max(0, ex - pad), max(0, ey - pad)
                            x2, y2 = min(w, ex + ew + pad), min(h, ey + eh + pad)
                            if x2 > x1 and y2 > y1:
                                eye_roi = gray[y1:y2, x1:x2]
                                resized = cv2.resize(eye_roi, (64, 64))
                                norm = (resized.astype(np.float32) / 255.0).reshape(1, 1, 64, 64)
                                pred = session.run([output_name], {input_name: norm})[0]
                                cnn_awake_prob = float(np.squeeze(pred))

                        # Considered closed if EAR is below threshold or CNN shows low awake probability
                        if ear < EYE_AR_THRESH or (ear < 0.20 and cnn_awake_prob < 0.20):
                            is_eyes_closed_now = True

                        # 2. YAWN DETECTION (MAR)
                        mouth = shape[MOUTH_IDXS].astype(int)
                        mar = mouth_aspect_ratio(mouth)
                        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 200, 0), 1)

                        if mar > MOUTH_AR_THRESH:
                            yawn_counter += 1
                            if yawn_counter >= YAWN_SUSTAINED_FRAMES:
                                is_yawning_now = True
                                recent_yawn_timestamp = curr_time
                        else:
                            yawn_counter = max(0, yawn_counter - 1)

                        # 3. 3D HEAD POSE (Turning Left / Right)
                        pitch, yaw, roll, line_pts = compute_head_pose(shape, (h, w))

                        if baseline_yaw is None:
                            baseline_yaw = yaw
                        else:
                            baseline_yaw = 0.98 * baseline_yaw + 0.02 * yaw

                        delta_yaw = abs(yaw - baseline_yaw)

                        # Draw nose pointer
                        if line_pts is not None:
                            cv2.line(frame, line_pts[0], line_pts[1], (255, 0, 0), 2)

                        if delta_yaw > YAW_DISTRACT_THRESH:
                            direction = "RIGHT" if (yaw - baseline_yaw) > 0 else "LEFT"
                            distracted_counter += 1
                            if distracted_counter >= 6:
                                is_turned_now = True
                                turned_direction = direction
                                recent_turn_timestamp = curr_time
                        else:
                            distracted_counter = max(0, distracted_counter - 1)

        # Update Eye Closure Duration (for > 2.0 seconds rule)
        if is_eyes_closed_now:
            if eye_closed_start_time is None:
                eye_closed_start_time = curr_time
            eyes_closed_duration = curr_time - eye_closed_start_time
        else:
            eye_closed_start_time = None
            eyes_closed_duration = 0.0

        eyes_not_awake = eyes_closed_duration >= EYES_CLOSED_TIME_THRESH

        # Fatigue memory window (within last 4 seconds)
        had_recent_yawn = (curr_time - recent_yawn_timestamp) < 4.0
        had_recent_turn = is_turned_now or ((curr_time - recent_turn_timestamp) < 3.0)

        # -------------------------------------------------------------
        # ALERT HIERARCHY LOGIC:
        # 1. "Driver Drowsy": All 3 conditions occur together / in sequence
        #    (Eyes closed + Yawn detected + Head turned/tilt)
        # 2. "Not Awake": Eyes closed for > 2.0 seconds
        # 3. "Distracted": Head turned left or right
        # 4. "Yawn Detected": Yawn active
        # 5. "STATUS: DRIVER ATTENTIVE & AWAKE"
        # -------------------------------------------------------------
        
        all_three_drowsy = (eyes_closed_duration >= 1.0 or eyes_not_awake) and had_recent_yawn and had_recent_turn

        cv2.rectangle(frame, (0, 0), (w, 52), (20, 20, 20), -1)

        if all_three_drowsy:
            # Critical Alert: All 3 combined
            cv2.putText(frame, "CRITICAL ALERT: DRIVER DROWSY!", (20, 37), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
        elif eyes_not_awake:
            # Eyes closed > 2 seconds
            cv2.putText(frame, f"ALERT: NOT AWAKE! ({eyes_closed_duration:.1f}s)", (20, 37), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)
        elif is_turned_now:
            # Turning left or right
            cv2.putText(frame, f"DISTRACTED: LOOKING {turned_direction}", (20, 37), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 69, 255), 2)
        elif is_yawning_now:
            # Yawning
            cv2.putText(frame, "YAWN DETECTED (SIGNS OF FATIGUE)", (20, 37), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "STATUS: DRIVER ATTENTIVE & AWAKE", (20, 37), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 0), 2)

        # Bottom Diagnostics Bar
        eye_timer_str = f"Eye Closed: {eyes_closed_duration:.1f}s" if eyes_closed_duration > 0.3 else "Eyes: OPEN"
        metrics_txt = f"EAR: {ear:.2f} | MAR: {mar:.2f} | Yaw: {delta_yaw:.1f} deg | CNN Eye: {cnn_awake_prob*100:.0f}% | {eye_timer_str}"
        cv2.putText(frame, metrics_txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 135, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Driver Attention & Drowsiness Monitoring", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session terminated.")

if __name__ == "__main__":
    main()
