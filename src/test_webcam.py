import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
# Using the converted bestModel (from bestModel.h5)
MODEL_PATH = ROOT / "models" / "bestModel.onnx"
YUNET_PATH = ROOT / "models" / "face_detection_yunet_2023mar.onnx"

IMG_SIZE = 64
CLASSES = ["sleepy", "awake"]  # 0: sleepy, 1: awake (sigmoid output: P(awake))

def main():
    print("=" * 60)
    print("LIVE DROWSINESS DETECTION TEST")
    print("=" * 60)
    print(f"Loading ONNX Model: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 1. Initialize ONNX Runtime Session
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. Initialize YuNet Face Detector
    use_face_detector = YUNET_PATH.exists()
    face_detector = None
    if use_face_detector:
        print(f"Loading YuNet Face Detector: {YUNET_PATH}")
        # dummy initial size, updated on frame
        face_detector = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (640, 480), score_threshold=0.6)
    else:
        print("YuNet model not found. Using center-crop fallback.")

    # 3. Open Video Capture (DirectShow backend on Windows)
    print("\nAttempting to open camera using DirectShow (cv2.CAP_DSHOW)...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("DirectShow index 0 failed, trying default backend...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nStarting live feed! Press 'q' or 'ESC' to exit.")

    # Camera warm-up read
    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.05)

    fps = 0.0
    prev_time = time.time()
    drowsy_counter = 0
    DROWSY_THRESHOLD_FRAMES = 15  # alert after ~0.5 - 1s of consecutive sleepy frames
    empty_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            empty_frame_count += 1
            if empty_frame_count > 30:
                print("Warning: Camera stream stopped responding.")
                break
            time.sleep(0.01)
            continue

        empty_frame_count = 0

        h, w = frame.shape[:2]

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        eye_crops = []
        face_box = None

        if face_detector is not None:
            face_detector.setInputSize((w, h))
            _, faces = face_detector.detect(frame)
            if faces is not None and len(faces) > 0:
                # Pick largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                face = faces[0]
                fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                face_box = (fx, fy, fw, fh)

                # YuNet keypoints:
                # 4,5: right_eye (from subject's perspective) / left in image
                # 6,7: left_eye / right in image
                r_eye_x, r_eye_y = int(face[4]), int(face[5])
                l_eye_x, l_eye_y = int(face[6]), int(face[7])

                # Extract bounding boxes around each eye landmark
                box_radius = int(fw * 0.15)
                for ex, ey, name in [(r_eye_x, r_eye_y, "Right"), (l_eye_x, l_eye_y, "Left")]:
                    x1 = max(0, ex - box_radius)
                    y1 = max(0, ey - box_radius)
                    x2 = min(w, ex + box_radius)
                    y2 = min(h, ey + box_radius)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        eye_crops.append((crop, (x1, y1, x2, y2), name))

        # Fallback if no face detected: center region
        if len(eye_crops) == 0:
            cx, cy = w // 2, h // 2
            half_box = 80
            x1, y1, x2, y2 = max(0, cx - half_box), max(0, cy - half_box), min(w, cx + half_box), min(h, cy + half_box)
            crop = frame[y1:y2, x1:x2]
            eye_crops.append((crop, (x1, y1, x2, y2), "Center ROI"))

        # Draw Face Box if found
        if face_box is not None:
            fx, fy, fw, fh = face_box
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 200, 0), 2)
            cv2.putText(frame, "Face", (fx, max(20, fy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        overall_state = "awake"
        max_sleepy_prob = 0.0

        for crop, (x1, y1, x2, y2), label_name in eye_crops:
            # Preprocess to 64x64 grayscale, normalized [0, 1] as used by bestModel.h5
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            normalized = resized.astype(np.float32) / 255.0
            onnx_input = normalized.reshape(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

            # Inference (sigmoid output: 1.0 = awake, 0.0 = sleepy)
            outputs = session.run([output_name], {input_name: onnx_input})[0]
            awake_prob = float(np.squeeze(outputs))
            sleepy_prob = 1.0 - awake_prob

            pred_label = "awake" if awake_prob >= 0.5 else "sleepy"
            conf = awake_prob if pred_label == "awake" else sleepy_prob

            if sleepy_prob > max_sleepy_prob:
                max_sleepy_prob = sleepy_prob

            # Color: Green for awake, Red for sleepy
            color = (0, 0, 255) if pred_label == "sleepy" else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{pred_label} ({conf*100:.0f}%)",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

        # Drowsiness Logic with temporal smoothing
        if max_sleepy_prob > 0.5:
            drowsy_counter += 1
        else:
            drowsy_counter = max(0, drowsy_counter - 1)

        is_drowsy_alert = drowsy_counter >= DROWSY_THRESHOLD_FRAMES

        # Top Banner Status
        banner_color = (0, 0, 220) if is_drowsy_alert else ((0, 160, 255) if drowsy_counter > 5 else (0, 200, 0))
        cv2.rectangle(frame, (0, 0), (w, 55), (20, 20, 20), -1)

        status_text = "DROWSINESS ALERT! WAKE UP!" if is_drowsy_alert else ("EYES CLOSING..." if drowsy_counter > 5 else "DRIVER AWAKE")
        cv2.putText(frame, status_text, (20, 38), cv2.FONT_HERSHEY_DUPLEX, 0.9, banner_color, 2)

        # Stats on screen
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Sleepy Prob: {max_sleepy_prob*100:.1f}%", (w - 220, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Instructions
        cv2.putText(frame, "Press 'q' or 'ESC' to exit", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("QNX Drowsiness Detection - Live Webcam Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam session ended.")

if __name__ == "__main__":
    main()
