import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# Load models.
yolo_model = YOLO("face.pt")
cnn_model = load_model("emotions.h5")
class_labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "suprise"}

cap = cv2.VideoCapture(0)
DETECTION_INTERVAL = 5  # Run YOLO detection every 5 frames
frame_count = 0

# List to hold tracked faces.
# Each element is a dict with keys: 'bbox', 'template', 'yolo_label', and 'yolo_conf'
tracked_faces = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Precompute grayscale version of frame for template matching.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run YOLO detection every DETECTION_INTERVAL frames.
    if frame_count % DETECTION_INTERVAL == 0:
        results = yolo_model(frame, conf=0.6)
        tracked_faces = []  # Reset tracked faces on detection frame

        for result in results:
            if not result.boxes:
                continue

            boxes = result.boxes  # Boxes object with attributes xyxy, conf, and cls.
            # Iterate through all detected boxes.
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = box.astype(int)
                # Validate ROI boundaries.
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                face_roi = frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    continue
                # Create a grayscale template for tracking.
                face_template = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                # Retrieve YOLO confidence and class.
                yolo_conf = float(boxes.conf[i].cpu().numpy())
                yolo_cls = int(boxes.cls[i].cpu().numpy())
                yolo_label = result.names.get(yolo_cls, str(yolo_cls)) if hasattr(result, "names") else str(yolo_cls)

                tracked_faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'template': face_template,
                    'yolo_label': yolo_label,
                    'yolo_conf': yolo_conf
                })
    else:
        # Update tracked faces via template matching.
        for face in tracked_faces:
            template = face['template']
            res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            h, w = template.shape
            new_bbox = (top_left[0], top_left[1], top_left[0] + w, top_left[1] + h)
            
            # Update template if the new region is valid.
            new_roi = frame[new_bbox[1]:new_bbox[3], new_bbox[0]:new_bbox[2]]
            if new_roi.size != 0:
                face['template'] = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
            face['bbox'] = new_bbox

    # Process each tracked face for CNN emotion prediction.
    for face in tracked_faces:
        x1, y1, x2, y2 = face['bbox']
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0:
            continue

        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        try:
            face_resized = cv2.resize(face_gray, (48, 48))
        except Exception as e:
            continue  # Skip if resizing fails
        face_normalized = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_normalized, axis=[0, -1])  # add batch and channel dimensions
        prediction = cnn_model.predict(face_input)
        predicted_idx = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        emotion_label = class_labels.get(predicted_idx, str(predicted_idx))
        
        # Draw the bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Display the emotion prediction.
        cv2.putText(frame, f"Emotion: {emotion_label}", 
                    (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)
        # Display the YOLO detection label and its confidence.
        cv2.putText(frame, f"{face.get('yolo_label', 'face')}", 
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Real-Time Face Detection and Emotion Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
