import cv2
from deepface import DeepFace

class EmotionAnalyzer:
    def __init__(self):
        # DeepFace handles model loading on its first analyze call, 
        # but we encapsulate here to keep the architecture clean.
        pass

    def analyze_frame(self, frame):
        """
        Analyzes a single BGR frame for faces and emotions.
        Returns the processed frame (with bounding boxes) and a list of detected emotions.
        """
        emotions_detected = []
        try:
            # analyze the frame. enforce_detection=False ensures we don't crash if no face is found
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            # DeepFace.analyze returns a dict for 1 face or list of dicts for >1 face
            if isinstance(results, dict):
                results = [results]
                
            for res in results:
                # Get the dominant emotion
                dominant_emotion = res.get('dominant_emotion')
                if dominant_emotion:
                    emotions_detected.append(dominant_emotion)
                
                # Draw bounding box and label
                region = res.get('region')
                if region:
                    x_box, y_box, w_box, h_box = region['x'], region['y'], region['w'], region['h']
                    cv2.rectangle(frame, (x_box, y_box), (x_box + w_box, y_box + h_box), (0, 255, 0), 2)
                    cv2.putText(frame, dominant_emotion, (x_box, y_box - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
        except Exception as e:
            # Catching exceptions from DeepFace (usually when it fails completely to process)
            pass
            
        return frame, emotions_detected
