import cv2
import time
import argparse
from emotion_analyzer import EmotionAnalyzer
from visualization import plot_emotion_timeline

def main(video_source=0, skip_frames=10):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'. Check if your webcam is connected or file path is correct.")
        return

    analyzer = EmotionAnalyzer()
    
    # To store statistics over time. Format: [{'time': 0.5, 'emotion': 'happy'}, ...]
    emotion_timeline = []
    
    frame_count = 0
    print("Starting video processing...")
    print("Press 'q' in the video window to quit early.")
    
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended.")
            break
            
        # Process every Nth frame to save compute and run in near real-time
        if frame_count % skip_frames == 0:
            processed_frame, emotions = analyzer.analyze_frame(frame)
            
            # Record current timestamp relative to start
            current_time = time.time() - start_time
            if emotions:
                for emp in emotions:
                    emotion_timeline.append({
                        'time': current_time,
                        'emotion': emp
                    })
        else:
            # We still need to display the feed when we are skipping frames
            processed_frame = frame
            
        # Display the resulting frame
        cv2.imshow('Crowd Emotion Analyzer (Press Q to quit)', processed_frame)
        
        frame_count += 1
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted processing.")
            break

    # Release handles
    cap.release()
    cv2.destroyAllWindows()
    
    print("Processing complete. Generating engagement report...")
    if emotion_timeline:
        plot_emotion_timeline(emotion_timeline, output_path="emotion_report.png")
    else:
        print("No emotions were detected during the session. Cannot generate graph.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze crowd emotions from video/webcam")
    parser.add_argument("--source", default=0, help="Video file path or webcam index (default: 0)")
    parser.add_argument("--skip", type=int, default=5, help="Number of frames to skip to run faster. Higher means faster but less data points (default: 5)")
    args = parser.parse_args()
    
    # if source is a digit string (like '0' or '1'), convert to int for webcam index
    src = int(args.source) if str(args.source).isdigit() else args.source
    
    main(video_source=src, skip_frames=args.skip)
