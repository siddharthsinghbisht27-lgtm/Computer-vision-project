# Crowd Emotion Analyzer

A Computer Vision project that analyzes the behavior and response of a crowd (or audience) during a show by detecting their facial expressions from a video feed.

## Features
- Analyzes video files or live webcam feeds for faces.
- Detects the dominant emotion (Happy, Sad, Neutral, Surprised, Angry, Disgust, Fear) of each person in the frame using the `DeepFace` library.
- Overlays bounding boxes and emotion labels on the video feed.
- Generates a summary graph (timeline) of crowd emotions at the end of the session, helping performers/organizers gauge audience engagement.

## Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)

## Installation & Setup

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <your-repo-url>
   cd "CV project"
   ```

2. **Create a Virtual Environment (Recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On the first run, the DeepFace library will automatically download pre-trained weights for the emotion detection model to your `.deepface` folder.*

## Usage

You can run the script on your webcam or on a pre-recorded video of an audience.

### Run with Webcam
```bash
python main.py --source 0
```

### Run with a Video File
```bash
python main.py --source path_to_your_video.mp4
```

### Optional Arguments
- `--skip`: Number of frames to skip. Since deep learning models are computationally heavy, the script skips frames to maintain near real-time performance. Default is `5` frames.
  ```bash
  python main.py --source 0 --skip 15
  ```

### Controls
- Press **`q`** while on the video window to stop processing. The script will then end the feed, generate an engagement graph, and save it as `emotion_report.png` in the directory.

## Project Structure
- `main.py`: The entry point script that handles argument parsing, video IO, and coordinates processing.
- `emotion_analyzer.py`: Encapsulates the DeepFace ML model for detecting faces and emotions.
- `visualization.py`: Contains the logic to generate the timeline graphs using `matplotlib`.
- `requirements.txt`: Python package dependencies.
