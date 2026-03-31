# Project Report: Crowd Emotion Analyzer

## 1. Problem Statement
During live performances, shows, or corporate presentations, it is often difficult for performers and event organizers to accurately gauge audience engagement in real-time. Simply relying on the volume of applause or delayed post-event surveys misses the nuanced emotional reactions occurring during the event. An automated way to analyze facial expressions within a crowd can provide continuous, un-biased metrics on audience sentiment.

## 2. Approach to Solving It
We developed a Computer Vision pipeline using Python that processes video feeds (live or pre-recorded). The pipeline involves two main steps:
1. **Face Detection & Emotion Recognition:** 
   We utilized the `DeepFace` library, which provides a robust wrapper around several state-of-the-art CNNs (Convolutional Neural Networks) for facial analysis. For every Nth frame in the video stream, the system locates faces and determines the dominant emotion of each person.
2. **Data Aggregation and Visualization:** 
   The detected emotions are mapped to timestamps relative to the start of the session. Using `pandas` and `matplotlib`, this data is aggregated to plot a timeline graph. This graph visualizes the frequency of different emotions (e.g., Happy, Neutral, Sad, Surprised) over the duration of the video.

## 3. Key Decisions Made
- **Pre-trained Network vs. Building from Scratch:** We decided to use a pre-trained robust model rather than building a basic CNN from scratch. Facial Expression Recognition (FER) requires substantial and complex datasets. Leveraging established open-source models ensures high accuracy and robustness, which is critical when analyzing diverse, unconstrained faces in a crowd.
- **Frame Skipping:** To handle video processing efficiently and simulate near real-time performance without requiring a specialized GPU setup, the `main.py` controller was designed to evaluate every Nth frame (controlled via a `--skip` parameter).
- **Separation of Concerns:** The code is split into logical modules (`main.py` for IO, `emotion_analyzer.py` for ML, `visualization.py` for reporting) to keep it clean and maintainable.

## 4. Challenges Faced
- **Compute Overhead:** Analyzing multiple faces simultaneously via deep learning can be computationally intensive, significantly slowing down video playback. Adjusting the frame-skip logic helped mitigate this issue locally, allowing the video stream to stay near real-time.
- **Variable Lighting and Angles:** In crowd videos, many faces are tilted, occluded, or poorly lit. Setting `enforce_detection=False` within the analyzer was critical to ensure the script doesn't crash when the face detector fails to find a high-confidence bounding box.

## 5. What Was Learned
- Gained hands-on experience integrating deep learning-based Computer Vision models into real-world applications.
- Learned how to structure object-oriented CV applications (separating ML logic, video IO, and visualization).
- Understood how to bridge ML predictions with data analysis tools (`pandas`/`matplotlib`) to create actionable insights and graphical reports.
