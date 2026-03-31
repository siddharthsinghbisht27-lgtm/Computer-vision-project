import pandas as pd
import matplotlib.pyplot as plt

def plot_emotion_timeline(emotion_data, output_path="emotion_report.png"):
    """
    Takes a list of emotion dicts and plots a timeline chart.
    emotion_data: [{'time': 1.2, 'emotion': 'happy'}, {'time': 1.5, 'emotion': 'neutral'}, ...]
    """
    if not emotion_data:
        print("No emotion data to plot.")
        return

    df = pd.DataFrame(emotion_data)
    
    # We bucket time into 1-second intervals for smoother graphs
    df['time_bin'] = df['time'].astype(int)
    
    # Count occurrences of each emotion per time bin
    emotion_counts = df.groupby(['time_bin', 'emotion']).size().unstack(fill_value=0)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    for emotion in emotion_counts.columns: # Each column is a different emotion
        plt.plot(emotion_counts.index, emotion_counts[emotion], label=emotion, marker='o')
        
    plt.title('Crowd Emotion Timeline During the Show')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Number of Faces Detected')
    plt.legend(title='Emotions', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(output_path)
    print(f"Engagement report saved successfully to {output_path}")
    plt.show()
