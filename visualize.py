import json
import matplotlib.pyplot as plt
import os
import argparse

def plot_metrics(metrics_file):
    if not os.path.exists(metrics_file):
        print(f"Error: {metrics_file} not found. Please ensure training has started and logged metrics.")
        return

    with open(metrics_file, "r") as f:
        log_history = json.load(f)

    steps_train = []
    loss_train = []
    
    steps_eval = []
    loss_eval = []
    wer_eval = []

    # Parse JSON history
    for entry in log_history:
        if "loss" in entry:
            steps_train.append(entry["step"])
            loss_train.append(entry["loss"])
        elif "eval_loss" in entry:
            steps_eval.append(entry["step"])
            loss_eval.append(entry["eval_loss"])
            if "eval_wer" in entry:
                wer_eval.append(entry["eval_wer"])

    # Create figure with 2 subplots (Loss and WER)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Training and Validation Loss
    ax1.plot(steps_train, loss_train, label='Training Loss', color='blue', alpha=0.6)
    if steps_eval:
        ax1.plot(steps_eval, loss_eval, label='Validation Loss', color='red', marker='o')
    
    ax1.set_title("Model Loss over Time")
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Word Error Rate
    if wer_eval:
        ax2.plot(steps_eval, wer_eval, label='Validation WER %', color='green', marker='s', linewidth=2)
        ax2.set_title("Word Error Rate (WER) over Time")
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("WER (%)")
        
        # Invert Y axis since lower WER is better
        ax2.invert_yaxis()
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
    else:
        ax2.text(0.5, 0.5, "No WER evaluation data yet...", ha='center', va='center')
        ax2.set_title("Word Error Rate (WER)")

    plt.tight_layout()
    output_image = "training_metrics.png"
    plt.savefig(output_image, dpi=300)
    print(f"Successfully generated visualization: {output_image}")
    
    # We don't block execution if running in headless server
    try:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Whisper Training Metrics")
    parser.add_argument("--file", type=str, default="./whisper-small-marathi/metrics_history.json", help="Path to the JSON log file")
    args = parser.parse_args()
    
    plot_metrics(args.file)
