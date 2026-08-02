import os
import sys
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

def main():
    print("=" * 60)
    print("  GOMI — Local Kaggle Auto-Labeler")
    print("=" * 60)
    
    # 1. Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "..", "..", "gomi-datasets", "unlabeled", "chunks", "unlabeled_kaggle_100k_chunk_1.csv")
    output_file = os.path.join(script_dir, "datasets", "kaggle_100k_labeled_1.csv")
    
    if not os.path.isfile(input_file):
        print(f"Error: Could not find {input_file}")
        sys.exit(1)
        
    print(f"\n[1/4] Loading {input_file}...")
    df = pd.read_csv(input_file)
    messages = df["message"].astype(str).tolist()
    print(f"      Loaded {len(messages)} raw commits.")

    # 2. Setup Device
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"\n[2/4] Loading Sentiment AI on {device_name}...")
    
    # We use batch size 64 for GPU, 16 for CPU to prevent RAM blowouts
    batch_size = 64 if device == 0 else 16
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    pipe = pipeline(
        "text-classification", 
        model="GitRatBCSAD/gomi-sentiment", 
        device=device, 
        batch_size=batch_size,
        token=token
    )

    # 3. Labeling (with progress bar)
    print(f"\n[3/4] Auto-labeling (this may take a while on CPU)...")
    
    # Hugging Face pipelines with generator output for tqdm
    results = []
    for out in tqdm(pipe(messages, truncation=True, max_length=128), total=len(messages)):
        results.append(out)

    # 4. Save results
    print("\n[4/4] Attaching labels and saving...")
    df["reconciled_emotion"] = [res["label"] for res in results]
    df["confidence"] = [res["score"] for res in results]
    
    df.to_csv(output_file, index=False)
    print(f"      Done! Saved to {output_file}")
    print(f"      You can now add this to train_sentiment.py!")

if __name__ == "__main__":
    main()
