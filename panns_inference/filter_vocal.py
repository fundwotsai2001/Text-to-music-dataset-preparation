import os
import glob
import json
import torch
import numpy as np
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as T
from panns_inference import SoundEventDetection, labels
from tqdm import tqdm 
import argparse

# Define a custom Dataset that loads audio files from a folder
class AudioFolderDataset(Dataset):
    def __init__(self, folder_path, target_sr=32000, transform=None):
        self.target_sr = target_sr
        self.transform = transform
        self.folder_path = folder_path
        mp3_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.mp3'):
                    mp3_files.append(os.path.join(root, file))
        self.files = mp3_files
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        # Load audio using torchaudio; waveform shape: (channels, time)
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono by averaging channels if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample to target_sr if needed
        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)
        
        # Remove channel dimension: resulting shape (time,)
        waveform = waveform.squeeze(0)
        
        sample = {"waveform": waveform, "file_path": file_path}
        if self.transform:
            sample = self.transform(sample)
        return sample

# Define a collate function to pad waveforms in a batch to the same length
def collate_fn(batch):
    waveforms = [item["waveform"] for item in batch]
    file_paths = [item["file_path"] for item in batch]
    # Find the maximum length in this batch
    max_length = max(waveform.shape[0] for waveform in waveforms)
    padded_waveforms = []
    for waveform in waveforms:
        if waveform.shape[0] < max_length:
            pad_amount = max_length - waveform.shape[0]
            padded_waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            padded_waveform = waveform[:max_length]
        padded_waveforms.append(padded_waveform)
    
    batch_waveforms = torch.stack(padded_waveforms, dim=0)  # shape: (batch_size, max_length)
    return {"waveform": batch_waveforms, "file_path": file_paths}

# Function to check if the framewise output contains vocal-related labels
def detect_vocal_in_sed(framewise_output, vocal_keywords, threshold=0.05):
    """
    Args:
      framewise_output: numpy array of shape (time_steps, classes_num)
      vocal_keywords: list of strings (keywords for vocal-related labels)
      threshold: float, confidence threshold
      
    Returns:
      True if any vocal-related class exceeds the threshold, False otherwise.
    """
    # For each class, take the maximum confidence across time
    classwise_max = np.max(framewise_output, axis=0)
    for idx, score in enumerate(classwise_max):
        tag = labels[idx]
        # Check if the label contains any vocal-related keyword
        if any(keyword in tag.lower() for keyword in vocal_keywords):
            if score > threshold:
                return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filter audio that contain vocal")
    parser.add_argument("--audio_folder", type=str, help="The audio folder path")
    parser.add_argument("--json_path", type=str, default="./filtered_vocal_all.json", help="path for json that record the non-vocal audio")
    args = parser.parse_args()  # Parse the arguments
    device = 'cuda'  # Use 'cuda' if available, otherwise 'cpu'
    audio_folder = args.audio_folder # '/home/fundwotsai/Music-Controlnet-light/mtg_full_47s'
    json_path = args.json_path # "Jamendo_filtered_vocal_all.json"
    # Create the dataset and DataLoader
    dataset = AudioFolderDataset(audio_folder, target_sr=32000)
    dataloader = DataLoader(dataset, batch_size=24, shuffle=False, collate_fn=collate_fn, num_workers=12)
    
    # Initialize the SoundEventDetection model
    sed = SoundEventDetection(
        checkpoint_path=None, 
        device=device, 
        interpolate_mode='nearest'
    )
    
    # Define vocal-related keywords
    vocal_keywords = [
        'sing', 'speech', 'vocal', 'talk', 'voice', 
        'conversation', 'monologue', 'babbling', 
        'shout', 'bellow', 'whoop', 'yell', 
        'whispering', 'choir', 'rapping', 
        'yodeling', 'chant', 'mantra', 'humming'
    ]
    music_keywords = ["music"]
    non_vocal_files = []  # List to store file paths with detected vocals
    vocal_files = []
    # Process batches from the DataLoader
    for batch in tqdm(dataloader):
        waveforms = batch["waveform"].to(device)  # shape: (batch_size, max_length)
        file_paths = batch["file_path"]
        
        # Run inference for the entire batch
        batch_framewise_outputs = sed.inference(waveforms)  # shape: (batch_size, time_steps, classes_num)
        with torch.no_grad():
            # Check each sample in the batch for vocal-related events
            for i, file_path in enumerate(file_paths):
                framewise_output = batch_framewise_outputs[i]
                if not detect_vocal_in_sed(framewise_output, vocal_keywords, threshold=0.02):
                    non_vocal_files.append(file_path)
        torch.cuda.empty_cache()
    # Save the list of detected vocal files to a JSON file
    with open(json_path, "w") as f:
        json.dump(non_vocal_files, f, indent=4)
    
    print(f"Vocal detection complete. Results saved to {json_path}")
    print(len(non_vocal_files))