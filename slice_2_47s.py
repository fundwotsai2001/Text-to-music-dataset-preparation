import os
import torchaudio
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # For progress bar
from torchaudio import transforms as T
torchaudio.set_audio_backend("sox_io")
import argparse
def process_file(file_path, input_root, output_root, segment_length=2097152):
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 44100:
            resampler = T.Resample(sample_rate, 44100)
            waveform = resampler(waveform)
        if waveform.shape[0] == 2 and waveform.shape[1] >= segment_length:
            num_segments = waveform.shape[1] // segment_length
            relative_path = os.path.relpath(os.path.dirname(file_path), input_root)
            out_dir = os.path.join(output_root, relative_path)
            os.makedirs(out_dir, exist_ok=True)
            base_name, _ = os.path.splitext(os.path.basename(file_path))
            for i in range(num_segments):
                start, end = i * segment_length, (i + 1) * segment_length
                segment_waveform = waveform[:, start:end]
                
                # Save to a temporary WAV file
                temp_wav = os.path.join(out_dir, f"{base_name}_chunk{i}.mp3")
                torchaudio.save(temp_wav, segment_waveform, 44100)
        else:
            print(f"Skipping '{file_path}' (not stereo or too short).")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def slice_audio_files_parallel(input_root, output_root, segment_length=2097152, num_workers=8):
    # Collect all mp3 files
    mp3_files = []
    for root, _, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith('.mp3'):
                mp3_files.append(os.path.join(root, file))
    
    # Process files in parallel with a progress bar
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_file, file_path, input_root, output_root, segment_length)
            for file_path in mp3_files
        ]
        # Wrap the as_completed iterator with tqdm to show progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            # Optionally handle results or exceptions
            future.result()

if __name__ == "__main__":
    # Paths
    parser = argparse.ArgumentParser(description="slicing audio")
    parser.add_argument("--input_folder", type=str, help="The audio folder path")
    parser.add_argument("--output_folder", type=str, default="./FMA_47s", help="Output path for the sliced audio")
    args = parser.parse_args()  # Parse the arguments
    input_folder = args.input_folder # /mnt/gestalt/home/lonian/datasets/mtg_full/ "/mnt/gestalt/database/FMA/fma_track/audio"
    output_folder = args.output_folder
    # Slice all files, discarding any under 2,097,152 samples
    slice_audio_files_parallel(input_folder, output_folder, segment_length=2097152, num_workers=20)
