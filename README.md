# Text to music dataset prepare
The implementation for the audio captioning pipeline used in MuseControlLite.

Follow the instructions below:
1. Clone the github repo, and install the required package:
```
git clone https://github.com/fundwotsai2001/Text-to-music-dataset-preparation.git
cd Text-to-music-dataset-preparation
pip install -r requirements
```

2. Download the full-length version of Jamendo via:
```
python3 ./mtg-jamendo-dataset/scripts/download/download.py --dataset raw_30s --type audio ./mtg-full --unpack --remove
```
`../mtg-full` is the output directory.

3. Resample the audio to 44100 hz, and slice it to shape (2, 2097152), this is optional if you are not using MuseControlLite.
```
python slice_2_47s.py --input_folder ./mtg-full --output_folder ./mtg-full-47s
```
4. Use sound event detection to filter out audio that contains vocal.
```
python ./panns_inference/filter_vocal.py --audio_folder ./mtg_full_47s --json_path ./filtered_vocal_all.json
```
`--json_path` is the output json file that contains audio paths that do not contain vocal.

5. Use `Qwen/Qwen2-Audio-7B-Instruct` to obtain all captions for the audio.
```
python Qwen2audio_captioning.py --input_json ./filtered_vocal_all.json --output_json ./filtered_vocal_all_caption1.json
```
6. Optional, if you are evaluating with the [Song Describer Dataset](https://github.com/mulab-mir/song-describer-dataset), or benchmarking with [MuseControlLite](https://github.com/fundwotsai2001/MuseControlLite), you should filter the [Song Describer Dataset] from the the Jamendo dataset.

