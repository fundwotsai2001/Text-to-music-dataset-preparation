import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import json
from tqdm import tqdm
import os
import argparse

# Initialize the processor and model
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter audio that contain vocal")
    parser.add_argument("--input_json", type=str, help="Input json output by filter_volcal.py")
    parser.add_argument("--output_json", type=str, default="./filtered_vocal_all_caption.json", help="output json file")
    args = parser.parse_args()  # Parse the arguments
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda", torch_dtype=torch.bfloat16
    )
    with open(args.input_json) as f:
        meta = json.load(f)
    results = []
    for audio_path1, audio_path2, audio_path3, audio_path4 in tqdm(zip(meta[::4], meta[1::4], meta[2::4], meta[3::4]), 
                            total=len(meta)//4):
        ### first meta #######
        audio_full_path1 = audio_path1
        # Define a simple conversation for music captioning with a local audio path
        conversation1 = [
            {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
            {'role': 'user', 'content': [
                {'type': 'audio', "audio_url": f"{audio_full_path1}"},  # Replace with your local file path
                {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
            ]}
        ]
        ### second meta #######
        audio_full_path2 = audio_path2
        # Define a simple conversation for music captioning with a local audio path
        conversation2 = [
            {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
            {'role': 'user', 'content': [
                {'type': 'audio', "audio_url": f"{audio_full_path2}"},  # Replace with your local file path
                {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
            ]}
        ]


        ### first meta #######
        audio_full_path3 = audio_path3
        # Define a simple conversation for music captioning with a local audio path
        conversation3 = [
            {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
            {'role': 'user', 'content': [
                {'type': 'audio', "audio_url": f"{audio_full_path3}"},  # Replace with your local file path
                {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
            ]}
        ]
        ### second meta #######
        audio_full_path4 = audio_path4
        # Define a simple conversation for music captioning with a local audio path
        conversation4 = [
            {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
            {'role': 'user', 'content': [
                {'type': 'audio', "audio_url": f"{audio_full_path4}"},  # Replace with your local file path
                {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
            ]}
        ]

        
        # Generate the text prompt from the conversation
        conversations = [conversation1, conversation2, conversation3, conversation4]
        text = [processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) for conversation in conversations]


        # Load audio data from the provided local file path
        audios = []
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if ele["type"] == "audio":
                            audios.append(
                                librosa.load(
                                    ele['audio_url'], 
                                    sr=processor.feature_extractor.sampling_rate)[0]
                            )
                        

        # Prepare inputs for the model (pass audios directly without extra list wrapping)
        inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate).to('cuda')
        inputs.input_ids = inputs.input_ids.to("cuda")
        # Generate the music caption
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        # Remove the prompt part from the generated output
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        # Decode and print the generated caption
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(response)
        results.append({
            "path": audio_path1,
            "Qwen_caption": response[0],
        })
        results.append({
            "path": audio_path2,
            "Qwen_caption": response[1],
        })
        results.append({
            "path": audio_path3,
            "Qwen_caption": response[2],
        })
        results.append({
            "path": audio_path4,
            "Qwen_caption": response[3],
        })
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=4)