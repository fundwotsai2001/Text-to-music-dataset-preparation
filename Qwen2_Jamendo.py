import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
from io import BytesIO
from urllib.request import urlopen
import json
from tqdm import tqdm
import os
# Initialize the processor and model
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda:2", torch_dtype=torch.bfloat16
)
with open("Jamendo_filtered_vocal_all.json") as f:
    meta = json.load(f)
results = []
for meta_item1, meta_item2, meta_item3, meta_item4 in tqdm(zip(meta[::4], meta[1::4], meta[2::4], meta[3::4]), 
                         total=len(meta)//4):
    ### first meta #######
    audio_path1 = meta_item1.get('path')
    audio_full_path1 = os.path.join("mtg_full_47s", audio_path1)
    # Define a simple conversation for music captioning with a local audio path
    conversation1 = [
        {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
        {'role': 'user', 'content': [
            {'type': 'audio', "audio_url": f"{audio_full_path1}"},  # Replace with your local file path
            {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
        ]}
    ]
    ### second meta #######
    audio_path2 = meta_item2.get('path')
    audio_full_path2 = os.path.join("mtg_full_47s", audio_path2)
    # Define a simple conversation for music captioning with a local audio path
    conversation2 = [
        {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
        {'role': 'user', 'content': [
            {'type': 'audio', "audio_url": f"{audio_full_path2}"},  # Replace with your local file path
            {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
        ]}
    ]


     ### first meta #######
    audio_path3 = meta_item3.get('path')
    audio_full_path3 = os.path.join("mtg_full_47s", audio_path3)
    # Define a simple conversation for music captioning with a local audio path
    conversation3 = [
        {'role': 'system', 'content': 'You are a helpful assistant for music captioning.'},
        {'role': 'user', 'content': [
            {'type': 'audio', "audio_url": f"{audio_full_path3}"},  # Replace with your local file path
            {'type': 'text', 'text': 'Generate a detailed caption for this music piece, primarily focusing on instrumentation, genre, mood, rhythm, and scenario.'}
        ]}
    ]
    ### second meta #######
    audio_path4 = meta_item4.get('path')
    audio_full_path4 = os.path.join("mtg_full_47s", audio_path4)
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
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True, sampling_rate=processor.feature_extractor.sampling_rate).to('cuda:2')
    inputs.input_ids = inputs.input_ids.to("cuda:2")
    # Generate the music caption
    generate_ids = model.generate(**inputs, max_length=256)
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
with open("Qwen_caption_Jamendo.json", "w") as f:
    json.dump(results, f, indent=4)