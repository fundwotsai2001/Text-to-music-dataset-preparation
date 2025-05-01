import csv
import json
import os

# 1. Load CSV paths (without the ".mp3")
csv_paths = set()
with open('./filtered_vocal_all_caption1.json', newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # row['path'] is e.g. "34/1004034.mp3"
        base = os.path.splitext(row['path'])[0]  # "34/1004034"
        csv_paths.add(base)

# 2. Load your JSON list
with open('./Qwen_caption.json', 'r', encoding='utf-8') as f:
    json_list = json.load(f)

# 3. Find JSON entries whose base path (before "_chunk") isn't in the CSV
not_matched = []
for entry in json_list:
    json_path = entry['path']               # e.g. "88/1394788_chunk0.mp3"
    no_ext = os.path.splitext(json_path)[0] # "88/1394788_chunk0"
    base = no_ext.split('_chunk')[0]        # "88/1394788"
    if base not in csv_paths:
        not_matched.append(entry)

# 4. Write out the mismatches
with open('not_matched.json', 'w', encoding='utf-8') as f:
    json.dump(not_matched, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(not_matched)} non-matching entries to not_matched.json")
