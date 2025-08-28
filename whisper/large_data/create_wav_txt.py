# Configuration
output_file = 'wav.txt'
audio_folder = '/home/zahra/hypo-light/generate_data/whisper/large_data/noisy_test_data'
start_index = 0
end_index = 28859  # audio_028859.wav corresponds to 28859

# Generate the list
with open(output_file, 'w') as f:
    for i in range(start_index, end_index + 1):
        utt_id = f'utt_id_{i + 1}'
        audio_file = f'{audio_folder}/noisy_test_{i}.wav'
        line = f'{utt_id} {audio_file}\n'
        f.write(line)

print(f'List generated and saved to {output_file}')


import csv

# Configuration
input_csv = '/home/zahra/hypo-light/generate_data/test_data/transcriptions.csv'   # Path to your CSV file
output_txt = 'txt.txt'

# Processing
with open(input_csv, 'r', encoding='utf-8') as csvfile, open(output_txt, 'w', encoding='utf-8') as outfile:
    reader = csv.DictReader(csvfile)
    for idx, row in enumerate(reader, start=1):
        utt_id = f'utt_id_{idx}'
        transcription = row['transcription'].strip()
        outfile.write(f'{utt_id} {transcription}\n')

print(f'Transcriptions saved to {output_txt}')
