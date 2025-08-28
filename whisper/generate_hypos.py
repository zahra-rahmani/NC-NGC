import whisper
import os, random, copy
import numpy as np
import torch
import pandas as pd
from tqdm.notebook import tqdm
import collections, json
import editdistance
from hazm import Normalizer
from num2words import num2words
import re
from itertools import islice

hazm_normalizer = Normalizer()

def calculate_wer(pre, ref):
    return editdistance.eval(pre.split(), ref.split()) / max(len(ref.split()), 1)

EN_TO_FA_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

def convert_numbers_to_words(text, convert_large_numbers=False):
    """
    Convert numbers in the text to Persian words selectively.
    - Numbers up to 9999 will be converted to words
    - Long numbers (e.g. phone numbers, IDs) will be kept as digits
    - Persian and English digits are supported
    """
    def replace_num(match):
        num_str = match.group()
        try:
            num_int = int(num_str)

            if (not convert_large_numbers and num_int > 9999) or len(num_str) > 4:
                return num_str  

            return num2words(num_int, lang='fa')
        except:
            return num_str  # fallback if conversion fails

    # English digits
    text = re.sub(r'\d+', replace_num, text)

    # Persian digits
    text = re.sub(r'[۰-۹]+', 
                  lambda m: replace_num(
                      ''.join(str("۰۱۲۳۴۵۶۷۸۹".index(d)) for d in m.group())), 
                  text)

    return text

def remove_punct(text):
    """Remove punctuation for WER calculation"""
    # Keep only Persian letters, digits, and spaces
    return re.sub(r'[^\w\sءآ-ی۰-۹]', '', text)

def normalize_text_hazm(text):
    """
    Persian text normalization using Hazm:
    - Unicode unification
    - Correct spacing & half-spacing
    - Converts English numbers to Persian
    - Removes extra spaces
    - Removes punctuation
    """
    try:
        # Convert small/valid numbers to Persian words
        text = convert_numbers_to_words(text)

        # Hazm normalization
        text = hazm_normalizer.normalize(text)

        # Convert English digits to Persian digits
        text = text.translate(EN_TO_FA_DIGITS)

        # Remove punctuation
        text = remove_punct(text)

        # Remove extra spaces
        text = ' '.join(text.split())

    except Exception as e:
        print(f"Normalization failed: {text}, Error: {e}")
        text = '<UNK>'

    return text if text.strip() else '<UNK>'


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model('large')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HP dataset generation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--asr_wav', type=str, help='wav list file')
    parser.add_argument('--asr_txt', type=str, help='transcription file')
    parser.add_argument('--hp_json', type=str, help='generated hp data file')
    parser.add_argument('--limit', type=int, default=None, help='maximum number of samples to process')
    args = parser.parse_args()

    f_wav = open(args.asr_wav, 'r')
    f_txt = open(args.asr_txt, 'r')

    json_file = []
    id = 0
    wer = 0

    wav_iter = islice(f_wav, args.limit) if args.limit else f_wav

    for line in wav_iter:
        audio_path = line.strip().split()[-1]
        gt = ' '.join(f_txt.readline().strip().split()[1:])
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language='fa', beam_size=10)
        results = whisper.decode(model, mel, options)

        input = []
        for result in results:
            if len(input) < 5 and len(result) > 0 and result not in input:
                input.append(result)
        if len(input) < 5:
            for _ in range(5 - len(input)):
                repeat = copy.deepcopy(random.choice(input))
                input.append(repeat)

        # Normalize hypotheses
        for i in range(len(input)):
            input[i] = normalize_text_hazm(input[i])

        # Normalize ground-truth
        output = normalize_text_hazm(gt)

        data = {"input": input, "output": output}
        json_file.append(data)

        # Calculate WER (punctuation removed)
        cur_wer = calculate_wer(input[0], output)
        id += 1
        wer += cur_wer
        print(f'Utterance {id}: WER = {cur_wer}')

    f_wav.close()
    f_txt.close()

    wer /= id
    print(f'Final WER = {wer}')

    with open(args.hp_json, 'w', encoding='utf-8') as f:
        json.dump(json_file, f, ensure_ascii=False, indent=2)
