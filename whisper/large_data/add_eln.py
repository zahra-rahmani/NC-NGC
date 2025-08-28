import json
import torch
import itertools
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------- ELN function ----------
def compute_ELN(hypotheses, sbert_model):
    """
    hypotheses: list of N ASR hypotheses
    sbert_model: SentenceTransformer model
    returns: ELN (tensor)
    """
    N = len(hypotheses)

    # --- Utterance-level ---
    embeddings = sbert_model.encode(hypotheses, convert_to_tensor=True)  # [N, D]
    utt_diffs = []
    for i, j in itertools.combinations(range(N), 2):
        diff = embeddings[i] - embeddings[j]
        utt_diffs.append(diff)
    ELN_utt = torch.cat(utt_diffs, dim=0)  # [(N*(N-1)/2) * D]

    # --- Token-level ---
    token_lists = [hyp.split() for hyp in hypotheses]
    max_len = max(len(toks) for toks in token_lists)
    for i, toks in enumerate(token_lists):
        token_lists[i] = toks + ["Ø"] * (max_len - len(toks))  # pad

    all_tokens = [tok for toks in token_lists for tok in toks]
    token_embeddings = sbert_model.encode(all_tokens, convert_to_tensor=True)
    token_embeddings = token_embeddings.view(N, max_len, -1)  # [N, T, D]

    tok_diffs = []
    for i, j in itertools.combinations(range(N), 2):
        diff_sum = (token_embeddings[i] - token_embeddings[j]).sum(dim=0)
        tok_diffs.append(diff_sum)
    ELN_tok = torch.cat(tok_diffs, dim=0)

    # --- Combine ---
    ELN = torch.cat([ELN_utt, ELN_tok], dim=0)
    return ELN


# ---------- Dataset Processing ----------
def build_dataset(raw_file, out_file, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    # Load raw dataset
    with open(raw_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Load SBERT model
    sbert_model = SentenceTransformer(model_name)

    formatted_data = []

    # tqdm progress bar
    for example in tqdm(raw_data, desc="Processing dataset", unit="example"):
        hypotheses = example["input"]
        label = example["output"]

        # Compute ELN
        ELN = compute_ELN(hypotheses, sbert_model)

        formatted_data.append({
            "hypotheses": hypotheses,
            "label": label,
            "eln": ELN  # keep as tensor
        })

    # Save entire dataset as .pt
    torch.save(formatted_data, out_file)

    print(f"✅ Final dataset saved to {out_file}, total {len(formatted_data)} examples.")



# ---------- Run ----------
if __name__ == "__main__":
    build_dataset(
        raw_file="data_large_test.json",   # your input dataset file
        out_file="data_large_test.pt" # output with ELN + formatting
    )


