from transformers import AutoTokenizer, AutoModel
import torch

def encode_sentence(sentence, isfile):
    """Encode sentences"""
    model_id = "unsloth/gemma-2-2b-it" # for BPE
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    text = [sentence] # overridden if isfile
    if isfile:
        with open(sentence, "r") as f:
            text = f.read().split("\n")
    # Begin encoding sentences
    embeddings = []
    for sentence in text:
        print(f"Sentence: {sentence}")
        tokens = tokenizer(sentence, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**tokens)
        embeddings = outputs.last_hidden_state
        print(f"Embeddings: {embeddings}")
        attention_mask = tokens['attention_mask']