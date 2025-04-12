from transformers import AutoTokenizer, AutoModel
import torch

class ReceiptAttentionModel(torch.nn.Module):
    def __init__(self, hidden_size):
        super(ReceiptAttentionModel, self).__init__()
        # learnable query vector
        self.query = torch.nn.Parameter(torch.randn(hidden_size))
        # refine attention weights
        self.attention_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 64),
            torch.nn.GELU(), # delinearlize
            torch.nn.Linear(64, 1)
        )
        
    def forward(self, token_embeddings, attention_mask):
        # dot product with query vector
        dot_scores = torch.matmul(token_embeddings, self.query)
        # delinearize with nn
        net_scores = self.attention_net(token_embeddings).squeeze(-1)
        attention_scores = dot_scores + net_scores
        
        # filter and normalize
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -10000.0)
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
        weighted_embeddings = torch.sum(
            token_embeddings * attention_weights.unsqueeze(-1), 
            dim=1
        ) # NOTE Weighted fixed-length embeddings
        return weighted_embeddings, attention_weights

def embed_sentences(sentences, isfile=False):
    # params
    text = [sentences]
    if isfile:
        with open(sentences, "r") as f:
            text = [f.read()]
    print(f"Sentences: {text}")
    print()

    # Setup models
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    # attention weights
    attention_module = ReceiptAttentionModel(model.config.hidden_size)
    model.eval()
    attention_module.eval()
    
    # encodings
    encoded_input = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    # embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Apply attention weights to embeddings
        sentence_embeddings, attention_weights = attention_module(
            model_output.last_hidden_state, 
            encoded_input['attention_mask']
        )
        # normailze
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    print()
    print(f"Embeddings: {sentence_embeddings}")
    return sentence_embeddings, attention_weights

def initialize_receipt_attention(attention_module):
    receipt_keywords = [ # We're pretending these were found through analysis
        "PAID", "paid", "Paid",
        "TOTAL", "total", "Total",
        "RECEIPT", "receipt", "Receipt",
        # TODO add more
    ]
    # TODO train on keywords
    return attention_module