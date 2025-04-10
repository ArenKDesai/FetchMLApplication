# Fetch ML Apprentice Take Home Exercise
## Solution
The dockerfile can be run like so:

`docker run -it imagename "your sentence here"`

with these optional arguments:
* -f --file
    * the sentence you passed was a filename. Iterates through the files separated by newline (\n)
* -s --sentence-classification 
    * performs sentence classification
* -n --named-entity-recognition
    * performs named entity recognition

For example:

```bash
docker build -t fetchML .
docker run --gpus all --ipc=host -it fetchML receipt1.txt -f -sc -sa
```
`-it` is optional, but it allows you to see the `stdout` before the program terminates. 

## Explanations
### Task 1: Sentence Transformer Implementation
I figured Fetch might appreciate something built on the concept of scanning receipts, so I developed a fixed-length sentence embedder with a `ReceiptAttention` attention module. It's developed with two metrics in mind. The first is a keyword similarity metric for developers to input words they want `attention` focused on, and the second is a neural network so the model can learn its own complex patterns. 

`ReceiptAttention` is initialized as so:
```Python
def __init__(self, hidden_size):
    super(ReceiptAttention, self).__init__()
    # learnable query vector
    self.query = torch.nn.Parameter(torch.randn(hidden_size))
    # refine attention weights
    self.attention_net = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, 64),
        torch.nn.GELU(), # delinearlize
        torch.nn.Linear(64, 1)
    )
```
I wanted us developers to have some control over what phrases we focus `attention` on, so `self.query` is the key that allows the model to target keywords. However, I also wanted the model to learn more complex phrases on its own, so `attention_net` allows us to target more complex patterns. 

Next, scores are calculated during the forward pass:
```Python
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
```
The dot product between the embeddings and the query vector allows us to find basic similarity between the query vector and the embdedded sentences, and the pass through `attention_net` finds the complex patterns. The combination of the two is the score. 

The rest is filtering out padding tokens and converting scores into weights indicating importance in the sentences. 

**NOTE**: The model currently doesn't actually train the attention module on keywords (see `initialize_receipt_attention` on line 73 of `task1.py`). This would've been a nice touch, but the document specifies that training isn't a requirement, and I already spent too much time trying to figure out why the NVIDIA contianer toolkit wasn't working on my laptop!

### Task 2: Multi-Task Learning Expansion