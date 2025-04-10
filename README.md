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
docker run --gpus all --ipc=host -it fetchML examples.txt -f -sc -sa
```
`-it` is optional, but it allows you to see the `stdout` before the program terminates. 

## Explanations
### Task 1: Sentence Transformer Implementation
I figured Fetch might appreciate something built on the concept of scanning receipts, so I developed a fixed-length sentence embedder with a `ReceiptAttention` attention module. It's initialized as so:
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
