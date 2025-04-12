# Fetch ML Apprentice Take Home Exercise
## Solution
The dockerfile can be run like so:

`docker run -it imagename "your sentence here"`

with these optional arguments:
* -f --file
    * the sentence you passed was a filename. The contents of the file is considered a sentence. 
* -s --sentence-classification 
    * performs sentence classification
* -n --named-entity-recognition
    * performs named entity recognition

For example:

```bash
docker build -t fetchml .
docker run --gpus all --ipc=host --runtime=nvidia -it fetchml receipt1.txt -f -s -n
```

NOTE: This assumes you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed. If not, remove `--runtime=nvidia`. 

## Explanations
### Task 1: Sentence Transformer Implementation
Ex:
I figured Fetch might appreciate something built on the concept of scanning receipts, so I developed a fixed-length sentence embedder with a `ReceiptAttentionModel` model. It's developed with two metrics in mind. The first is a keyword similarity metric for developers to input words they want `attention` focused on, and the second is a neural network so the model can learn its own complex patterns. 

`ReceiptAttentionModel` is initialized as so:
```Python
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

#### Part A: Sentence Classification
I decided to concatenate three layers --- one for keyword detection, one for instance count, and one for layout --- that follows BERT's backbone with LayerNorm and GELU. I chose to target receipts as my case study, and the classification of those receipts is limited to "grocery", "restaurant", or "retail". I think building a more complex system may be beneficial for an unsupervisted task or if I were trying to categorize these receipts into a much broader vocabulary, but for this case, a simple model would be the fastest, easiest to maintain, and still effective. 

#### Part B: Named Entity Recognition
Admittedly, I've never done NER before. I referred to 2023 paper [Comprehensive Overview of Named Entity Recognition: Models, Domain-Specific Applications and Challenges](https://arxiv.org/abs/2309.14084) by Kalyani Pakhale. The paper referenced a BERTgrid model called ViBERTgrid introduced in a previous 2021 paper, [ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents](https://arxiv.org/abs/2105.11672) by Lin et al. The ViBERTgrid model captures textual and layout information well with a joint training strategy with a CNN, and I figured it would be a great fit for this project. 
Unfortunately, I couldn't get this working quite in time, and I didn't really have the training data anyways. The ViBERTgrid model is what I would implement with more time...but for this project, I decided to take a supervised learning approach and implemented a linear chain CRF. I don't have the training data to train it, but this approach was relitively simple and, in my experience, suprisingly effective. 

#### Task 2 Summary
I decided to keep sentence classification simple with three extra linear layers that follow BERT's backbone, and for named entity recognition I implemented a linear chain CRF that was inspired by the ViBERTgrid model. 