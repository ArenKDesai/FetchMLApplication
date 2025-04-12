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

### Task 3: Training Considerations
#### Freezing
1. "If the entire network should be frozen"

The entire network should be frozen primarily if the developer is using the model to extract certain features. For example, the model developed for Tasks 1 and 2 could be frozen completely if I just need to use it as a quick and effective embedding generator. 

The network could also be frozen if resources are restricted, such as having a limited dataset or being low in computational power. However, in this case, it may also be better to downgrade to a smaller or more deterministic model without the data or resources to take advantage of a large network. 

Freezing the entire network would disable the improvement of the model, so this should only be done if the model's output is satisfactory and there is no reason to improve it. 

2. "If only the transformer backbone should be frozen"

The transformer backbone should be frozen if the developer is fine-tuning a model for a specific task. Most often, the first 75% of layers can be frozen and only last 25% of layers retrained, and the model will perform substantially better without a significant portion of the training time. 

This means the earlier layers of the model, which typically extract universal features, wouldn't be fine-tuned on the developer's use case. This is usually fine for specific tasks that rely on the later layers, but if the developer is training the model for research or creating a model at a company like Google or Meta, they may want to retrain the entire model. 

3. "If only one of the task-specific heads (either for Task A or Task B) should be frozen"

Task-specific heads can be frozen if the developer wants to limit the speed of the model's inference, and doesn't always need some of the outputs. For example, if I wanted to classify a receipt to belonging from a certain merchant category but I wanted a quick, easy output that didn't care about named entities (or if I wanted to test the output of my sentence classifier without relying on the NER), I could freeze the NER head. 

This could throw off the trainig of the model if it learns to stop using the output of specific heads of the model, so this should only be done for inference. However, this could be useful for testing. 

#### Transfer Learning

1. "The choice of a pre-trained model"

Pre-trained models are effective most of the time. Models like BERT can be used and modified with one extra layer, like BERTgrid models, for significant improvements on the model output for specific use cases. I would research pretrained models for my specific use case and test my GPU architecture to see what models I can handle. Then, I would analyze the architecture of my pretrained model and decide what layers I need to add or modify.

2. "The layers you would freeze/unfreeze"

If I have strong GPUs or a good cluster, I would probably test the output of the pretrained model in total as a baseline. However, for actual development, I would freeze the first 75% of layers for fine-tuning, and if the output of the model is unsatisfactory, I might try unfreezeing some of the later layers. 

3. "The rationale behind these choices"

For my case, I used BERT, which was typically a conversional, general model. My model isn't a generative model, though, so I would probably retrain more than the last 25% of layers. 

Freezing the earlier layers of a model would typically also help avoid catastrophic forgetting. This may not be a problem while using BERT, but if I were using a pretrained version of ViBERTgrid, I may want to keep some of those layers frozen so it doesn't forget how to utilize layout information. 

### Task 4: Training Loop Implementation
