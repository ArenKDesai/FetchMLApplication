from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF
import torch

def initialize_receipt_attention(attention_module):
    receipt_keywords = [ # We're pretending these were found through analysis
        "PAID", "paid", "Paid",
        "TOTAL", "total", "Total",
        "RECEIPT", "receipt", "Receipt",
        # TODO add more
    ]
    # TODO train on keywords
    return attention_module

class EnhancedClassificationHead(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedClassificationHead, self).__init__()
        self.receipt_feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.2)
        )
        
        self.keyword_detector = torch.nn.Linear(256, 64)
        self.amount_detector = torch.nn.Linear(256, 64)
        self.layout_detector = torch.nn.Linear(256, 64)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(256 + 64*3, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, embeddings, attention_weights):
        base_features = self.receipt_feature_extractor(embeddings)
        
        keyword_features = self.keyword_detector(base_features)
        amount_features = self.amount_detector(base_features)
        layout_features = self.layout_detector(base_features)
        
        combined_features = torch.cat([
            base_features, 
            keyword_features, 
            amount_features, 
            layout_features
        ], dim=1)
        
        logits = self.classifier(combined_features)
        return logits

class EnhancedNERHead(torch.nn.Module):
    def __init__(self, input_size, num_tags):
        super(EnhancedNERHead, self).__init__()
        
        self.context_encoder = torch.nn.GRU(
            input_size, 
            128, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True,
            dropout=0.2
        )
        
        self.price_detector = torch.nn.Linear(256, 64) 
        self.date_detector = torch.nn.Linear(256, 64)
        self.product_detector = torch.nn.Linear(256, 64)
        self.store_detector = torch.nn.Linear(256, 64)
        
        self.emissions = torch.nn.Linear(256 + 64*4, num_tags)
        
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, token_embeddings, attention_mask):
        context_out, _ = self.context_encoder(token_embeddings)
        
        price_features = self.price_detector(context_out)
        date_features = self.date_detector(context_out)
        product_features = self.product_detector(context_out)
        store_features = self.store_detector(context_out)
        
        combined_features = torch.cat([
            context_out, 
            price_features, 
            date_features, 
            product_features, 
            store_features
        ], dim=2)
        
        emissions = self.emissions(combined_features)
        return emissions
    
    def decode(self, emissions, mask):
        return self.crf.decode(emissions, mask)

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

class EnhancedMultiTaskReceiptModel(torch.nn.Module):
    def __init__(self, num_classes, num_ner_tags):
        super(EnhancedMultiTaskReceiptModel, self).__init__()
        
        self.model_id = "google-bert/bert-base-uncased" # switched to bert from task 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.transformer = AutoModel.from_pretrained(self.model_id)
        
        self.attention_module = ReceiptAttentionModel(self.transformer.config.hidden_size)
        self.attention_module = initialize_receipt_attention(self.attention_module)
        
        self.classification_head = EnhancedClassificationHead(
            self.transformer.config.hidden_size, 
            num_classes
        )
        
        self.ner_head = EnhancedNERHead(
            self.transformer.config.hidden_size, 
            num_ner_tags
        )
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None, ner_labels=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        token_embeddings = outputs.last_hidden_state
        sentence_embeddings, attention_weights = self.attention_module(
            token_embeddings, 
            attention_mask
        )
        
        classification_logits = self.classification_head(sentence_embeddings, attention_weights)
        
        ner_emissions = self.ner_head(token_embeddings, attention_mask)
        
        # for training
        loss = None
        if labels is not None and ner_labels is not None:
            classification_loss = torch.nn.functional.cross_entropy(
                classification_logits, 
                labels
            )
            
            ner_loss = -self.ner_head.crf(
                ner_emissions, 
                ner_labels, 
                mask=attention_mask.bool(), 
                reduction='mean'
            )
            
            loss = classification_loss + ner_loss
        
        ner_tags = self.ner_head.decode(ner_emissions, attention_mask.bool())
        
        return {
            'classification_logits': classification_logits,
            'ner_emissions': ner_emissions,
            'ner_tags': ner_tags,
            'attention_weights': attention_weights,
            'sentence_embeddings': sentence_embeddings,
            'loss': loss
        }

def classify_recognize(sentences, isfile=False, do_classification=True, do_ner=True):
    # params
    text = [sentences]
    if isfile:
        with open(sentences, "r") as f:
            text = [f.read()]
    
    receipt_classes = ["grocery", "restaurant", "retail"]
    ner_tags = ["O", # outside
                "B-PRODUCT", "I-PRODUCT", 
                "B-PRICE", "I-PRICE", 
                "B-DATE", "I-DATE", 
                "B-STORE", "I-STORE", 
                "B-PAYMENT", "I-PAYMENT"]
    
    model = EnhancedMultiTaskReceiptModel(
        num_classes=len(receipt_classes),
        num_ner_tags=len(ner_tags)
    )
    
    encoded_input = model.tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(**encoded_input)
    
    results = {}
    if do_classification:
        classification_probs = torch.nn.functional.softmax(
            outputs['classification_logits'], dim=1
        )
        predicted_classes = torch.argmax(classification_probs, dim=1)
        results['classification'] = {
            'predicted_classes': [receipt_classes[i] for i in predicted_classes],
            'probabilities': classification_probs.tolist()
        }
        print(f"Classifications: {results['classification']['predicted_classes']}")
        print(f"Classification Probabilities: {classification_probs.tolist()}")
    
    if do_ner:
        predicted_tags = outputs['ner_tags']
        
        results['ner'] = []
        for i, sentence in enumerate(text):
            tokens = model.tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][i])
            
            tags = []
            for j in range(len(tokens)):
                if j < len(predicted_tags[i]):
                    tags.append(ner_tags[predicted_tags[i][j]])
                else:
                    tags.append("O")  
            
            tag_results = list(zip(tokens, tags))
            results['ner'].append(tag_results)
            
            tag_counts = {}
            for _, tag in tag_results:
                if tag not in tag_counts:
                    tag_counts[tag] = 0
                tag_counts[tag] += 1
            
            print(f"Named Entities for sentence {i+1}:")
            print(f"  - Tokens: {len(tokens)}")
            print(f"  - Entity distribution: {tag_counts}")
            
            current_entity = None
            current_text = []
            entities_found = []
            
            for token, tag in tag_results:
                if tag.startswith("B-"):
                    if current_entity:
                        entities_found.append((current_entity, " ".join(current_text)))
                    current_entity = tag[2:]  
                    current_text = [token]
                elif tag.startswith("I-") and current_entity == tag[2:]:
                    current_text.append(token)
                elif tag == "O" and current_entity:
                    entities_found.append((current_entity, " ".join(current_text)))
                    current_entity = None
                    current_text = []
            
            if current_entity:
                entities_found.append((current_entity, " ".join(current_text)))
            
            print(f"  - Entities found: {entities_found}")
    
    return results