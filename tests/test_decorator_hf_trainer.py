import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, TrainingArguments, Trainer
from datasets import Dataset
from core.analytical_decorator import analytical_solver

# Wrap the HF model in our analytical solver
@analytical_solver(lr=30.0, lr_decay=0.9, lam=1e-3, momentum=0.5)
class AnalyticalBertClassifier(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        # Use a tiny BERT config for fast testing
        config = BertConfig(
            vocab_size=1000,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_labels = num_labels
        
    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use pooled output for classification
        pooled_output = outputs.pooler_output 
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits}

def main():
    print("Initializing Analytical Hugging Face Model...")
    model = AnalyticalBertClassifier()
    
    print("Checking if analytical hooks attached...")
    assert hasattr(model, '_analytical_state'), "Decorator failed!"
    
    # Create fake dataset
    data = {
        "input_ids": torch.randint(0, 1000, (64, 16)).tolist(),
        "labels": torch.randint(0, 2, (64,)).tolist()
    }
    dataset = Dataset.from_dict(data)
    
    training_args = TrainingArguments(
        output_dir="./test_trainer",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-3, # Adam LR for non-analytical layers (LayerNorm/Embedding)
        logging_steps=1,
        report_to="none",
        use_cpu=True # Force CPU for fast simple testing
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    print("Starting Hugging Face Trainer loop...")
    # The Trainer will call loss.backward() automatically!
    # Our hooks will transparently intercept gradients to solve Dense/Conv layers analytically.
    trainer.train()
    
    print("Trainer loop completed successfully! The Analytical Solver integrates flawlessly with HF Trainer.")

if __name__ == "__main__":
    main()
