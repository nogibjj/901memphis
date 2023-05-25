from transformers import AutoTokenizer
from datasets import load_dataset, load_metric
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
import torch

#load dataset
dataset = load_dataset("snli")


#load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')


features = tokenizer(['A soccer game with multiple males playing','A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something','An older and younger man smiling.', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)

# # can use if needed to reduce memory usage and training time
# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))


# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )

# trainer.train() # train the model