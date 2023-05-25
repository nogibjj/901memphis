# from transformers import pipeline
# from transformers import 
# classifier = pipeline("zero-shot-classification")
# res =classifier(
#     "This is a course about Python list comprehension", 
#     candidate_labels=["education", "politics", "business"], 
# )

# print(res)

# from transformers import pipeline

# classifier = pipeline("zero-shot-classification", model='cross-encoder/nli-roberta-base')

# sent = "Apple just announced the newest iPhone X"
# candidate_labels = ["technology", "sports", "politics"]
# res = classifier(sent, candidate_labels)
# print(res)

# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch

# model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
# tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')

# features = tokenizer(['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

# model.eval()
# with torch.no_grad():
#     scores = model(**features).logits
#     label_mapping = ['contradiction', 'entailment', 'neutral']
#     labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
#     print(labels)

# from sentence_transformers import CrossEncoder
# model = CrossEncoder('cross-encoder/nli-roberta-base')
# scores = model.predict([('A man is eating pizza', 'A man eats something'), ('A black race car starts up in front of a crowd of people.', 'A man is driving down a lonely road.')])

# #Convert scores to labels
# label_mapping = ['contradiction', 'entailment', 'neutral']
# labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
# print(labels)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-roberta-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-roberta-base')

features = tokenizer(['A soccer game with multiple males playing','A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something','An older and younger man smiling.', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)
