from transformers import pipeline
#enter the task for pipleine
classifier = pipeline("sentiment-analysis")
#apply classifier and enter the data 
res =classifier("I have been waiting for a HuggingFace course all my life")
#label = positive with a score

print(res)
