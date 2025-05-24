from transformers import pipeline

model_path = "citizenlab/twitter-xlm-roberta-base-sentiment-finetunned"

sentiment_classifier = pipeline("text-classification", model=model_path, tokenizer=model_path)
print("Starting prediction...")
print(sentiment_classifier("this is a lovely message"))
print(sentiment_classifier("you are an idiot and you and your family should go back to your country"))
