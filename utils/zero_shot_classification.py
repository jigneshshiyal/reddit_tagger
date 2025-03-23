from transformers import pipeline

# Load the Zero-Shot Classification Model (DeBERTa)
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")

# Function for classifying text
def get_zero_shot_classification(text, labels):
    result = classifier(text, candidate_labels=labels, multi_label=False)
    return result["labels"][0]  # Get highest confidence label
