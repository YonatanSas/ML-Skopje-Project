# Antisemitic Tweet Detector

## Overview
This project implements a machine learning solution for detecting antisemitic content in tweets.
Using the RoBERTa-large language model fine-tuned on labeled tweet data, the system can analyze text and determine if it contains antisemitic content with high accuracy.

## Features
- Text preprocessing
- Fine-tuned RoBERTa model for antisemitism detection 
- Oversampling to handle class imbalance
- Evaluation metrics focused on antisemitic content detection
- Easy-to-use prediction function for new tweets

## Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Evaluate

## Dataset
The model is trained on a labeled dataset of tweets, where each tweet is classified as either antisemitic (1) or non-antisemitic (0).
The dataset requires preprocessing to handle typical social media text features like mentions, hashtags, and URLs.

## Data Preprocessing
Text preprocessing includes:
- Converting to lowercase
- Removing URLs
- Removing mentions (@username)
- Preserving hashtag content while removing # symbol
- Removing numbers and special characters
- Keeping essential punctuation (., !, ?)
- Removing extra whitespace

## Model Architecture
- Base model: RoBERTa-large
- Output: Binary classification (antisemitic/non-antisemitic)
- Training parameters:
  - Learning rate: 2e-5
  - Batch size: 64
  - Training epochs: 2
  - Weight decay: 0.01
  - Early stopping based on F1 score

## Training Process
1. Data is loaded and preprocessed
2. Class imbalance is addressed through oversampling of minority class
3. Tokenization is performed using RoBERTa tokenizer
4. Model is fine-tuned with custom evaluation metrics
5. Best model is saved based on F1 score for antisemitic class detection

## Evaluation Results
The model is evaluated on a held-out test set (20% of the data) with the following metrics:
- Overall accuracy
- Precision for antisemitic content
- Recall for antisemitic content
- F1 score for antisemitic content
- Confusion matrix visualization

## Using the Model
To use the trained model for prediction:

```python
def predict_tweet(tweet_text, model):
    """
    Predicts whether a tweet is antisemitic or not.
    
    Parameters:
    tweet_text (str): The text of the tweet to analyze
    model: The trained RoBERTa model
    
    Returns:
    tuple: (is_antisemitic (bool), confidence (float), detailed_results (dict))
    """
    import torch
    from transformers import RobertaTokenizerFast
    import re
    
    # Load tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained("RoBERTa-large")
    
    # Set model to evaluation mode
    model.eval()
    
    # Text cleaning function
    def clean_text(text):
        if text is None or not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[0-9]+', '', text)
        text = re.sub(r'[\(\)\[\]\{\}]', '', text)
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Clean the tweet
    cleaned_tweet = clean_text(tweet_text)
    
    # Tokenize
    inputs = tokenizer(cleaned_tweet, return_tensors="pt", truncation=True, padding=True)

    # Move tensors to model's device (GPU or CPU)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Make prediction without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract model prediction
    logits = outputs.logits
    
    # Move to CPU before converting to NumPy
    probabilities = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    # Get predicted class and confidence
    predicted_class = int(torch.argmax(logits, dim=1).item())
    confidence = float(probabilities[predicted_class])
    
    # Return results with interpretation
    is_antisemitic = predicted_class == 1
    
    return is_antisemitic, confidence, {
        "cleaned_text": cleaned_tweet,
        "probabilities": {
            "non_antisemitic": float(probabilities[0]),
            "antisemitic": float(probabilities[1])
        },
        "prediction": "antisemitic" if is_antisemitic else "non-antisemitic"
    }
```

### Example Usage:
```python
# Example tweets
tweet_example1 = "I love Israel and Jewish culture."
tweet_example2 = "Jews control the media and the banks."

# Make predictions
result1 = predict_tweet(tweet_example1, model)
result2 = predict_tweet(tweet_example2, model)

print(f"'{tweet_example1}' --> {result1[2]['prediction']} (confidence: {result1[1]:.2%})")
print(f"'{tweet_example2}' --> {result2[2]['prediction']} (confidence: {result2[1]:.2%})")
```

## Technical Notes
- When using the model on GPU, make sure to transfer tensors to CPU before converting to NumPy arrays
- For batch processing, consider implementing a more efficient version of the prediction function

## Future Improvements
- Experiment with different transformer models like BERT, DistilBERT
- Implement multilingual detection capabilities
- Add explainability features to highlight problematic parts of text
- Deploy as a web service or API

## Ethical Considerations
This model is intended to help identify harmful content, but should be used with human oversight.
False positives are possible, and context matters in determining antisemitism.
The system should be part of a broader content moderation strategy, not a standalone solution.

## Acknowledgments
- Hugging Face for the Transformers library
- The RoBERTa team for the base model
- Contributors to the dataset
