from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Mean pooling for sentence embedding


def get_bert_embedding_batch(texts, batch_size=32):
    if isinstance(texts, str):
        texts = [texts]  # Convert single string to list

    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            outputs = model(**inputs).last_hidden_state  # (batch_size, seq_len, hidden_dim)
            
            batch_embeddings = outputs.mean(dim=1)  # (batch_size, hidden_dim)
            embeddings.extend([tmp.squeeze().numpy() for tmp in batch_embeddings.cpu()])

    return embeddings  # Return as a single tensor



def convert_into_bow(corpus):    
    vectorizer = CountVectorizer()
    vectorizer.fit(corpus) 
    return vectorizer.transform(corpus).toarray()

def convert_tfidf(text):
  tfidf = TfidfVectorizer()
  X_tfidf = tfidf.fit_transform(text)
  return X_tfidf