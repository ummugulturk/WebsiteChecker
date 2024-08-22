import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from f1_score import average_f1_score  # F1 skorunu hesaplayan modül

# Vektörize edilmiş veri CSV dosyasını okuma
df = pd.read_csv('vectorized_data.csv')

# Gerçek log verilerini burada işleyin
log_data = df[['method', 'url', 'status_200', 'status_404', 'status_500', 'size']].apply(
    lambda row: f"Method: {row['method']}, URL: {row['url']}, Status 200: {row['status_200']}, Status 404: {row['status_404']}, Status 500: {row['status_500']}, Size: {row['size']}", axis=1
).tolist()

def load_faiss_index(index_path='faiss_index.index'):
    index = faiss.read_index(index_path)
    print(f"FAISS Index loaded with dimension: {index.d}")
    return index

def retrieve_similar_vectors(query_vector, index, k=5):
    assert query_vector.shape[1] == index.d, f"Query vector dimension {query_vector.shape[1]} does not match FAISS index dimension {index.d}"
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    return distances, indices

def setup_generation_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=150)
    return generator

def vectorize_query(query, vectorizer):
    return vectorizer.transform([query]).toarray().astype(np.float32)

def process_query(query, index, generator, vectorizer):
    query_vector = vectorize_query(query, vectorizer)
    distances, indices = retrieve_similar_vectors(query_vector, index, k=10)

    if len(indices[0]) == 0:
        return "No similar logs found."

    similar_logs = []
    for i in indices[0]:
        if i < len(log_data):
            similar_logs.append(log_data[i])
        else:
            print(f"Index {i} out of range for log_data.")

    if not similar_logs:
        return "No similar logs found."

    context = " ".join(similar_logs)
    print(f"Context: {context}")
    response = generator(f"Answer the question based on the context: {context}. Question: {query}")

    return response[0]['generated_text']

if __name__ == "__main__":
    index = load_faiss_index()
    generator = setup_generation_model()

    vectorizer = TfidfVectorizer()
    text_data = df['url'] + ' ' + df['method']
    vectorizer.fit(text_data)

    query = "What are the most accessed pages?"
    response = process_query(query, index, generator, vectorizer)

    print("Response: \n", response)

    # Örnek doğru cevaplar (ground truth) ve tahminler (predictions)
    predictions = [response]
    ground_truths = ["The most accessed page is /home."]

    # Ortalama F1 skoru hesaplama
    avg_f1_score = average_f1_score(predictions, ground_truths)
    print(f"Average F1 Score: {avg_f1_score:.2f}")

print("Log Data Length:", len(log_data))
print("Sample Log Data:", log_data[:5])
