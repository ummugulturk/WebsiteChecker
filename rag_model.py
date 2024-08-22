import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Örnek log verileri (log_data)
log_data = [
    "User accessed /index.html",
    "User accessed /about.html",
    "User accessed /contact.html",
    "User accessed /login",
    # Diğer log verileri burada...
]

# FAISS indeksini yükleme
def load_faiss_index(index_path='faiss_index.index'):
    index = faiss.read_index(index_path)
    print(f"FAISS Index loaded with dimension: {index.d}")
    return index

# Bilgi alma (retrieval) işlemi
def retrieve_similar_vectors(query_vector, index, k=5):
    assert query_vector.shape[1] == index.d, f"Query vector dimension {query_vector.shape[1]} does not match FAISS index dimension {index.d}"
    distances, indices = index.search(query_vector.reshape(1, -1), k)
    return distances, indices

# Jeneratif model (generation) kurulumu
def setup_generation_model():
    model_name = "t5-small"  # T5 modelini kullanabilirsiniz
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    generator = pipeline('text2text-generation', model=model, tokenizer=tokenizer, max_new_tokens=100)  # 100 token kadar uzun cevaplar üret
    return generator

# Sorgu vektörizasyonu
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query]).toarray().astype(np.float32)

# Sorgu işleme
def process_query(query, index, generator, vectorizer):
    # Query'yi TF-IDF ile vektörize edin
    query_vector = vectorize_query(query, vectorizer)

    # Benzer vektörleri çekin
    distances, indices = retrieve_similar_vectors(query_vector, index)

    # Benzer log kayıtlarını alın
    similar_logs = [log_data[i] for i in indices[0]]

    # Jeneratif model ile yanıt oluşturun
    context = " ".join(similar_logs)
    response = generator(f"Answer the question based on the context: {context}. Question: {query}")

    return response[0]['generated_text']

# Verileri ve modelleri yükleyin
if __name__ == "__main__":
    index = load_faiss_index()
    generator = setup_generation_model()

    # TF-IDF vektörleştiriciyi yükleyin
    vectorizer = TfidfVectorizer()
    df = pd.read_csv('vectorized_data.csv')
    text_data = df['url'] + ' ' + df['method']
    vectorizer.fit(text_data)  # Eğitim

    # Kullanıcıdan sorgu alın
    query = "What are the most accessed pages?"
    response = process_query(query, index, generator, vectorizer)

    print("Response: \n", response)
