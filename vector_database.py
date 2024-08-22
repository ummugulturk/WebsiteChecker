import faiss
import numpy as np
import pandas as pd

# FAISS ile veritabanı yükleme
def load_faiss_index(vectors, index_path='faiss_index.index'):
    # Vektörlerin boyutunu almak
    dimension = vectors.shape[1]

    # FAISS Index oluşturma
    index = faiss.IndexFlatL2(dimension)

    # Vektörleri ekleme
    index.add(vectors)

    # FAISS Index dosyasına kaydetme
    faiss.write_index(index, index_path)
    print(f"FAISS index saved at {index_path}")

# Verileri yükleme (örnek veri yolu)
if __name__ == "__main__":
    # Vektörize edilmiş verileri yükleme
    df = pd.read_csv('vectorized_data.csv')  # Vektörize edilmiş verilerin olduğu CSV dosyasını yükleme
    vector_columns = ['about', 'contact', 'delete', 'get', 'html', 'index', 'login', 'post']
    vectors = df[vector_columns].to_numpy().astype(np.float32)

    # FAISS ile yükleme
    load_faiss_index(vectors)
