import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Log dosyasını okuma
log_file_path = "simulated_access.log"
with open(log_file_path, 'r') as file:
    logs = file.readlines()

# Log desenini tanımlama ve ayıklama
log_pattern = r'(?P<ip>\d+\.\d+\.\d+\.\d+)\s-\s-\s\[(?P<datetime>[^\]]+)\]\s"(?P<method>[A-Z]+)\s(?P<url>[^"]+)\sHTTP/[0-9.]+"\s(?P<status>\d+)\s(?P<size>\d+)'
parsed_logs = []

for log in logs:
    match = re.match(log_pattern, log)
    if match:
        parsed_logs.append(match.groupdict())

# DataFrame oluşturma
df = pd.DataFrame(parsed_logs)

# Veri tiplerini kontrol edin
print(df.info())

# Eksik verileri kontrol edin
print(df.isnull().sum())


# Eksik verileri gerekirse doldurun veya kaldırın
df = df.dropna()  # Örneğin, eksik verileri kaldırma


# IP adreslerinin geçerli olup olmadığını kontrol etme
valid_ip = df['ip'].apply(lambda x: re.match(r'\d+\.\d+\.\d+\.\d+', x) is not None)
df = df[valid_ip]

# IP adresleri geçerliyse, mesaj yazdırma
if valid_ip.all():
    print("IP adreslerinin yapılandırması doğru bir şekilde yapıldı.")
else:
    print("Geçersiz IP adresleri bulundu ve çıkarıldı.")


# datetime sütununu datetime formatına dönüştürme
df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%b/%Y:%H:%M:%S +0000')

# Ek özellikler ekleme
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour

# Metin verilerini birleştirme ve TF-IDF vektörizasyonu
text_data = df['url'] + ' ' + df['method']  # Metin verilerini birleştirme
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(text_data)

# TF-IDF matrisini DataFrame'e dönüştürme
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

# Kategorik verilerin one-hot encoding ile dönüştürülmesi
df = pd.get_dummies(df, columns=['status'])

# IP adreslerini sayısal bir değere dönüştürme
df['ip_encoded'] = df['ip'].apply(lambda x: int(''.join([f"{int(octet):03}" for octet in x.split('.')])))

# Vektörleri işlenmiş veriye ekleme
df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)

# Veriyi CSV dosyasına kaydetme
df.to_csv('vectorized_data.csv', index=False)

print("Veri işleme ve vektörizasyon tamamlandı.")

