Web Trafik Logları ile Soru-Cevap Sistemi
Proje Açıklaması
Bu proje, bir web sitesi için oluşturulan trafik loglarını kullanarak bir soru-cevap (Q&A) sistemi geliştirir. Sistem, kullanıcıların doğal dildeki sorularına yanıt vermek için Retrieval-Augmented Generation (RAG) modelini kullanır. Proje, veri hazırlığından model entegrasyonuna kadar tüm süreci kapsar.

Proje Adımları
Veri Oluşturma

log_generator.py: Rastgele IP adresleri, zaman damgaları, HTTP yöntemleri ve durum kodları içeren 1000 satırlık log verileri oluşturur. Log dosyası simulated_access.log olarak kaydedilir.
Veri İşleme ve Vektörleştirme

data_process.py: Log dosyasını okur, verileri temizler ve dönüştürür. IP adresleri, HTTP yöntemleri ve durum kodları gibi özellikleri TF-IDF vektörizasyonu ile vektörlere dönüştürür ve sonuçları vectorized_data.csv dosyasına kaydeder.
FAISS Veritabanı Oluşturma

vector_database.py: TF-IDF vektörlerini FAISS kullanarak bir veri tabanına yükler. FAISS, benzer vektörleri hızlı bir şekilde aramak için kullanılır ve veritabanı faiss_index.index olarak kaydedilir.
Soru-Cevap Sistemi

deneme.py: Kullanıcının sorgularını işler. FAISS ile benzer log verilerini arar ve T5 modelini kullanarak uygun yanıtları oluşturur. Ayrıca, yanıtların doğruluğunu değerlendirmek için F1 skoru hesaplar. Yanıtlar ve F1 skoru ekrana yazdırılır.
F1 Skoru Hesaplama

f1_skor.py: Tahminlerin doğruluğunu ölçen bir F1 skoru hesaplar. Doğru cevaplar ile tahminler arasındaki benzerliği değerlendirir.
Kurulum ve Çalıştırma
Gerekli kütüphaneleri yükleyin:

bash
Kodu kopyala
pip install pandas numpy faiss-cpu transformers scikit-learn
Projeyi sırasıyla çalıştırın:

bash
Kodu kopyala
python log_generator.py
python data_process.py
python vector_database.py
python deneme.py
python f1_skor.py
Kullanım
log_generator.py ile log verilerini oluşturun.
data_process.py ile bu verileri işleyip vektörleştirin.
vector_database.py ile vektörleri FAISS veri tabanına yükleyin.
deneme.py ile sorgulara yanıt alın ve F1 skorunu hesaplayın.
f1_skor.py ile yanıtların doğruluğunu değerlendirin.
