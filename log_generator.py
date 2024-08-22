import random
from datetime import datetime, timedelta

# Rastgele IP adresi üretme fonksiyonu
def generate_ip():
    return f"{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"


# Rastgele zaman damgası üretme fonksiyonu
def generate_timestamp(start_year=2020, end_year=2024):
    # Başlangıç ve bitiş tarihlerinin belirlenmesi
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)

    # Rastgele bir tarih seçimi
    random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))

    # Rastgele bir saat ve dakika eklenmesi
    random_time = random_date + timedelta(seconds=random.randint(0, 86400))  # 86400 saniye bir günü temsil eder

    return random_time.strftime('%d/%b/%Y:%H:%M:%S +0000')

# Rastgele log satırı üretme fonksiyonu
def generate_log_line():
    ip = generate_ip()
    timestamp = generate_timestamp()
    method = random.choice(["GET", "POST", "DELETE"])
    url = random.choice(["/index.html", "/about.html", "/contact.html", "/login"])
    status = random.choice([200, 404, 500])
    size = random.randint(1000, 5000)
    log_line = f'{ip} - - [{timestamp}] "{method} {url} HTTP/1.1" {status} {size}'
    return log_line

# Log dosyasına 1000 satır yazma
with open("simulated_access.log", "w") as log_file:
    for _ in range(1000):
        log_file.write(generate_log_line() + "\n")

print("Log dosyası oluşturuldu: simulated_access.log")
