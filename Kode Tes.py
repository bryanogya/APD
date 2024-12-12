import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import csv
from datetime import datetime
import os

# Fungsi untuk ekstraksi fitur HOG
def extract_hog_features(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 0, 0])  # Rentang warna oranye
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)

    # Terapkan mask ke gambar grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Ekstraksi fitur HOG
    fd, _ = hog(
        masked_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True
    )
    return fd

# Fungsi untuk mendeteksi warna orange pada gambar
def detect_orange_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([0, 100, 150])
    upper_orange = np.array([100, 255, 255])
    mask = cv2.inRange(hsv_image, lower_orange, upper_orange)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Fungsi untuk memuat model SVM yang sudah dilatih
def load_trained_model(model_path):
    svm_model = cv2.ml.SVM_load(model_path)
    return svm_model

# Fungsi untuk memproses frame kamera dan melakukan prediksi
def process_frame(frame, svm_model):
    resized_frame = cv2.resize(frame, (64, 128))
    frame_with_orange = detect_orange_color(resized_frame)
    features = extract_hog_features(frame_with_orange)
    
    features = features.reshape(1, -1).astype(np.float32)
    _, result = svm_model.predict(features)
    
    return int(result[0][0])

# Fungsi untuk menyimpan log deteksi ke CSV
def log_detection(log_file, timestamp, detected_objects, status):
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, detected_objects, status])

# Main function untuk deteksi live dengan logging
def live_detection(model_path, log_file):
    svm_model = load_trained_model(model_path)
    
    # Membuka kamera eksternal (USB Camera)
    cap = cv2.VideoCapture(0)  # Ganti dengan indeks kamera eksternal Anda, bisa 1, 2, dst.
    
    if not cap.isOpened():
        print("Tidak dapat membuka kamera.")
        return
    
    prev_time = datetime.now()  # Menyimpan waktu sebelumnya untuk log tiap menit
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break
        
        # Proses frame dan prediksi (Hanya sekali per frame)
        label1 = process_frame(frame, svm_model)  # Helm
        label2 = process_frame(frame, svm_model)  # Vest
        
        # Menentukan hasil deteksi
        if label1 == 0 and label2 == 1:  # Helm dan Vest terdeteksi
            detected_objects = "Helm, Vest"
            status = "Atribut Lengkap"
        elif label1 == 0:  # Hanya Helm terdeteksi
            detected_objects = "Helm"
            status = "Atribut Tidak Lengkap"
        elif label2 == 1:  # Hanya Vest terdeteksi
            detected_objects = "Vest"
            status = "Atribut Tidak Lengkap"
        else:  # Tidak ada atribut yang terdeteksi
            detected_objects = "Tidak ada"
            status = "Atribut Tidak Lengkap"
        
        # Tulis log deteksi ke file CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_detection(log_file, timestamp, detected_objects, status)
        
        # Menampilkan hasil deteksi di frame
        cv2.putText(frame, f"Deteksi: {detected_objects}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Menampilkan frame hasil deteksi
        cv2.imshow('Live Detection', frame)
        
        # Cek apakah sudah lewat 1 menit, dan perbarui waktu log
        current_time = datetime.now()
        if (current_time - prev_time).seconds >= 60:
            prev_time = current_time
        
        # Menunggu jika tombol 'q' ditekan untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Menutup kamera dan jendela
    cap.release()
    cv2.destroyAllWindows()

# Membuat file log CSV jika belum ada
log_file = "detection_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Detected Objects", "Status"])

# Menjalankan deteksi live dengan logging
model_path = 'svm_apd_model.xml'  # Ganti dengan path ke model SVM yang sudah dilatih
live_detection(model_path, log_file)
