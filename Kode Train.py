import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
import albumentations as A  # Tidak perlu import pytorch
import joblib

# Fungsi untuk augmentasi gambar
def augment_image(image):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=40, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomSizedCrop(min_max_height=(100, 200), height=128, width=64, p=0.5),
    ])
    augmented = transform(image=image)
    return augmented['image']

# Fungsi untuk ekstraksi fitur HOG berdasarkan warna oranye
def extract_hog_features(image):
    # Deteksi warna oranye
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


# Fungsi untuk membaca gambar dan label dari folder
def load_data(data_dir, resize_dim=(64, 128)):
    images = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Memproses folder: {folder}")

            # Tentukan label berdasarkan nama folder
            if folder.lower() == 'lengkap':
                label = 2 
            elif folder.lower() == 'vest':
                label = 1  
            elif folder.lower() == 'helm':
                label = 0  
            else:
                continue  # Jika folder bukan helm, vest, atau lengkap, abaikan

            # Memproses setiap gambar dalam folder
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                image = cv2.imread(img_path)
                if image is not None:
                    try:
                        # Resize gambar terlebih dahulu
                        image_resized = cv2.resize(image, resize_dim)

                        # Augmentasi gambar
                        image_augmented = augment_image(image_resized)

                        # Ekstraksi fitur HOG berdasarkan warna oranye
                        features = extract_hog_features(image_augmented)

                        images.append(features)
                        labels.append(label)  # Menambahkan label sesuai folder
                    except Exception as e:
                        print(f"Gagal memproses gambar {img_path}: {e}")
                else:
                    print(f"Gagal membaca gambar: {img_path}")

    return np.array(images), np.array(labels)

# Mempersiapkan dataset
data_dir = r'E:/Kuliah/SMT 5/VISI KOMPUTER/fix/dataset2'  # Ganti dengan path ke folder dataset Anda
if not os.path.exists(data_dir):
    print(f"Folder dataset tidak ditemukan: {data_dir}")
    exit(1)

X, y = load_data(data_dir)

# Membagi dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Konversi ke tipe data yang benar
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.int32)

# Memastikan dimensi data sesuai untuk pelatihan
print(f"Dimensi data latih: {X_train.shape}")
print(f"Dimensi data uji: {X_test.shape}")

# Menyiapkan pipeline model dengan StandardScaler dan SVM
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='linear', random_state=42)
)

# Menentukan parameter untuk hyperparameter tuning
param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

# Hyperparameter tuning menggunakan GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)

# Melatih model dengan GridSearchCV
print("Melatih model dengan hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Menampilkan hasil terbaik dari GridSearchCV
print(f"Best parameters: {grid_search.best_params_}")

# Menyimpan model SVM terbaik
best_model = grid_search.best_estimator_
print("Model terbaik telah dilatih.")

# Memprediksi data uji
y_pred = best_model.predict(X_test)

# Evaluasi hasil
accuracy = np.mean(y_pred == y_test) * 100
print(f'Accuracy: {accuracy:.2f}%')

# Menyimpan model ke dalam format XML menggunakan cv2.FileStorage
model_params = {
    'support_vectors': best_model.named_steps['svc'].support_vectors_,
    'dual_coef': best_model.named_steps['svc'].dual_coef_,
    'intercept': best_model.named_steps['svc'].intercept_,
    'coef_': best_model.named_steps['svc'].coef_,
    'n_support': best_model.named_steps['svc'].n_support_,
    'classes': best_model.named_steps['svc'].classes_,
}

# Menyimpan ke dalam file XML
xml_filename = 'svm_apd_best_model.xml'
with cv2.FileStorage(xml_filename, cv2.FILE_STORAGE_WRITE) as fs:
    for key, value in model_params.items():
        fs.write(key, value)
    
print(f"Model disimpan dalam format XML sebagai '{xml_filename}'")
