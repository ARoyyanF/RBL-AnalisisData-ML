import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import time # Import modul time

def preprocess_data(file_path="solar_system_positions_with_velocity.csv"):
    """
    Memuat, membersihkan, dan memproses data untuk klasifikasi objek tata surya spesifik,
    dengan fokus pada Barycenter dan Moon.
    """
    try:
        df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File '{file_path}' tidak ditemukan. Pastikan file ada di direktori yang benar.")
        return None, None

    print("Dataset awal dimuat.")
    print(f"Jumlah baris awal: {len(df_original)}")

    def categorize_celestial_object(row):
        """Mengategorikan objek dan mengekstrak nama target untuk klasifikasi."""
        name_str = str(row['name'])
        try:
            naif_id = int(row['naif_id'])
        except ValueError:
            print(f"Peringatan: NAIF ID tidak valid '{row['naif_id']}' untuk nama '{name_str}'. Mengklasifikasikan sebagai 'Other'.")
            return "Other", name_str.upper() # Penanganan jika naif_id bukan integer

        name_upper = name_str.upper()

        # Ekstrak nama spesifik dari format "ID NAMA_OBJEK"
        parts = name_upper.split()
        # Default specific name adalah bagian setelah ID, atau nama lengkap jika tidak ada ID numerik di awal
        specific_name_extracted = " ".join(parts[1:]) if (len(parts) > 1 and parts[0].isdigit()) else name_upper
        
        # Hapus ID numerik dari awal specific_name_extracted jika masih ada (untuk konsistensi)
        # Ini mungkin redundan jika split sudah benar, tapi sebagai pengaman.
        sub_parts = specific_name_extracted.split()
        if len(sub_parts) > 0 and sub_parts[0].isdigit(): # Cek jika bagian pertama dari hasil ekstrak adalah digit
             # Ini seharusnya tidak terjadi jika logika split di atas sudah benar,
             # tapi jika terjadi, gunakan nama asli tanpa ID awal.
             pass # Biarkan specific_name_extracted apa adanya jika sudah diproses

        # 1. Barycenters (NAIF IDs 1-9)
        if 1 <= naif_id <= 9:
            return "Barycenter", specific_name_extracted # e.g., "MERCURY BARYCENTER"

        # 2. Sun (NAIF ID 10) - akan difilter nanti
        if naif_id == 10:
            return "Sun", "SUN"

        # 3. Planets (NAIF IDs 199, 299, ..., 999) - akan difilter nanti
        planet_naif_ids = [199, 299, 399, 499, 599, 699, 799, 899, 999] # Pluto termasuk
        if naif_id in planet_naif_ids:
            return "Planet", specific_name_extracted # e.g., "MERCURY"
        
        # 4. Jika bukan Barycenter, Sun, atau Planet, diasumsikan sebagai Moon.
        # Ini berdasarkan asumsi bahwa dataset utama berisi objek-objek ini.
        # specific_name_extracted akan menghasilkan "MOON", "MIMAS", "TITAN", dll.
        return "Moon", specific_name_extracted

    # Pastikan kolom 'name' dan 'naif_id' ada
    if not {'name', 'naif_id'}.issubset(df_original.columns):
        print("Kolom 'name' atau 'naif_id' tidak ditemukan dalam file CSV.")
        return None, None
        
    # Terapkan fungsi kategorisasi untuk membuat kolom baru
    categorized_results = df_original.apply(
        lambda row: pd.Series(categorize_celestial_object(row)), axis=1
    )
    categorized_results.columns = ['object_type', 'target_for_classification']
    
    # Gabungkan hasil kategorisasi dengan DataFrame asli
    df_processed = pd.concat([df_original, categorized_results], axis=1)

    # Filter untuk hanya menyertakan Barycenter dan Moon
    df_selected_objects = df_processed[df_processed['object_type'].isin(['Barycenter', 'Moon'])].copy()

    if df_selected_objects.empty:
        print("Tidak ada data Barycenter atau Moon yang ditemukan setelah proses filter.")
        return None, None
    
    print(f"Jumlah baris setelah filter (hanya Barycenter & Moon): {len(df_selected_objects)}")
    print("Contoh nama target yang akan diklasifikasikan (Barycenter & Moon):", df_selected_objects['target_for_classification'].unique()[:20])

    features_cols = ['x_au', 'y_au', 'z_au', 'vx_au_per_day', 'vy_au_per_day', 'vz_au_per_day']
    
    # Periksa apakah semua kolom fitur ada
    if not all(col in df_selected_objects.columns for col in features_cols):
        missing_cols = [col for col in features_cols if col not in df_selected_objects.columns]
        print(f"Satu atau lebih kolom fitur tidak ditemukan: {missing_cols}")
        return None, None

    X = df_selected_objects[features_cols]
    y_raw = df_selected_objects['target_for_classification']

    # Pembersihan akhir untuk y_raw (menghapus NaN atau string kosong)
    if y_raw.isnull().any() or (y_raw == '').any():
        print("Peringatan: Ada nama target kosong atau NaN sebelum pembersihan akhir. Membersihkan...")
        valid_indices = y_raw.dropna().index
        # Pastikan untuk hanya mengambil indeks di mana y_raw bukan string kosong
        non_empty_string_indices = y_raw[y_raw != ''].index
        valid_indices = valid_indices.intersection(non_empty_string_indices)
        
        y_raw = y_raw.loc[valid_indices]
        X = X.loc[valid_indices] # Pastikan X juga difilter dengan indeks yang sama
    
    if y_raw.empty:
        print("Tidak ada target yang valid setelah pembersihan nama target akhir.")
        return None, None

    print(f"Jumlah kelas akhir (objek spesifik) untuk diklasifikasi: {y_raw.nunique()}")
    return X, y_raw

def main():
    """
    Fungsi utama untuk menjalankan seluruh alur kerja klasifikasi.
    """
    X, y_raw = preprocess_data()

    if X is None or y_raw is None:
        print("Preprocessing gagal. Keluar dari program.")
        return

    if X.empty or y_raw.empty:
        print("Tidak ada fitur atau target setelah preprocessing. Keluar dari program.")
        return

    # Membuat direktori output jika belum ada
    output_dir = "output_klasifikasi_spesifik_lengkap" # Nama direktori bisa disesuaikan jika perlu
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput akan disimpan di direktori: {os.path.abspath(output_dir)}")

    # Encoding variabel target
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_
    num_classes = len(class_names)

    if num_classes == 0:
        print("Tidak ada kelas yang ditemukan setelah label encoding. Periksa kolom target.")
        return
        
    print(f"\nJumlah kelas (objek spesifik) setelah encoding: {num_classes}")
    # print(f"Nama kelas terenkode: {class_names}") # Bisa sangat panjang

    # Pembagian data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
    print(f"Ukuran data latih: {X_train.shape}, Ukuran data uji: {X_test.shape}")

    # Penskalaan Fitur
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Fitur telah diskalakan.")

    models = {
        "SVM": SVC(random_state=42, class_weight='balanced', C=1.0),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    }

    results_summary = {}
    classification_reports_dict = {}
    confusion_matrices_plot_paths = {}

    print("\n--- Memulai Pelatihan dan Evaluasi Model ---")

    for name, model in models.items():
        print(f"\n--- Melatih dan Mengevaluasi {name} ---")
        start_time = time.time() 
        if name == "SVM" or name == "KNN":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else: 
            model.fit(X_train, y_train) 
            y_pred = model.predict(X_test)
        end_time = time.time() 
        training_time = end_time - start_time
        print(f"Waktu pelatihan {name}: {training_time:.2f} detik")

        accuracy = accuracy_score(y_test, y_pred)
        
        unique_labels_in_y_test_pred = np.unique(np.concatenate((y_test, y_pred)))
        
        # Filter class_names dan labels untuk classification_report berdasarkan label yang benar-benar ada
        # Ini penting jika beberapa kelas tidak muncul di y_test atau y_pred (meskipun stratify membantu)
        current_labels_for_report = [l for l in np.arange(num_classes) if l in unique_labels_in_y_test_pred]
        current_class_names_for_report = [class_names[l] for l in current_labels_for_report if l < len(class_names)]


        if not current_class_names_for_report: 
            print(f"Tidak ada label yang cocok untuk laporan klasifikasi model {name}. Melanjutkan...")
            report_dict = {}
            report_str = "Tidak ada label yang cocok untuk laporan."
        else:
            report_dict = classification_report(y_test, y_pred, target_names=current_class_names_for_report, labels=current_labels_for_report, zero_division=0, output_dict=True)
            report_str = classification_report(y_test, y_pred, target_names=current_class_names_for_report, labels=current_labels_for_report, zero_division=0)

        results_summary[name] = {"accuracy": accuracy, "report_dict": report_dict, "training_time": training_time}
        classification_reports_dict[name] = report_str

        print(f"Akurasi {name}: {accuracy:.4f}")

        # Confusion Matrix
        # Untuk CM, kita gunakan semua class_names asli agar ukurannya konsisten
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(num_classes)) 
        plt.figure(figsize=(max(15, num_classes // 1.5 if num_classes > 0 else 15), 
                           max(12, num_classes // 2 if num_classes > 0 else 12)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, 
                    annot_kws={"size": 8 if num_classes > 20 else 10}) 
        plt.title(f'Confusion Matrix untuk {name} (Objek Spesifik)', fontsize=16)
        plt.ylabel('Label Sebenarnya', fontsize=14)
        plt.xlabel('Label Prediksi', fontsize=14)
        plt.xticks(rotation=90, ha='right', fontsize=min(10, 200/num_classes if num_classes > 1 else 10) ) 
        plt.yticks(rotation=0, fontsize=min(10, 200/num_classes if num_classes > 1 else 10))
        plt.tight_layout()
        cm_plot_filename = os.path.join(output_dir, f"cm_spesifik_{name.replace(' ', '_')}.png")
        plt.savefig(cm_plot_filename)
        confusion_matrices_plot_paths[name] = cm_plot_filename
        print(f"Plot confusion matrix {name} disimpan sebagai: {cm_plot_filename}")
        plt.close()

    # --- Perbandingan Model ---
    print("\n--- Perbandingan Akurasi dan Waktu Pelatihan Model ---")
    comparison_data = []
    for model_name_loop, metrics in results_summary.items(): # Ganti nama variabel loop
        comparison_data.append({
            "Algoritma": model_name_loop,
            "Akurasi": metrics["accuracy"],
            "Waktu Training (s)": metrics.get("training_time", "N/A") 
        })

    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.sort_values(by="Akurasi", ascending=False))

    # --- Contoh Prediksi ---
    true_label_name = "N/A" 
    new_data_point_unscaled_df = pd.DataFrame() # Inisialisasi
    predicted_category_label = "N/A" # Inisialisasi

    if not comparison_df.empty:
        best_model_name = comparison_df.sort_values(by="Akurasi", ascending=False).iloc[0]["Algoritma"]
        best_model_instance = models[best_model_name]
        print(f"\n--- Contoh Prediksi menggunakan {best_model_name} ---")

        if not X_test.empty: # Pastikan X_test tidak kosong
            # Ambil sampel dari X_test (DataFrame asli sebelum scaling)
            idx_sample = np.random.randint(0, len(X_test))
            new_data_point_unscaled_df = X_test.iloc[[idx_sample]] # Ini adalah DataFrame
            
            # Dapatkan label sebenarnya dari y_test (yang sudah di-encode)
            true_label_encoded = y_test[X_test.index[idx_sample]] # Gunakan indeks asli dari X_test
            true_label_name = label_encoder.inverse_transform([true_label_encoded])[0]

            print(f"Data baru (diambil dari data uji, tidak diskalakan):\n{new_data_point_unscaled_df.to_string(index=False)}")
            print(f"Label sebenarnya untuk data ini: {true_label_name}")

            if best_model_name == "SVM" or best_model_name == "KNN":
                # Untuk prediksi, gunakan data yang sudah diskalakan dari X_test_scaled
                # Kita perlu menemukan baris yang sesuai di X_test_scaled
                # Cara termudah adalah dengan menggunakan indeks yang sama jika X_test dan X_train_scaled dibuat dengan cara yang konsisten
                # atau re-scale sampel yang diambil dari X_test
                new_data_point_for_prediction_scaled = scaler.transform(new_data_point_unscaled_df)
                predicted_category_encoded = best_model_instance.predict(new_data_point_for_prediction_scaled)
                predicted_category_label = label_encoder.inverse_transform(predicted_category_encoded)[0]
            else: # Random Forest (dilatih pada data tidak diskalakan)
                predicted_category_encoded = best_model_instance.predict(new_data_point_unscaled_df)
                predicted_category_label = label_encoder.inverse_transform(predicted_category_encoded)[0]

            print(f"Prediksi kategori untuk data baru menggunakan {best_model_name}: {predicted_category_label}")
        else:
            print("Data uji kosong, tidak dapat melakukan contoh prediksi.")
    else:
        print("Tidak ada model yang dievaluasi, contoh prediksi dilewati.")
        best_model_name = "N/A"


    # --- Simpan Ringkasan Laporan ---
    summary_report_path = os.path.join(output_dir, "laporan_klasifikasi_spesifik_lengkap.txt")
    with open(summary_report_path, "w") as f:
        f.write("Klasifikasi Objek Tata Surya Spesifik (Barycenter & Moon) - Ringkasan Laporan Lengkap\n") # Judul disesuaikan
        f.write("=====================================================================================\n\n")
        f.write(f"Jumlah total kelas (objek spesifik): {num_classes}\n")
        if num_classes < 50 and num_classes > 0: 
             f.write(f"Nama kelas: {', '.join(class_names)}\n\n")
        elif num_classes > 0 :
            f.write(f"Nama kelas: (terlalu banyak untuk dicetak, {num_classes} kelas)\n\n")
        else:
            f.write("Nama kelas: Tidak ada kelas yang valid.\n\n")


        f.write("--- Perbandingan Performa Model (Akurasi & Waktu Training) ---\n") 
        if not comparison_df.empty:
            f.write(comparison_df.sort_values(by="Akurasi", ascending=False).to_string(index=False))
        else:
            f.write("Tidak ada data perbandingan model.")
        f.write("\n\n")

        f.write("--- Laporan Klasifikasi Detail dan Path Confusion Matrix ---\n")
        for name_model_loop_report in models.keys(): 
            if name_model_loop_report in results_summary:
                f.write(f"\n** Algoritma: {name_model_loop_report} **\n")
                f.write(f"Akurasi: {results_summary[name_model_loop_report]['accuracy']:.4f}\n")
                f.write(f"Waktu Pelatihan: {results_summary[name_model_loop_report].get('training_time', 'N/A'):.2f} detik\n") 
                f.write("Laporan Klasifikasi Lengkap:\n")
                f.write(classification_reports_dict[name_model_loop_report])
                f.write("\n")
                if name_model_loop_report in confusion_matrices_plot_paths:
                    f.write(f"Plot Confusion Matrix: {os.path.abspath(confusion_matrices_plot_paths[name_model_loop_report])}\n")
                else:
                    f.write("Plot Confusion Matrix: Tidak tersedia.\n")
                f.write("---------------------------------------------------\n")
            else:
                f.write(f"\n** Algoritma: {name_model_loop_report} **\n")
                f.write("Hasil tidak tersedia.\n")
                f.write("---------------------------------------------------\n")

        f.write("\n--- Contoh Prediksi ---")
        if not comparison_df.empty and not X_test.empty and not new_data_point_unscaled_df.empty:
            f.write(f"\nMenggunakan model: {best_model_name}\n")
            f.write(f"Data baru (diambil dari data uji, tidak diskalakan):\n{new_data_point_unscaled_df.to_string(index=False)}\n")
            f.write(f"Label sebenarnya untuk data ini: {true_label_name}\n") 
            f.write(f"Prediksi kategori: {predicted_category_label}\n")
        else:
            f.write("\nTidak dapat melakukan contoh prediksi (data uji kosong, tidak ada model terbaik, atau sampel tidak terpilih).\n")

    print(f"\nRingkasan laporan disimpan sebagai: {summary_report_path}")
    print(f"Semua plot disimpan di direktori: {os.path.abspath(output_dir)}")
    print("\nProses selesai.")

if __name__ == "__main__":
    main()

