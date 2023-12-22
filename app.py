# Saya akan membuat sebuah aplikasi untuk memprediksi jenis tanaman
# berdasarkan beberapa fitur yang ada pada dataset
# Nitrogen (N), Phosphorus (P), dan Potassium (K), Temperatur, humidity, ph, rainfall
# Saya akan menggunakan algoritma KNN untuk memprediksi jenis tanaman
# kemudian di deploy menggunakan streamlit

# Import library
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Judul
st.title('Aplikasi Prediksi Jenis Tanaman')

# Deskripsi
st.write("""Aplikasi ini dapat memprediksi jenis tanaman
    berdasarkan beberapa atribut yang dibutuhkan
    """)

st.sidebar.header('Silahkan input data')

# Fungsi untuk menerima input dari user


def input_user():
    N = st.sidebar.number_input('Nitrogen (N)', 0.0, 200.0, 0.0)
    P = st.sidebar.number_input('Phosphorus (P)', 0.0, 200.0, 0.0)
    K = st.sidebar.number_input('Potassium (K)', 0.0, 200.0, 0.0)
    temperature = st.sidebar.number_input('Temperature', 0.0, 50.0, 0.0)
    humidity = st.sidebar.number_input('Humidity', 0.0, 100.0, 0.0)
    ph = st.sidebar.number_input('ph', 0.0, 14.0, 0.0)
    rainfall = st.sidebar.number_input('Rainfall', 0.0, 300.0, 0.0)

    # Memasukkan nilai fitur ke dalam dictionary
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }

    # Mengubah dictionary menjadi dataframe
    features = pd.DataFrame(data, index=[0])
    return features


# Menjalankan fungsi input_user()
df = input_user()

# Menampilkan nilai fitur
st.subheader('Data yang telah di input')
st.write(df)

# Membaca dataset
data = pd.read_csv('Dataset.csv')

# Memisahkan fitur dan label
X = data.drop(columns=['label'])
y = data['label']

# Membagi data menjadi data train dan data test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Membuat model KNN
model = KNeighborsClassifier(n_neighbors=3)
knn = model.fit(X_train, y_train)

# Tombol untuk memprediksi jenis tanaman
btn = st.button('Prediksi')

# Membuat kondisi jika tombol ditekan
if btn:
    # Memeriksa apakah semua nilai input adalah 0
    if all(value == 0 for value in df.values.flatten()):
        st.warning('Silahkan input data terlebih dahulu.')
    # Memberikan peringatan jika masih ada atribut yang bernilai 0
    elif any(value == 0 for value in df.values.flatten()):
        st.warning('Masih ada atribut yang bernilai 0.')
    # Jika semua nilai input bukan 0
    else:

        # Memprediksi jenis tanaman
        prediksi = np.array([df['N'], df['P'], df['K'], df['temperature'],
                             df['humidity'], df['ph'], df['rainfall']])
        prediksi = prediksi.reshape(1, -1)
        hasil = knn.predict(prediksi)
        st.subheader('Prediksi')
        st.write(hasil)
