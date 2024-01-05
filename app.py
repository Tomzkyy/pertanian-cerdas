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

st.set_page_config(page_title='Aplikasi Prediksi Jenis Tanaman', page_icon='🌱')

# Judul dengan spacing
st.markdown("<h1 style='text-align: center; color: black;'>Aplikasi Prediksi Jenis Tanaman</h1>",
            unsafe_allow_html=True)

# Deskripsi with spacing
st.markdown("<div style='text-align: center; margin-bottom: 20px;'>Aplikasi ini dapat memprediksi jenis tanaman berdasarkan beberapa faktor yang dibutuhkan.</div>", unsafe_allow_html=True)


# Fungsi untuk menerima input dari user


def input_user():
    N = st.number_input('Nitrogen (N)', 0.0, 500.0, 0.0)
    P = st.number_input('Phosphorus (P)', 0.0, 500.0, 0.0)
    K = st.number_input('Potassium (K)', 0.0, 500.0, 0.0)
    temperature = st.number_input('Temperature', 0.0, 50.0, 0.0)
    humidity = st.number_input('Humidity', 0.0, 100.0, 0.0)
    ph = st.number_input('ph', 0.0, 14.0, 0.0)
    rainfall = st.number_input('Rainfall', 0.0, 300.0, 0.0)

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
btn = st.button('Prediksi', type='primary')

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
        knn.fit(X_train, y_train)
        hasil = knn.predict(prediksi)

        print(hasil)
        st.subheader(str(hasil[0]))

        # Menampilkan gambbar tanaman seusai prediksi
        if hasil[0] == 'rice':
            st.image('beras.jpeg')
        elif hasil[0] == 'maize':
            st.image('jagung.jpeg')
        elif hasil[0] == 'chickpea':
            st.image('buncis.jpeg')
        elif hasil[0] == 'kidneybeans':
            st.image('kacang-merah.jpeg')
        elif hasil[0] == 'pigeonpeas':
            st.image('kacang-polong.jpeg')
        elif hasil[0] == 'mothbeans':
            st.image('kacang-ngengat.jpeg')
        elif hasil[0] == 'mungbean':
            st.image('kacang-hijau.jpeg')
        elif hasil[0] == 'blackgram':
            st.image('black-gram.jpeg')
        elif hasil[0] == 'lentil':
            st.image('kacang-kacangan.jpeg')
        elif hasil[0] == 'pomegranate':
            st.image('delima.jpeg')
        elif hasil[0] == 'banana':
            st.image('pisang.jpeg')
        elif hasil[0] == 'mango':
            st.image('mangga.jpeg')
        elif hasil[0] == 'grapes':
            st.image('anggur.jpeg')
        elif hasil[0] == 'watermelon':
            st.image('semangka.jpeg')
        elif hasil[0] == 'muskmelon':
            st.image('muskmelon.jpeg')
        elif hasil[0] == 'apple':
            st.image('apel.jpeg')
        elif hasil[0] == 'orange':
            st.image('jeruk.jpeg')
        elif hasil[0] == 'papaya':
            st.image('pepaya.jpeg')
        elif hasil[0] == 'coconut':
            st.image('kelapa.jpeg')
        elif hasil[0] == 'cotton':
            st.image('kapas.jpeg')
        elif hasil[0] == 'jute':
            st.image('goni.jpeg')
        elif hasil[0] == 'coffee':
            st.image('kopi.jpeg')
