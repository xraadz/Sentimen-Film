import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# ========================
# Stopwords & Normalisasi
# ========================
STOPWORDS_ID = {
    'yang', 'dan', 'di', 'ke', 'dari', 'ini', 'itu', 'untuk', 'dengan',
    'saya', 'kamu', 'dia', 'adalah', 'karena', 'pada', 'dalam', 'atau',
    'tapi', 'jadi', 'tidak', 'ya', 'bisa', 'akan', 'jika', 'seperti',
    'film', 'nonton', 'aja'
}

kamus_normalisasi = {
    'bgt': 'banget', 'bgs': 'bagus', 'gk': 'tidak', 'ga': 'tidak', 'gak': 'tidak',
    'tdk': 'tidak', 'nggak': 'tidak', 'bgttt': 'banget', 'jelekkk': 'jelek',
    'nyesel': 'menyesal', 'parahh': 'parah', 'kzl': 'kesal', 'gg': 'tidak'
}

def normalisasi_kata(teks):
    kata2 = teks.split()
    return ' '.join([kamus_normalisasi.get(k, k) for k in kata2])

def hapus_stopword(teks):
    return ' '.join([k for k in teks.split() if k not in STOPWORDS_ID])

def bersihkan_teks(teks):
    teks = teks.lower()
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    teks = normalisasi_kata(teks)
    teks = hapus_stopword(teks)
    return teks

def deteksi_kata_kasar(teks):
    kasar = [
        'anjing', 'goblok', 'bangsat', 'tai', 'kontol', 'tolol', 'jelek',
        'gajelas', 'sampah', 'nyesel', 'busuk', 'parah', 'bodoh', 'brengsek', 'anyink', 'anyinkkk', 'titit', 'titid', 'asu'
    ]
    return any(kata in teks.lower() for kata in kasar)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("Opini_FilmX_Dataset_Bersih.csv")
        df = df[['text_tweet', 'sentiment']].dropna()
        df['text_tweet'] = df['text_tweet'].astype(str).apply(bersihkan_teks)
        return df
    except:
        st.error("❌ File 'Opini_FilmX_Dataset_Bersih.csv' tidak ditemukan.")
        return None

@st.cache_resource
def train_model(df):
    X = df['text_tweet']
    y = df['sentiment']
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2))),
        ('clf', LogisticRegression(class_weight='balanced'))
    ])
    pipe.fit(X, y)
    return pipe

# ========================
# START APP
# ========================
st.set_page_config(page_title="Sentimen Film", layout="wide")

# Sidebar navigasi
st.sidebar.title("🔍 Navigasi")
page = st.sidebar.selectbox("📂 Pilih Halaman:", ["🎯 Prediksi", "📊 Dashboard"])

data = load_data()
model = train_model(data) if data is not None else None

# ========================
# Halaman: Prediksi
# ========================
if page == "🎯 Prediksi":
    st.title("🎬 Prediksi Sentimen Komentar Film")
    st.write("Masukkan komentar kamu, lalu sistem akan memprediksi apakah komentarnya positif atau negatif.")

    st.subheader("💬 Tulis komentar kamu:")
    user_input = st.text_area("Contoh: Film ini sangat bagus!")

    if st.button("🔮 Prediksi Sentimen"):
        if user_input.strip() == "":
            st.warning("Komentar tidak boleh kosong.")
        else:
            clean_input = bersihkan_teks(user_input)

            if deteksi_kata_kasar(clean_input):
                prediction = "negative"
                reason = "Terdeteksi kata kasar"
            else:
                prediction = model.predict([clean_input])[0]
                reason = "Komentar anda baik"

            if prediction == "positive":
                st.success(f"✅ Sentimen: **POSITIF**\n📌 ({reason})")
            else:
                st.error(f"⚠️ Sentimen: **NEGATIF**\n📌 ({reason})")

# ========================
# Halaman: Dashboard
# ========================
elif page == "📊 Dashboard":
    st.title("📊 Dashboard Dataset Komentar")
    if data is not None:
        st.subheader("🔢 Jumlah Sentimen")

        sentimen_count = data['sentiment'].value_counts().reset_index()
        sentimen_count.columns = ['sentiment', 'count']
        st.dataframe(sentimen_count)

        st.subheader("📊 Visualisasi Pie Chart:")
        fig1, ax1 = plt.subplots()
        ax1.pie(sentimen_count['count'], labels=sentimen_count['sentiment'], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("📊 Visualisasi Bar Chart:")
        fig2, ax2 = plt.subplots()
        ax2.bar(sentimen_count['sentiment'], sentimen_count['count'], color=['green', 'red'])
        ax2.set_xlabel("Sentimen")
        ax2.set_ylabel("Jumlah")
        ax2.set_title("Distribusi Sentimen")
        st.pyplot(fig2)

        # ========================
        # Komentar Positif
        # ========================
        st.subheader("🟢 Komentar Positif")
        data_positive = data[data['sentiment'] == 'positive'].reset_index(drop=True)

        pos_total = len(data_positive)
        pos_page = st.number_input("Halaman Positif:", min_value=1, max_value=(pos_total - 1) // 5 + 1, value=1, step=1)
        pos_start = (pos_page - 1) * 5
        pos_end = pos_start + 5
        st.dataframe(data_positive.iloc[pos_start:pos_end])
        st.caption(f"Menampilkan {pos_start+1}-{min(pos_end, pos_total)} dari {pos_total} komentar positif.")

        # ========================
        # Komentar Negatif
        # ========================
        st.subheader("🔴 Komentar Negatif")
        data_negative = data[data['sentiment'] == 'negative'].reset_index(drop=True)

        neg_total = len(data_negative)
        neg_page = st.number_input("Halaman Negatif:", min_value=1, max_value=(neg_total - 1) // 5 + 1, value=1, step=1)
        neg_start = (neg_page - 1) * 5
        neg_end = neg_start + 5
        st.dataframe(data_negative.iloc[neg_start:neg_end])
        st.caption(f"Menampilkan {neg_start+1}-{min(neg_end, neg_total)} dari {neg_total} komentar negatif.")

    else:
        st.error("Dataset tidak tersedia.")
