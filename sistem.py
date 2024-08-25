import pickle
import streamlit as st
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Proses Data
df = pd.read_csv('spotify.csv')
df = df.head(5000).drop('link', axis=1).reset_index(drop=True)
df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex=True)

stemmer = PorterStemmer()

def tokenization(txt): 
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

df['text'] = df['text'].apply(lambda x: tokenization(x))

tfidvector = TfidfVectorizer(analyzer='word', stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)

def recommendation(selected_song):
    selected_song = selected_song.strip().lower()
    idx = df[df['song'].str.strip().str.lower() == selected_song].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])

    songs = []
    for m_id in distances[1:6]:
        song_name = df.iloc[m_id[0]].song
        artist_name = df.iloc[m_id[0]].artist
        songs.append((song_name, artist_name))
    return songs

# Fungsi membaca atau membuat data dokumen feedback pengguna
try:
    user_feedback = pickle.load(open('user_feedback.pkl', 'rb'))
except (FileNotFoundError, EOFError):
    user_feedback = {}

def save_user_feedback(song_id, feedback):
    if song_id in user_feedback:
        user_feedback[song_id].append(feedback)
    else:
        user_feedback[song_id] = [feedback]

    pickle.dump(user_feedback, open('user_feedback.pkl', 'wb'))

# Streamlit
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://img.freepik.com/free-photo/flat-lay-music-background-with-acoustic-guitar_169016-21058.jpg?t=st=1724415689~exp=1724419289~hmac=2d273cea6ae4d4f4adabafcf9f74e7c47df801520e45c2ab8b813a1119ddb936&w=1060");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Sistem Rekomendasi Lagu')

# Sidebar
if  st.sidebar.checkbox('Tampilkan DataFrame'):
    st.subheader('List DataFrame:')
    st.write(df.head(20))

# Format tampilan pilihan lagu
def format_option(option):
    song = df.loc[df['song'] == option, 'song'].values[0]
    artist = df.loc[df['song'] == option, 'artist'].values[0]
    return f"{song} - {artist}"

# Input pilihan lagu
selected_song = st.selectbox('Pilih lagu:', df['song'].values, format_func=format_option)
selected_song = selected_song.strip().lower()

# Tombol tampilkan rekomendasi
if st.button('Tampilkan rekomendasi'):
    recommended_music_names = recommendation(selected_song)

# Fungsi menampilkan lagu yang dipilih
    selected_song_details = df[df['song'] == selected_song]
    if not selected_song_details.empty:
        st.write(f"**Artis:** {selected_song_details['artist'].values[0]}")
        st.write(f"**Judul lagu:** {selected_song}")

# Fungsi menampilkan rekomendasi lagu
    st.subheader('Rekomendasi Lagu:')
    for i, (song_name, artist_name) in enumerate(recommended_music_names):
        c = st.container(border=True)
        c.write(f"{i+1}. **Artis:** {artist_name}, **Judul lagu:** {song_name}")

# Text area feedback
user_feedback_text = st.text_area('Berikan feedback (Opsional):')

# Tombol kirim feedback
if st.button('Kirim feedback'):
    # Fungsi simpan feedback
    save_user_feedback(selected_song, user_feedback_text)

    # Fungsi menampilkan feedback
    st.write(f"**Feedback anda:** {user_feedback_text}")

# Tombol tampilkan riwayat feedback
if st.button('Tampilkan riwayat feedback'):
    # Fungsi menampilkan riwayat feedback
    if selected_song in user_feedback:
        st.subheader(f'Feedback untuk {selected_song}:')
        for feedback in user_feedback[selected_song]:
            c = st.container(border=True)
            c.write(feedback)
    else:
        st.write(f'Belum ada feedback untuk {selected_song}')
