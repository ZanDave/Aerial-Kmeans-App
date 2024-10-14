# 🚀 Clustering Gambar Udara Terintegrasi

Ini adalah aplikasi Streamlit untuk proyek Ujian Tengah Semester (UTS) mata kuliah Data Mining. Aplikasi ini memungkinkan pengguna untuk melatih atau memuat model machine learning yang dapat melakukan clustering gambar udara khususnya pantai. Dengan teknik segmentasi dan clustering seperti Felzenszwalb dan MiniBatchKMeans, aplikasi ini menyediakan visualisasi gambar hasil clustering beserta metrik skor siluet.

[![Buka di Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aerial-uts-datmin.streamlit.app/)

## Fitur
- **Pelatihan Model**: Latih model clustering baru menggunakan dataset gambar udara (aerial)
- **Memuat Model Pre-trained**: Muat model yang sudah dilatih sebelumnya untuk proses clustering yang lebih cepat.
- **Clustering Gambar**: Unggah gambar untuk dilakukan segmentasi menjadi beberapa cluster menggunakan Felzenszwalb dan MiniBatchKMeans.
- **Visualisasi**: Lihat gambar asli dan gambar hasil clustering dengan label cluster dan skor siluet secara berdampingan.
- **Metrik**: Tampilkan metrik clustering seperti jumlah cluster dan skor siluet.