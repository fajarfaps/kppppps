# Tugas Kerja Praktek (KP)
Penerapan DeepLearning Imange Clasfication Penentuan Penyakit Tuberkulosis menggunakan Alogritma CNN


## Menjalankan di Google Colab

1. Instal Streamlit:

    ```bash
    !pip install streamlit -q
    ```

2. Dapatkan Alamat IP Publik:

    ```bash
    !wget -q -O - ipv4.icanhazip.com
    ```

3. Jalankan aplikasi Streamlit dan buka ke publik dengan menggunakan `localtunnel`:

    ```bash
    !streamlit run /content/streamlit_daun_gc.py & npx localtunnel --port 8501
    ```

Dengan langkah-langkah di atas, Anda dapat menjalankan aplikasi Streamlit di Google Colab dan mengaksesnya melalui alamat yang di-generate oleh localtunnel.
