import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image



# Load model
model1 = load_model('/content/model_paru.h5')
model2 = load_model('/content/model_paru.h5')
# Isi class_labels
class_labels = ['normal', 'tubercolusis', 'pneunomia']

# Fungsi untuk halaman Home
# Fungsi untuk halaman Home
def home_page():
    st.header("Selamat Datang di Halaman Home, Mate!", divider='rainbow')
    multi = ''' Aplikasi ini akan membantu kalian untuk memprediksi atau menentukan kesehatan sayuran pada pertanian. Kalian dapat melihat hasil prediksi dengan cara memilih foto sayuran yang akan ditentukan kesehatan, persentase kecocokan, dan deskripsi kesehatannya. Aplikasi ini diharapkan dapat membantu sektor pertanian untuk menentukan kesehatan sayurannya sehingga dapat menentukan apakah sayuran itu akan layak untuk diperjual belikan atau tidak.'''
    st.markdown(multi)

    # Tambahkan CSS untuk styling
    st.markdown(
        """
        <style>
            .column-padding {
                padding: 40px;
            }
            .image-caption {
                text-align: center;
                font-style: italic;
            }
            .column-spacing {
                margin-right: 20px;
                margin-left: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    image = Image.open('/content/Vege Healt.png')
    st.image('Vege Healt.png', caption='')
    st.write("")
    st.markdown("---")
    image = Image.open('/content/Vege Tutor.png')
    st.image('Vege Tutor.png', caption='')
    st.write("")

    col1, col2, col3 = st.columns(3)

    # Gambar yang vocab
    col1.write(" ")
    with col1:
        st.write("Vocab Image")
        image = Image.open('/content/vocab.jpg')
        st.image(image, caption="", use_column_width=True)
        st.markdown("<p class='image-caption'>Gambar di atas merupakan representasi dari split data model machine learning yang di mana pembagian pengujian yaitu 85:15. 85 persen untuk Training Samples dan 15 persen untuk Test Samples.</p>", unsafe_allow_html=True)

    # Gambar yang vocab
    col2.write(" ")
    with col2:
        st.write("Confusion Image")
        image = Image.open('/content/confus.jpg')
        st.image(image, caption='', use_column_width=True)
        st.markdown("<p class='image-caption'>Gambar yang berisikan tentang Skor Kepercayaan pada class dan prediksinya.</p>", unsafe_allow_html=True)

    # Gambar yang vocab
    col3.write(" ")
    with col3:
        st.write("Accuracy Class Image")
        image = Image.open('/content/acc_class.jpg')
        st.image(image, caption='', use_column_width=True)
        st.markdown("<p class='image-caption'>Merepresentasikan class healthy yang mendefinisikan daun yang sehat memiliki akurasi 96 persen dengan 1530 sample, class early_blight yang merepresentasikan daun yang terkena hama sehingga menimbulkan bintik-bintik pada daunnya. class tersebut memiliki akurasi 98 persen dari 2450 sample, dan yang terakhir adalah class late_blight yang merepresentasikan daun yang memiliki spot kering. Class tersebut memiliki akurasi sebesar 96 persen dari 2140 sample.</p>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    
    col1.write(" ")
    with col1:
        st.write("Accuracy per EPOCH")
        image = Image.open('/content/image_daun/acc_epoch.jpg')
        st.image(image, caption='', use_column_width=True)
        st.markdown("<p class='image-caption'>Pada table tersebut terdapat keterangan berupa tingkatan akurasinya yang memiliki nilai hampir 1 atau 100 persen.</p>", unsafe_allow_html=True)

    col3.write(" ")
    with col3:
        st.write("Loss per EPOCH")
        image = Image.open('/content/image_daun/loss_epoch.jpg')
        st.image(image, caption='', use_column_width=True)
        st.markdown("<p class='image-caption'>Pada table tersebut terdapat keterangan berupa tingkatan Loss akurasi yang memiliki nilai hampir 1 atau 100 persen.</p>", unsafe_allow_html=True)

# Fungsi untuk halaman Vegetable Health Predictions
def predictions_page():
    st.title(" ")
    uploaded_image = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    st.title(" ")

    if uploaded_image is not None:
        # Proses gambar yang diunggah
        img = image.load_img(uploaded_image, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Lakukan prediksi dengan model-model
        prediction1 = model1.predict(img_array)
        prediction2 = model2.predict(img_array)
        ensemble_prediction = (prediction1 + prediction2) / 2  # Ambil rata-rata prediksi dari model-model

        class_index = np.argmax(ensemble_prediction[0])
        class_label = class_labels[class_index]
        confidence_score = ensemble_prediction[0][class_index]

        # Tampilkan hasil prediksi
        st.image(uploaded_image, caption='Gambar yang diunggah', use_column_width=True)
        st.write(f'Kelas yang Terdeteksi : {class_label}')
        st.write(f'Skor Kepercayaan: {confidence_score * 100:.2f}%')
        

    elif class_labels == 'normal':
        st.write('LU KAGA KENA PENYAKIT')
        st.write('Tetap konsisten ya dalam menjaga sayurannya, sayuran kamu sehat tuh')
    elif class_labels == '  tubercolusis':
        st.write('LU KENA TB')
    elif class_labels == 'pneunomia':
        st.write('LU PNEUNOMIA')
    elif class_labels == '-':
        st.write('Objek tidak diketahui')

# Fungsi untuk halaman About Us
def about_us_page():
    st.title("Halaman About Us")
    # Tambahkan konten atau informasi tentang tim pengembang atau Anda di sini
# Fungsi untuk footer
# Fungsi untuk footer
def footer():
    st.title(" ")
    st.markdown("---")
    st.write("© 2023 VEGE HEALTH. All rights reserved.")
    st.write("Author: Fajar Pangestu Amandaru")
    st.write("Contact us at: fajar.faps@gmail.com")

    # Tambahkan CSS untuk styling
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                bottom: 0;
                left: 0;
                width: 100%;
                background-color: #f1f1f1;
                text-align: center;
                padding: 10px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Terapkan class CSS pada elemen footer
    st.markdown("<div class='footer'>", unsafe_allow_html=True)
    st.title(" ")
    st.markdown("</div>", unsafe_allow_html=True)

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="VEGE HEALTH")
st.sidebar.title("Welcome Mate!")
options = ["Home", "Vegetable Health Predictions", "About Us"]
selection = st.sidebar.selectbox("Yuk Explore !", options)
# Tampilkan konten sesuai dengan pilihan
if selection == "Home":
    home_page()
elif selection == "Vegetable Health Predictions":
    predictions_page()
else:
    about_us_page()

footer()