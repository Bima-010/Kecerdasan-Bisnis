import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model, encoder, dan data produk
model = joblib.load('iphone_quantity_model.pkl')
encoder = joblib.load('product_id_encoder.pkl')
product_data = joblib.load('product_data.pkl')

# Dictionary untuk lookup produk
product_dict = {f"{p['product_name']} ({p['storage']}, {p['color']})": p for p in product_data}

# Judul aplikasi
st.title("Prediksi Jumlah Pembelian iPhone")

# Deskripsi
st.write("Pilih produk dan diskon untuk melihat harga setelah diskon dan prediksi jumlah pembelian.")

# Input pengguna
st.subheader("Input Transaksi")
product_selection = st.selectbox("Pilih Produk", options=list(product_dict.keys()))
discount = st.selectbox("Pilih Diskon (%)", options=[0, 5, 10, 15])

# Detail produk
selected_product = product_dict[product_selection]
product_id = selected_product['product_id']
unit_price = selected_product['price']

# Menampilkan harga satuan
st.write(f"**Harga Satuan**: Rp {unit_price:,.0f}")

# Menghitung dan menampilkan harga setelah diskon
price_after_discount = unit_price * (1 - discount / 100)
st.write(f"**Harga Setelah Diskon**: Rp {price_after_discount:,.0f}")

# Tombol untuk prediksi
if st.button("Prediksi Jumlah Pembelian"):
    # Siapkan data input untuk prediksi
    input_data = {
        'unit_price': [unit_price],
        'discount': [discount]
    }
    # One-hot encoding product_id
    product_id_array = np.array([[product_id]])
    product_id_encoded = encoder.transform(product_id_array)
    product_id_df = pd.DataFrame(product_id_encoded, columns=[f'product_id_{int(i)}' for i in encoder.categories_[0]])
    
    # Gabungkan fitur
    input_df = pd.DataFrame(input_data)
    input_df = pd.concat([input_df, product_id_df], axis=1)
    
    # Pastikan kolom sesuai dengan model
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    
    # Prediksi
    predicted_quantity = model.predict(input_df)[0]
    st.success(f"📦 **Prediksi Jumlah Pembelian**: {predicted_quantity:.2f} unit")
    
    # Catatan tentang prediksi
    if predicted_quantity < 1 or predicted_quantity > 4:
        st.warning("Catatan: Prediksi mungkin tidak realistis karena jumlah pembelian harus antara 1-4 unit.")
