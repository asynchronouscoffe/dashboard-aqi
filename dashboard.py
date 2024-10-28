import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.cluster import KMeans

# Load dataset
@st.cache_data
def muat_data():
    return pd.read_csv('all_df.csv')

data = muat_data()

# Judul dan deskripsi
st.title('Dashboard Analisis Kualitas Udara')
st.markdown('Dashboard ini menampilkan hasil analisis data polutan yang terdapat pada Air Quality Dataset')

# Sidebar
st.sidebar.title('Navigasi')
options = st.sidebar.radio('Pilih Kolom Analisis', 
                            ['Distribusi Polutan', 'Perbandingan Polutan dengan Temperatur', 
                            'Analisis Korelasi', 'Visualisasi Kluster'])

# Visualisasi Distribusi Polutan
if options == 'Distribusi Polutan':
    st.subheader('Rata-rata Konsentrasi Polutan')

    polutan_mean = data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].mean()
    plt.figure(figsize=(7, 6))
    plt.pie(polutan_mean, labels=polutan_mean.index, autopct='%1.1f%%', startangle=90)
    plt.title('Rata-rata Konsentrasi Polutan')
    st.pyplot(plt)

# Visualisasi Perbandingan PM2.5 dan PM10 dengan Temperatur
elif options == 'Perbandingan Polutan dengan Temperatur':
    st.subheader('Perbandingan PM2.5 & PM10 dengan Temperatur')

    data['datetime'] =pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

    pm25_month_avg = data['PM2.5'].resample('M').mean()
    pm10_month_avg = data['PM10'].resample('M').mean()
    temp_month_avg = data['TEMP'].resample('M').mean()

    plt.figure(figsize=(12, 6))
    plt.plot(pm25_month_avg.index, pm25_month_avg, color='red', label='PM2.5 (µg/m³)')
    plt.plot(pm10_month_avg.index, pm10_month_avg, color='orange', label='PM10 (µg/m³)')
    plt.plot(temp_month_avg.index, temp_month_avg, color='blue', label='Temperatur (°C)')
    plt.title('Perbandingan Tingkat Konsentrasi PM2.5 & PM10 dengan Temperatur')
    plt.legend()
    st.pyplot(plt)

# Visualisasi Analisis Korelasi
elif options == 'Analisis Korelasi':
    st.subheader('Matrix Korelasi Polutan')

    matrix_korelasi = data[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix_korelasi, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matrix Korelasi Polutan')
    st.pyplot(plt)

# Visualisasi Kluster Polutan
elif options == 'Visualisasi Kluster':
    st.subheader('Klusterisasi Polutan')

    if 'Cluster' not in data.columns:
        X = data[['PM2.5', 'PM10']]
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        data.loc[X.index, 'Cluster'] = clusters

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='PM2.5', y='PM10', hue='Cluster', palette='viridis', s=100)

    for i in range(data.shape[0]):
        plt.annotate(data['station'][i],
                 (data['PM2.5'][i], data['PM10'][i]),
                 textcoords="offset points",
                 xytext=(5,-5),
                 ha='left', fontsize=9, alpha=0.7,
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    plt.title('Klusterisasi Polutan (PM2.5 dan PM10)')
    st.pyplot(plt)