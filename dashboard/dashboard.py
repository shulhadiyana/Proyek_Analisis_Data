from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr, ttest_ind

st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="deep")

BASE_DIR = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = Path(__file__).resolve().parent
CLEAN_DATA_PATH = DASHBOARD_DIR / "all_data.csv"
RAW_DATA_PATH = BASE_DIR / "data" / "PRSA_Data_Aotizhongxin_20130301-20170228.csv"

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}

@st.cache_data
def load_data():
    data_path = CLEAN_DATA_PATH if CLEAN_DATA_PATH.exists() else RAW_DATA_PATH
    df = pd.read_csv(data_path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df["season"] = df["month"].map(SEASON_MAP)
    return df

def filter_data(df, year_range, seasons):
    filtered = df[df["year"].between(year_range[0], year_range[1])].copy()
    if seasons:
        filtered = filtered[filtered["season"].isin(seasons)]
    return filtered

def monthly_pm25(df):
    return df.groupby("month")["PM2.5"].mean().round(2)

def o3_temp_corr(df, target_year=2015):
    summer = df[(df["year"] == target_year) & (df["month"].isin([6,7,8]))]
    if len(summer) > 1:
        corr, p = pearsonr(summer["TEMP"], summer["O3"])
        return corr, p
    return None, None

def pm10_wind_comparison(df, target_year=2016):
    year_data = df[df["year"] == target_year]
    low = year_data[year_data["WSPM"] < 1.0]["PM10"]
    high = year_data[year_data["WSPM"] > 5.0]["PM10"]
    if len(low) > 0 and len(high) > 0:
        t_stat, p_val = ttest_ind(low, high, equal_var=False)
        return low.mean(), high.mean(), p_val
    return None, None, None

def wind_direction_pm25(df):
    wind = df.groupby("wd").agg(avg_pm25=("PM2.5", "mean"), count=("wd", "size")).reset_index()
    return wind[wind["count"] >= 500].sort_values("avg_pm25", ascending=False)

def seasonal_pm25(df):
    return df.groupby("season")["PM2.5"].mean().reindex(["Spring", "Summer", "Autumn", "Winter"])

df = load_data()

min_year = int(df["year"].min())
max_year = int(df["year"].max())
season_options = ["Spring", "Summer", "Autumn", "Winter"]

with st.sidebar:
    st.title("Filter Dashboard")
    selected_years = st.slider(
        "Rentang tahun",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
    )
    selected_seasons = st.multiselect(
        "Pilih musim",
        options=season_options,
        default=season_options,
    )
    st.caption("Data tahun 2017 hanya sampai Februari.")

main_df = filter_data(df, selected_years, selected_seasons)

if main_df.empty:
    st.warning("Filter tidak menghasilkan data. Ubah rentang tahun atau musim.")
    st.stop()

st.title("Dashboard Kualitas Udara - Stasiun Aotizhongxin")
st.caption("Analisis tren PM2.5, korelasi suhu-O3, dan pengaruh kecepatan angin terhadap PM10.")

col1, col2, col3 = st.columns(3)
col1.metric("Rata-rata PM2.5", f"{main_df['PM2.5'].mean():.2f} µg/m³")
col2.metric("Rata-rata PM10", f"{main_df['PM10'].mean():.2f} µg/m³")
col3.metric("Rata-rata O3", f"{main_df['O3'].mean():.2f} µg/m³")

st.markdown("### Pertanyaan 1: Tren PM2.5 Bulanan")
monthly_data = monthly_pm25(main_df)
fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(monthly_data.index, monthly_data.values, color="skyblue")
ax.set_xlabel("Bulan")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Rata-rata PM2.5 per Bulan")
ax.set_xticks(range(1,13))
st.pyplot(fig)

peak_month = monthly_data.idxmax()
peak_value = monthly_data.max()
st.info(f"Rata-rata PM2.5 tertinggi terjadi pada bulan **{peak_month}** dengan nilai **{peak_value:.2f} µg/m³**.")

st.markdown("### Pertanyaan 2: Korelasi Suhu dan O3 (Musim Panas 2015)")
corr, p = o3_temp_corr(main_df, target_year=2015)
if corr is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    summer_data = main_df[(main_df["year"] == 2015) & (main_df["month"].isin([6,7,8]))]
    sns.scatterplot(x="TEMP", y="O3", data=summer_data, alpha=0.6, ax=ax)
    ax.set_xlabel("Suhu (°C)")
    ax.set_ylabel("O3 (µg/m³)")
    ax.set_title(f"Korelasi Suhu - O3, r = {corr:.3f} (p = {p:.5f})")
    st.pyplot(fig)
    if p < 0.05:
        st.success(f"Korelasi positif signifikan (r = {corr:.3f}, p < 0.05).")
    else:
        st.warning("Korelasi tidak signifikan secara statistik.")
else:
    st.info("Data musim panas 2015 tidak tersedia pada filter yang dipilih.")

st.markdown("### Pertanyaan 3: PM10 pada Angin Rendah vs Tinggi (2016)")
mean_low, mean_high, p_val = pm10_wind_comparison(main_df, target_year=2016)
if mean_low is not None:
    fig, ax = plt.subplots(figsize=(8, 5))
    data_2016 = main_df[main_df["year"] == 2016]
    low_wind = data_2016[data_2016["WSPM"] < 1.0]["PM10"]
    high_wind = data_2016[data_2016["WSPM"] > 5.0]["PM10"]
    bp = ax.boxplot([low_wind, high_wind], labels=["Angin rendah (<1 m/s)", "Angin tinggi (>5 m/s)"], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#90CAF9")
    ax.set_ylabel("PM10 (µg/m³)")
    ax.set_title(f"Perbandingan PM10 - Tahun 2016")
    st.pyplot(fig)
    st.write(f"Rata-rata PM10 angin rendah: **{mean_low:.2f}** µg/m³")
    st.write(f"Rata-rata PM10 angin tinggi: **{mean_high:.2f}** µg/m³")
    if p_val < 0.05:
        st.success(f"Perbedaan signifikan (p = {p_val:.5f}).")
    else:
        st.warning("Perbedaan tidak signifikan.")
else:
    st.info("Data untuk tahun 2016 tidak cukup atau tidak tersedia.")

st.markdown("### Analisis Tambahan: Arah Angin dan Musim")
col_left, col_right = st.columns(2)

with col_left:
    wind_df = wind_direction_pm25(main_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="avg_pm25", y="wd", data=wind_df.head(10), palette="Reds_r", ax=ax)
    ax.set_xlabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_ylabel("Arah Angin")
    ax.set_title("10 Arah Angin dengan PM2.5 Tertinggi")
    st.pyplot(fig)

with col_right:
    seasonal = seasonal_pm25(main_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=seasonal.index, y=seasonal.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Musim")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_title("Rata-rata PM2.5 per Musim")
    st.pyplot(fig)

st.caption("Data source: PRSA Aotizhongxin station (2013-2017) | Dashboard untuk Proyek Analisis Data")