```markdown
# Proyek Analisis Data Kualitas Udara - Stasiun Aotizhongxin

## Setup Environment - Anaconda
```
conda create --name air-quality-ds python=3.9
conda activate air-quality-ds
pip install -r requirements.txt
```

## Setup Environment - Shell/Terminal
```
mkdir proyek_analisis_data
cd proyek_analisis_data
pipenv install
pipenv shell
pip install -r requirements.txt
```

## Run Streamlit App
```
streamlit run dashboard/dashboard.py
```

## Data Source
Dataset: PRSA_Data_Aotizhongxin_20130301-20170228.csv (Publicly available air quality data from Beijing)