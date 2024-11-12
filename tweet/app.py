import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# Veriyi yükleme ve sütun isimlerini güncelleme
df = pd.read_csv('train.csv')

# Bağımlı ve bağımsız değişkenlerin seçimi
x = df.drop(['essay_id', 'text'], axis=1)
y = df[['text']]

# Eğitim ve test verilerini ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Ön işleme (StandardScaler ve OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['feeling']),
        ('cat', OneHotEncoder(), ['text'])
    ]
)

# Streamlit uygulaması
def rings_pred(feeling, text):
    input_data = pd.DataFrame({
        'text': [text],
        'feeling': [feeling]
        
    })
    
    
    input_data_transformed = preprocessor.fit_transform(input_data)

    model = joblib.load('Tweet.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

st.title("Abalone Veri seti ile Yaş Tahmini Regresyon Modeli")
st.write("Veri Gir")

text = st.selectbox('text', df['text'].unique())
feeling = st.selectbox('feeling', df['feeling'].unique())

    
if st.button('Predict'):
    rings = rings_pred(text,feeling)
    st.write(f'The predicted rings is: {rings:.2f}')