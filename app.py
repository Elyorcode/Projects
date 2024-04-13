import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px

#title
st.title('Transportni klassifikatsiya qiluvchi model')

#rasmni joylash
file = st.file_uploader('Rasm Yuklash', type=['png', 'jpeg', 'gif', 'svg'])

# Agar fayl yuklangan bo'lsa, uni ko'rsatish
if file is not None:
    # Faylning URL'ini olish
    file_url = st.image(file)
    
    # PIL convert
    img = PILImage.create(file)

    #model
    model = load_learner('transport_model.pkl')

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}')
    #plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)