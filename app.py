from dlpipeline import Pipeline
from PIL import Image

import os
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


st.set_page_config(layout='wide')

st.markdown(
    body="<h3 style='text-align: center;'>Chess Recognition</h3>",
    unsafe_allow_html=True
)

if os.path.isfile(path='chess_image.jpeg'):
    os.remove(path='./chess_image.jpeg')

uploaded_file = st.file_uploader(
    label='Choose a chess image below.', type=['jpeg'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save(fp='./chess_image.jpeg', format='JPEG')

    pipe = Pipeline(chess_image='./chess_image.jpeg')
    fen_label, interpretation = pipe.predict()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(figure_or_data=pipe.chess_image_display,
                        use_container_width=True)

    with col2:
        st.write(fen_label)
        st.write(interpretation)
