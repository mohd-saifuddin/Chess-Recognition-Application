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
    label='Please upload a chess image below.', type=['jpeg', 'png', 'jpg'])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image.save(fp='./chess_image.jpeg', format='JPEG')

        pipe = Pipeline(chess_image='./chess_image.jpeg')
        fen_label, interpretation = pipe.predict()

        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(figure_or_data=pipe.chess_image_display,
                            use_container_width=True)

        with col2:
            st.write(fen_label)
            st.write(interpretation)
            st.write(
                'Interested to know how I predict the FEN & interpret the same of a chess image?')
            st.write(
                'Please read this detailed [blog](https://medium.com/towards-data-science/chess-recognition-problem-a-deep-dive-solution-e4d8a439dc37) written by my creator [Mohammed Saifuddin](https://www.linkedin.com/in/mohammed-saifuddin-850a6b133/).')
    except:
        st.write('Please upload a valid chess image.')
        st.write(
            'I would recommend you to download the test chess images from the [dataset](https://www.kaggle.com/datasets/koryakinp/chess-positions) source.')
        st.write(
            'Please read this detailed [blog](https://medium.com/towards-data-science/chess-recognition-problem-a-deep-dive-solution-e4d8a439dc37) written by my creator [Mohammed Saifuddin](https://www.linkedin.com/in/mohammed-saifuddin-850a6b133/).')
