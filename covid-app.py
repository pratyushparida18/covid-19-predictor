import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
from numpy import asarray


model = load_model("model.h5")

def main():
    uploaded_file = None
    st.title('Covid Detector')
    if st.checkbox('JPG'):
        uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if st.checkbox('JPEG'):
        uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

    if uploaded_file is not None:
        result = prediction(uploaded_file)
        if result == 0:
            st.error("Covid-19 Detected")
        else:
            st.success("You are Safe")



def prediction(uploaded_file):
    img = Image.open(uploaded_file)
    img=img.resize((224,224))
    numpydata = asarray(img)
    numpydata = numpydata.reshape(-1, 224, 224, 3)
    sc_img=numpydata/255
    predict=model.predict_classes(sc_img)[0][0]
    return predict

if __name__ == '__main__':
    main()