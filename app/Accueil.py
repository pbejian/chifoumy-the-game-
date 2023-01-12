#===============================================================================

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import mediapipe as mp

#===============================================================================

def espace(n):
    """
    Cette fonction ne renvoie rien mais affiche n lignes vides
    dans une application streamlit.
    """
    for _ in range(n):
        st.write("")
    return None

#-------------------------------------------------------------------------------

def picture_to_df(picture):
    """
    This function take a picture of an hand as argument (created with
    streamlit.camera_input) and return a DataFrame which contains
    the mediapipe data of this hand.
    Do not forget to scale the dataframe after that
    """
    hand_list = []
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        img = Image.open(picture)
        img_array = np.array(img)
        # image = cv2.flip(img_array, 1)
        image = img_array
        results = hands.process(image)
        if not results.multi_hand_landmarks:
            return "No hand in this picture!"
        for hand_landmarks in results.multi_hand_landmarks:
            fingers = {}
            for i, finger in enumerate(hand_landmarks.landmark, start=1):
                fingers[f'{i}x'] = (finger.x)
                fingers[f'{i}y'] = (finger.y)
                fingers[f'{i}z'] = (finger.z)
            hand_list.append(fingers)
        return pd.DataFrame(hand_list)

#===============================================================================

html_title = "<h2 style='color:#FF036A'>Chifoumy : pierre, feuille, ciseaux, python et Spock !</h2>"
st.markdown(html_title, unsafe_allow_html=True)

st.markdown("""
Cette version étendue a été popularisée par la série « The Big Bang Theory ». Voici
[une vidéo de Sheldon Cooper](https://youtu.be/_PUEoDYpUyQ)
expliquant les règles du jeux à cinq positions. Nous avons remplacé le lézard par un python... car nous programmons en
[Python !](https://www.python.org).
""", unsafe_allow_html=True)


html_subtitle = "<h3 style='color:#44B7E3'>Testons la reconnaissance des cinq gestes.</h3>"
st.markdown(html_subtitle, unsafe_allow_html=True)

html_subtitle = "<p style='color:#000000'>NB - Pour une meilleure reconnaissance, approchez votre main de la camera .</p>"
st.markdown(html_subtitle, unsafe_allow_html=True)

picture = None
picture = st.camera_input(label=" ", disabled=False, key=666)
if picture:
    button1 = st.button("Tester la photo", key=1234)
    if button1:
        df = picture_to_df(picture)
        # st.write(type(df))
        if type(df) == type("toto"):
            st.write("Problème dans l'acquisition photo.")
        else:
            # Loading the trained scaler and the trained model
            my_scaler_spock = pickle.load(open("models/scaler_spock.pkl", "rb"))
            my_model_spock = pickle.load(open("models/model_spock.pkl", "rb"))

            # Scaling the new dataframe
            X_new = df
            X_new_scaled = my_scaler_spock.transform(X_new)

            # Prediction with the new data
            target = my_model_spock.predict(X_new_scaled)
            target = target[0]
            html_pierre ="<h3 style='color:#44B7E3'>Votre geste : pierre</h3>"
            html_feuille ="<h3 style='color:#44B7E3'>Votre geste : feuille</h3>"
            html_ciseaux ="<h3 style='color:#44B7E3'>Votre geste : ciseaux</h3>"
            html_python ="<h3 style='color:#44B7E3'>Votre geste : python</h3>"
            html_spock ="<h3 style='color:#44B7E3'>Votre geste : Spock</h3>"
            chifoudict = {0: html_pierre, 1: html_feuille, 2: html_ciseaux,
                          3: html_python, 4: html_spock}
            html_gesture = chifoudict[target]
            st.markdown(html_gesture, unsafe_allow_html=True)

#-------------------------------------------------------------------------------
# Conclusion avec le lien vers les sources sur GitHub

espace(2)
st.markdown("""
    <hr>
""", unsafe_allow_html=True)
espace(2)
st.write("""
📝 Sources de l'application :
[https://github.com/pbejian/chifoumy-plus/](https://github.com/pbejian/chifoumy-plus/)
""")
#-------------------------------------------------------------------------------
