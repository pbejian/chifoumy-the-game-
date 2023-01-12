#===============================================================================

import random
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle
import mediapipe as mp

#===============================================================================

IMAGE_PATH = "../images/"

#image_path = IMAGE_PATH + "logo_rock.png"
#chifoumi_image = Image.open(image_path)
#picture = st.image(chifoumi_image, width=600)
#picture = st.image(chifoumi_image, width=200)

#-------------------------------------------------------------------------------

html_title = "<h2 style='color:#FF036A'>Cette application vous est proposée par la Chifouteam</h2>"
st.markdown(html_title, unsafe_allow_html=True)

#-------------------------------------------------------------------------------

def espace(n):
    """
    Cette fonction ne renvoie rien mais affiche n lignes vides
    dans une application streamlit.
    """
    for _ in range(n):
        st.write("")
    return None

#-------------------------------------------------------------------------------




#-------------------------------------------------------------------------------
