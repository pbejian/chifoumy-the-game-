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

#image_path = IMAGE_PATH  +  "logo_rock.png"
#chifoumi_image = Image.open(image_path)
#picture = st.image(chifoumi_image, width=600)
#picture = st.image(chifoumi_image, width=200)

#-------------------------------------------------------------------------------

MAX_SCORE = 3

def scoring(machine_gesture, user_gesture):
    """
    0: pierre
    1: feuille
    2: ciseaux
    """
    if user_gesture==machine_gesture:
        return "null"
    elif user_gesture==0 and machine_gesture==2:
        return "user"
    elif user_gesture == 1 and machine_gesture == 0:
        return "user"
    elif user_gesture == 2 and machine_gesture == 1:
        return "user"
    else:
        return "machine"

#-------------------------------------------------------------------------------

def scoring_spock(machine_gesture, user_gesture):
    """
    0: pierre,
    1: feuille,
    2: ciseaux
    3: python
    4: spock
    """
    if user_gesture==machine_gesture:
        return "null"
    elif user_gesture==0 and machine_gesture==2:
        return "user"
    elif user_gesture==0 and machine_gesture==3:
        return "user"
    elif user_gesture==0 and machine_gesture==1:
        return "machine"
    elif user_gesture==0 and machine_gesture==4:
        return "machine"
    elif user_gesture==1 and machine_gesture==0:
        return "user"
    elif user_gesture==1 and machine_gesture==4:
        return "user"
    elif user_gesture==1 and machine_gesture==2:
        return "machine"
    elif user_gesture==1 and machine_gesture==3:
        return "machine"
    elif user_gesture==2 and machine_gesture==1:
        return "user"
    elif user_gesture==2 and machine_gesture==3:
        return "user"
    elif user_gesture==2 and machine_gesture==4:
        return "machine"
    elif user_gesture==2 and machine_gesture==0:
        return "machine"
    elif user_gesture==3 and machine_gesture==1:
        return "user"
    elif user_gesture==3 and machine_gesture==4:
        return "user"
    elif user_gesture==3 and machine_gesture==2:
        return "machine"
    elif user_gesture==3 and machine_gesture==0:
        return "machine"
    elif user_gesture==4 and machine_gesture==0:
        return "user"
    elif user_gesture==4 and machine_gesture==2:
        return "user"
    elif user_gesture==4 and machine_gesture==1:
        return "machine"
    elif user_gesture==4 and machine_gesture==3:
        return "machine"

#-------------------------------------------------------------------------------

def description(machine_gesture, user_gesture):
    """
    0: pierre,
    1: feuille,
    2: ciseaux,
    3: python,
    4: spoke
    """
    if user_gesture == machine_gesture:
        html_description = "style='text-align:center;color:#fffa03;font-size:30px'> Match nul !"
        return html_description
    elif user_gesture == 0 and machine_gesture == 2:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'>Gagn?? ! PIERRE ??crase CISEAUX</div>"
        return html_description
    elif user_gesture == 0 and machine_gesture == 3:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! PIERRE assomme PYTHON</div>"
        return html_description
    elif user_gesture == 0 and machine_gesture == 1:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! FEUILLE recouvre PIERRE</div>"
        return html_description
    elif user_gesture == 0 and machine_gesture == 4:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! SPOKE vaporise PIERRE</div>"
        return html_description
    elif user_gesture == 1 and machine_gesture == 0:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! FEUILLE recouvre PIERRE</div>"
        return html_description
    elif user_gesture == 1 and machine_gesture == 4:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! FEUILLE r??fute SPOKE</div>"
        return html_description
    elif user_gesture == 1 and machine_gesture == 2:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! CISEAUX d??coupent FEUILLE</div>"
        return html_description
    elif user_gesture == 1 and machine_gesture == 3:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! PYTHON mange FEUILLE</div>"
        return html_description
    elif user_gesture == 2 and machine_gesture == 1:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! CISEAUX d??coupent FEUILLE</div>"
        return html_description
    elif user_gesture == 2 and machine_gesture == 3:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! CISEAUX d??capitent PYTHON</div>"
        return html_description
    elif user_gesture == 2 and machine_gesture == 4:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! SPOKE casse CISEAUX</div>"
        return html_description
    elif user_gesture == 2 and machine_gesture == 0:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! PIERRE ??crase CISEAUX</div>"
        return html_description
    elif user_gesture == 3 and machine_gesture == 1:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! PYTHON mange FEUILLE</div>"
        return html_description
    elif user_gesture == 3 and machine_gesture == 4:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! PYTHON empoisonne SPOKE</div>"
        return html_description
    elif user_gesture == 3 and machine_gesture == 2:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! CISEAUX d??capitent PYTHON</div>"
        return html_description
    elif user_gesture == 3 and machine_gesture == 0:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! PIERRE assomme PYTHON</div>"
        return html_description
    elif user_gesture == 4 and machine_gesture == 0:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! SPOKE vaporise PIERRE</div>"
        return html_description
    elif user_gesture == 4 and machine_gesture == 2:
        html_description = "style='text-align:center;color:#6aff03;font-size:30px'> Gagn?? ! SPOKE casse CISEAUX</div>"
        return html_description
    elif user_gesture == 4 and machine_gesture == 1:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! FEUILLE r??fute SPOKE</div>"
        return html_description
    elif user_gesture == 4 and machine_gesture == 3:
        html_description = "style='text-align:center;color:#ff1a03;font-size:30px'> Perdu ! PYTHON empoisonne SPOKE</div>"
        return html_description

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

html_title = "<h2 style='color:#FF036A'>Jouons contre la machine !</h2>"
st.markdown(html_title, unsafe_allow_html=True)

#-------------------------------------------------------------------------------

file = open("scores.txt", "r")
for line in file:
    tab = line.split(",")
    user_score = int(tab[0])
    machine_score = int(tab[1])
file.close()

#-------------------------------------------------------------------------------

picture = None
placeholder1 = st.empty()
picture = placeholder1.camera_input("", key=666)
if picture:
    df = picture_to_df(picture)
    # st.write(type(df))
    if type(df) == type("toto"):
        st.write("Probl??me dans l'acquisition photo.")
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
        #st.write(type(target))
        #st.write(target.shape)
        #html_user_pierre ="<div style='color:#E37B01;font-size:30px'>Votre geste : pierre</div>"
        #html_user_feuille ="<div style='color:#AEC90E;font-size:30px'>Votre geste : feuille</div>"
        #html_user_ciseaux ="<div style='color:#8B4C89;font-size:30px'>Votre geste : ciseaux</div>"
        #----
        #chemin = IMAGE_PATH + "logo_rock.png"
        #st.write(chemin)
        #st.image(chemin)
        #----
        #user_dict = {0: html_user_pierre, 1: html_user_feuille, 2: html_user_ciseaux}
        #user_gesture = user_dict[target]
        #st.markdown(user_gesture, unsafe_allow_html=True)
        #----------------
        machine_play = random.randint(0, 4)
        #html_machine_pierre ="<div style='color:#E37B01;font-size:30px'>Geste machine : pierre</div>"
        #html_machine_feuille ="<div style='color:#AEC90E;font-size:30px'>Geste machine : feuille</div>"
        #html_machine_ciseaux ="<div style='color:#8B4C89;font-size:30px'>Geste machine : ciseaux</div>"
        #machine_dict = {0: html_machine_pierre, 1: html_machine_feuille, 2: html_machine_ciseaux}
        #machine_gesture = machine_dict[machine_play]
        #st.markdown(machine_gesture, unsafe_allow_html=True)
        #----------------
        # scoring
        result = scoring_spock(machine_play, target)
        if result=="machine":
            machine_score += 1
            #st.write(f"La machine vient de gagner la manche n?? {game_counter}.")
        elif result=="user":
            user_score += 1
            #st.write(f"L'humain vient de gagner la manche n?? {game_counter}.")
        elif result=="null":
            pass
            #st.write("Manche nulle entre l'humain et la machine.")
        user_html = f"<div style='color:#44B7E3;font-size:30px'>???? Score du joueur : {user_score}</div>"
        #st.markdown(user_html, unsafe_allow_html=True)
        machine_html = f"<div style='color:#44B7E3;font-size:30px'>???? Score de la machine : {machine_score}</div>"
        #st.markdown(machine_html, unsafe_allow_html=True)
        #st.write(f"???? Score du joueur : {user_score}")
        #st.write(f"???? Score de la machine : {machine_score}")
        file = open("scores.txt", "w")
        file.write(f"{user_score},{machine_score}")
        file.close()
        #st.write(f"??? Les scores {machine_score} et {user_score} ont ??t?? sauvegard??s.")
        #-------------------------------------------------------------------
        # Affichage am??lior??
        IMAGE_PIERRE_PATH = "https://www.bejian.fr/chifoumy/images/"
        machine_image_dict = {0: "logo_rock_machine.png",1: "logo_paper_machine.png",
                            2: "logo_scissors_machine.png",3: "logo_python_machine.png",
                            4: "logo_spock_machine.png"}
        human_image_dict = {0: "logo_rock_human.png",1: "logo_paper_human.png",
                            2: "logo_scissors_human.png", 3: "logo_python_human.png",
                            4: "logo_spock_human.png"}
        image_user = IMAGE_PIERRE_PATH + human_image_dict[target]
        image_machine = IMAGE_PIERRE_PATH + machine_image_dict[machine_play]
        html_description = description(machine_play, target)
        #st.write(image_user)
        #st.write(image_machine)
        #st.image(image_machine)
        #-------
        #ai_image_path = IMAGE_PIERRE_PATH + "ai.gif"
        #ai_html = f"""
        #<div style="display:flex;justify-content:center;align-items:center;width:100%;height:100%">
        #<img src='{ai_image_path}' width='300'>
        #</div>
        #"""
        #placeholder = st.empty()
        #placeholder = st.markdown(ai_html, unsafe_allow_html=True)
        #-------
        big_html = f"""
        <div style="display:flex;justify-content:center;align-items:center;width:100%;height:100%">
        <table>
        <tr>
            <th style='text-align:center;font-size:30px'>???? Humain</th>
            <th style='text-align:center;font-size:30px'>???? Machine</th>
        </tr>
        <tr>
            <td><img src='{image_user}' width='300'></td>
            <td><img src='{image_machine}' width='300'></td>
        </tr>
        <tr>
            <td style='text-align:center;font-size:50px;color:#44B7E3'>{user_score}</td>
            <td style='text-align:center;font-size:50px;color:#44B7E3'>{machine_score} </td>
        </tr>
        <tr>
            <td colspan='2' {html_description} </td>
        </tr>
        </table>
        </div>
        <br>
        """
        st.markdown(big_html, unsafe_allow_html=True)
        #-----------------------------------------------------------------------
        if machine_score==MAX_SCORE:
            final_html = f"<div style='color:#FF036A;font-size:30px'>?????? Victoire de la machine !</div>"
            st.markdown(final_html, unsafe_allow_html=True)
            file = open("scores.txt", "w")
            file.write("0,0")
            file.close()
        if user_score==MAX_SCORE:
            final_html = f"<div style='color:#FF036A;font-size:30px'>?????? Victoire de l'humain !</div>"
            st.markdown(final_html, unsafe_allow_html=True)
            file = open("scores.txt", "w")
            file.write("0,0")
            file.close()
        #-----------------------------------------------------------------------
