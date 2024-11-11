#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## Traitement d'images pour l'astrophotographie (CANON A7S)

**Objectif:**
* Aider à la focalisation en temps réel d'un appareil photo CANON A7S.
* Analyser les images capturées à partir du flux vidéo du logiciel LiveView.
* Fournir des outils de visualisation et d'analyse pour optimiser la mise au point.

**Fonctionnalités principales:**
* **Capture d'écran:** Acquisition régulière d'images à partir de la zone de LiveView spécifiée à l'aide de screenshot de l'écran du PC.
* **Traitement d'images:**
    * Recherche et suivi de la région d'intérêt (ROI).
    * Conversion en niveaux de gris.
    * Accumulation d'images pour améliorer le signal (activable).
    * Binarisation pour isoler les zones de fort contraste.
    * Calcul du barycentre et centrage sur la ROI.
    * Calcul de la transformée de Fourier et de la Full Width at Half Maximum (FWHM).
* **Visualisation:**
    * Affichage en temps réel des images traitées et des résultats.
    * Graphiques de l'évolution de la FWHM et du nombre de pixels blancs.
    * Spectre de Fourier pour l'analyse fréquentielle.
* **Interface utilisateur:**
    * Interface graphique intuitive pour contrôler les paramètres et visualiser les résultats.
    * Possibilité de zoomer sur l'image.
    * Mode spécifique pour l'analyse des masques de Bahtinov.

**Limitations:**
* Nécessite l'installation du logiciel LiveView de Canon.
* La précision des mesures dépend de la qualité du flux vidéo et des conditions d'observation.

**Auteur:** Thomas GAUTIER
**Date:** 08/11/2024
**Licence:** MIT
**Version:** V01
**Historique des modifications:** 
    * V01 - 08/11/2024 - Code initial

**Contributions:** 

Ce code est distribué sous la licence MIT.
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pyautogui
import Fonction.traitement_image as traitement_image  # Importer le module contenant les fonctions de traitement
import Fonction.Fenetres as Fenetres
import Fonction.config as config
import Fonction.action_menu as action_menu

def main():
    # Création de la fenêtre Tkinter
    config.root = tk.Tk()
    config.root.title("Capture d'écran et traitement")
    config.root.geometry("400x900+1500+40")  
    # Pour maintenir la fenêtre au premier plan :
    config.root.attributes('-topmost', True)
    config.root.bind("<KeyPress>", on_key_press)

    # Création d'une Frame principale
    main_frame = tk.Frame(config.root)
    main_frame.pack(fill="both", expand=True)

    # Création des labels et ajout à la Frame
    label_original = tk.Label(main_frame)
    label_original.pack(side="top", fill="both", expand=True)

    label_traite = tk.Label(main_frame)
    label_traite.pack(fill="both", expand=True)
    label_traite.bind("<Button-1>", on_click)

    label_graph = tk.Label(main_frame)
    label_graph.pack(side="bottom", fill="both", expand=True)

    # Création de la barre de menu
    action_menu.creation_menu_principal()

    # Mise à jour initiale de l'image
    config.index_tab=0
    config.tab_FWHM = np.zeros(400)
    config.tab_pix_bin = np.zeros(400)
    #update_image()
    update_image(label_original,label_traite,label_graph)

    # Boucle principale pour mettre à jour l'image toutes les 40ms (environ 25fps)
    while True:
        #update_image()
        update_image(label_original,label_traite,label_graph)
        config.root.update()
        #time.sleep(0.04)

    config.root.mainloop()
    
#####################################################################
## Traitement sur click
#####################################################################
def on_click(event):
    if event.y>372 :
        if event.x>10 and event.x<100 :
            if config.mode==1:
                config.mode=2
            else :
                config.mode=1

        if event.x>120 and event.x<200 :
            if config.zoom==1:
                config.zoom=2
            elif config.zoom==2:
                config.zoom=4
            elif config.zoom==4:
                config.zoom=8
            else :
                config.zoom=1

        if event.x>220 and event.x<320 :
            if config.empilage==1:
                config.empilage=10
            else :
                config.empilage=1

        if event.x>340 and event.x<380 :
            if config.start==0:
                config.start=1
                config.index_tab=0
                config.tab_FWHM = np.zeros(400)
                config.tab_pix_bin = np.zeros(400)
                config.val_FTM_max = 0
                config.val_FTM_log_min = 1000000
                config.val_BIN_min = 1000000
            else :
                config.start=0
    else:
        config.offset_x_ROI = event.x * 2
        config.offset_y_ROI = event.y * 2

#####################################################################
## Traitement sur clavier
#####################################################################
def create_snapshot_window(frame_):
    # Créer une nouvelle fenêtre

    image=traitement_image.duplication_image(frame_)

    snapshot_window = tk.Toplevel()
    snapshot_window.title("Snapshot")
    snapshot_window.geometry("800x800")

    # Créer un label pour afficher l'image dans la nouvelle fenêtre
    snapshot_label = tk.Label(snapshot_window)
    snapshot_label.pack()

    image=traitement_image.traitement_cercle(image)

    # Afficher l'image passée en argument dans la nouvelle fenêtre
    snapshot_image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    snapshot_label.configure(image=snapshot_image)
    snapshot_label.image = snapshot_image

def on_key_press(event):
    #global frame
    if event.char == ' ':  # Si la touche espace est pressée
        # Appeler la fonction pour créer la fenêtre de snapshot avec l'image actuelle
        create_snapshot_window(config.frame)
    elif event.char == '1': 
        config.mode=1
    elif event.char == '2': 
        config.mode=2

#####################################################################
## Traitement de l'image principale
#####################################################################
def update_image(label_original,label_traite,label_graph):

    # Capture d'écran
    try:
        screen_image = pyautogui.screenshot()
        config.nb_image =config.nb_image+1

    except Exception:
        print("Tiens une erreur")
        return
    
    # initialisation des variables
    portion_size_ROI=400
    config.frame = cv2.cvtColor(np.array(screen_image), cv2.COLOR_RGB2BGR)
    img_spectrum = np.zeros((300, 400, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Assurer que les offsets ne sont pas négatifs
    config.offset_x = max(0, config.offset_x)
    config.offset_y = max(0, config.offset_y)

    ###### Traitement de l'image de la partie centrale
    # Recadrage de l'image à la taille de capture souhaitée
    x = config.offset_x
    y = config.offset_y
    config.frame = config.frame[y:y+config.crop_size, x:x+config.crop_size]

    # Appeler la fonction de traitement de l'image depuis le module traitement_image pour la recherche de ROI
    processed_frame, config.offset_x_ROI, config.offset_y_ROI = traitement_image.Recherche_ROI(config.frame, config.offset_x_ROI, config.offset_y_ROI,portion_size_ROI)

    # Redimensionner l'image pour l'affichage (zoom de 0.5)
    resized_frame = cv2.resize(processed_frame, (config.display_size, config.display_size), interpolation=cv2.INTER_LINEAR)

    # ajout du menu cliquable
    resized_frame=Fenetres.Menu_cliquable(resized_frame)

    # Conversion en format PIL pour afficher avec Tkinter la partie centrale (la fenêtre d'acquisition)
    traite_image = Image.fromarray(resized_frame)
    traite_image_tk = ImageTk.PhotoImage(traite_image)
    label_traite.configure(image=traite_image_tk)
    label_traite.image = traite_image_tk

    ###### Traitement des images traitées (partie haute de la fenêtre)
    if config.mode==1:  #Correspond au mode standard avec FTM et mesure de pixels allumés après binarisation
        original_image_cv, img_spectrum, img_grey=Fenetres.maj_image_mode_stardard(0)
    else:  #Correspond au mode masque de Bahtinov
        original_image_cv=Fenetres.maj_image_mode_bahtinov(0)

    # Conversion en format PIL pour afficher avec Tkinter la partie haute (la fenêtre des traitements)
    original_image_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2RGB)))
    label_original.configure(image=original_image_tk)
    label_original.image = original_image_tk

    ###### Traitement des images du graphique (partie base de la fenêtre)
    if config.mode==1:
        img_graph_bas = Fenetres.maj_image_graphique_bas(img_spectrum, img_grey)
    else:
        img_graph_bas = np.zeros((300, 400, 3), dtype=np.uint8)  # Créer une image noire 
    # Conversion en format PIL pour afficher avec Tkinter la partie basse (les graphs)
    traite_image = Image.fromarray(img_graph_bas)
    traite_image_tk = ImageTk.PhotoImage(traite_image)
    label_graph.configure(image=traite_image_tk)
    label_graph.image = traite_image_tk   

#####################################################################
## Execution du main
#####################################################################
if __name__ == "__main__":
    main()
