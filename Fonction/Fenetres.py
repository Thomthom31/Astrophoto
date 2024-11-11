"""
## Module : interfaces

**Description:**
Ce module gère la création et la mise à jour des interfaces graphiques utilisées pour l'application d'astrophotographie. 
Il permet à l'utilisateur d'interagir avec le programme en ajustant les paramètres, visualisant les résultats et contrôlant le déroulement de l'acquisition.

**Fonctions principales:**

* **ouvrir_fenetre_reglages:** Crée une fenêtre pour ajuster les seuils de binarisation et d'autres paramètres.
* **mettre_a_jour_seuil:** Met à jour la valeur du seuil de binarisation dans la configuration.
* **mettre_a_jour_HLT:** Met à jour la valeur du seuil pour la détection des lignes.
* **ouvrir_fenetre_mesure:** Crée une fenêtre pour afficher des mesures spécifiques (à définir).
* **ouvrir_fenetre_Acqui:** Crée une fenêtre pour ajuster les offsets d'acquisition.
* **mettre_a_jour_offset_x:** Met à jour l'offset horizontal d'acquisition.
* **mettre_a_jour_offset_y:** Met à jour l'offset vertical d'acquisition.
* **maj_image_mode_stardard:** Met à jour l'affichage de l'image en mode standard.
* **maj_image_mode_bahtinov:** Met à jour l'affichage de l'image en mode Bahtinov.
* **maj_image_graphique_bas:** Met à jour le graphique en bas de l'interface.
* **Menu_cliquable:** Ajoute un menu cliquable à l'image principale pour contrôler les différents modes.

**Dépendances:**
* tkinter : Pour la création de l'interface graphique.
* cv2 : Pour le traitement d'images et l'affichage.
* numpy : Pour les opérations numériques sur les tableaux.
* config : Module contenant les paramètres de configuration.
* traitement_image : Module contenant les fonctions de traitement d'image.

**Auteur:** Thomas GAUTIER
**Date:** 08/11/2024
**Licence:** MIT

**Notes:**
* Ce module est fortement couplé au module `config` pour gérer les paramètres de l'application.
* Les fonctions de mise à jour des interfaces sont appelées périodiquement pour afficher les résultats des traitements en temps réel.
* Le design de l'interface utilisateur peut être amélioré en utilisant des widgets plus complexes de Tkinter ou en utilisant une bibliothèque graphique plus avancée.
"""


import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import Fonction.config as config
import Fonction.traitement_image as traitement_image  # Importer le module contenant les fonctions de traitement

###################################################################################################
## Fenêtre de réglage
###################################################################################################

def ouvrir_fenetre_reglages():
    # Créer une nouvelle fenêtre
    fenetre_reglages = tk.Toplevel()
    fenetre_reglages.title("Réglages des seuils")
    fenetre_reglages.geometry("400x200+1100+700")  # Définir la taille de la fenêtre
    fenetre_reglages.attributes('-topmost', True)

    # Réglage binaarisation
    label1 = tk.Label(fenetre_reglages, text=f"Seuil de binarisation")
    label1.pack()
    # Créer le slide bar
    scale1 = ttk.Scale(fenetre_reglages, from_=0, to=255, orient=tk.HORIZONTAL, length=300, command=lambda x: mettre_a_jour_seuil(x))
    scale1.set(config.val_seuil_BIN)  
    scale1.pack()

    label2 = tk.Label(fenetre_reglages, text=f"HLines_= threshold")
    label2.pack()

    # Créer le slide bar
    scale2 = ttk.Scale(fenetre_reglages, from_=0, to=100, orient=tk.HORIZONTAL, length=300, command=lambda x: mettre_a_jour_HLT(x))
    scale2.set(config.HLines_threshold)  
    scale2.pack()

    # Créer le bouton "Fermer"
    bouton_fermer = tk.Button(fenetre_reglages, text="Fermer", command=fenetre_reglages.destroy)
    bouton_fermer.pack()

def mettre_a_jour_seuil(nouvelle_valeur):
    # Correction du problème de compatibilité de type
    config.val_seuil_BIN = int(float(nouvelle_valeur))  # Convertir en float puis en int

def mettre_a_jour_HLT(nouvelle_valeur):
    # Correction du problème de compatibilité de type
    config.HLines_threshold = int(float(nouvelle_valeur))  # Convertir en float puis en int

###################################################################################################
## Fenêtre mesure
###################################################################################################
def ouvrir_fenetre_mesure():
    # Créer une nouvelle fenêtre
    fenetre_mesure = tk.Toplevel()
    fenetre_mesure.title("Mesure")
    fenetre_mesure.geometry("400x200+1100+700")  # Définir la taille de la fenêtre
    fenetre_mesure.attributes('-topmost', True)



    # Créer une image NumPy pour le dessin
    fenetre_mesure_image = np.zeros((400, 200, 3), dtype=np.uint8)  # Image noire de 400x400 pixels

    # Afficher l'image dans la fenêtre Tkinter
    fenetre_mesure_image_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(fenetre_mesure_image, cv2.COLOR_BGR2RGB)))
    fenetre_mesure_label = tk.Label(fenetre_mesure, image=fenetre_mesure_image_tk)
    fenetre_mesure_label.pack()




###################################################################################################
## Fenêtre d'acquisition
###################################################################################################
def ouvrir_fenetre_Acqui():
    # Créer une nouvelle fenêtre
    fenetre_acqui = tk.Toplevel()
    fenetre_acqui.title("Réglages des l'offsets d'aquisition")
    fenetre_acqui.geometry("400x300+1100+700")  # Définir la taille de la fenêtre

    labelx = tk.Label(fenetre_acqui, text=f"Offset X")
    labelx.pack()
    
    # Créer le slide bar
    scalex = ttk.Scale(fenetre_acqui, from_=0, to=800, orient=tk.HORIZONTAL, length=300, command=lambda x: mettre_a_jour_offset_x(x))
    scalex.set(config.offset_x)  # Initialiser avec la valeur globale
    scalex.pack()

    labely = tk.Label(fenetre_acqui, text=f"Offset Y")
    labely.pack()
    
    # Créer le slide bar
    scaley = ttk.Scale(fenetre_acqui, from_=0, to=300, orient=tk.VERTICAL, length=100, command=lambda x: mettre_a_jour_offset_y(x))
    scaley.set(config.offset_y)  # Initialiser avec la valeur globale
    scaley.pack()

    # Créer le bouton "Fermer"
    bouton_fermer = tk.Button(fenetre_acqui, text="Fermer", command=fenetre_acqui.destroy)
    bouton_fermer.pack()

def mettre_a_jour_offset_x(nouvelle_valeur):
    #global offset_x
    # Correction du problème de compatibilité de type
    config.offset_x = int(float(nouvelle_valeur))  # Convertir en float puis en int

def mettre_a_jour_offset_y(nouvelle_valeur):
    #global offset_y
    # Correction du problème de compatibilité de type
    config.offset_y = int(float(nouvelle_valeur))  # Convertir en float puis en int



###################################################################################################
## Fenêtre mode standard
###################################################################################################
def maj_image_mode_stardard(val):
        font = cv2.FONT_HERSHEY_SIMPLEX
        subimage = np.zeros((200, 400, 3), dtype=np.uint8)
        # Créer les deux frames de traitement
        frame_traitement_1 = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels 
        frame_traitement_2 = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels
        frame_traitement_3 = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels 
        frame_traitement_4 = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels
        frame_traitement_1b = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels 
        frame_traitement_2b = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels
        frame_traitement_3b = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels 
        frame_traitement_4b = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels
        ing_grey = np.zeros((100, 100, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels
        img_spectrum = np.zeros((300, 400, 3), dtype=np.uint8)

        # traitement d'image
        frame_traitement_1, frame_traitement_2 = traitement_image.Extract_ROI(config.frame, config.val_seuil_BIN, config.offset_x_ROI, config.offset_y_ROI,200)
        ing_grey=frame_traitement_2
        # condition d'empilage de l'image
        if config.empilage>1:
            frame_traitement_2=traitement_image.empilage_image(frame_traitement_2,config.empilage)

        frame_traitement_3, val_binarisation = traitement_image.calc_binarisation(frame_traitement_2, config.val_seuil_BIN)
        frame_traitement_4, val_FTM, img_spectrum = traitement_image.calculate_fourier_transform(frame_traitement_2)

        frame_traitement_1=traitement_image.zoom_image(frame_traitement_1,config.zoom)
        frame_traitement_2=traitement_image.zoom_image(frame_traitement_2,config.zoom)
        frame_traitement_3=traitement_image.zoom_image(frame_traitement_3,config.zoom)
        frame_traitement_4=traitement_image.zoom_image(frame_traitement_4,config.zoom)

        frame_traitement_2b=traitement_image.process_image_hist2(frame_traitement_2)
        #frame_traitement_2, frame_traitement_1b=traitement_image.detect_and_draw_lines(frame_traitement_2, config.HLines_threshold,100)

        # ajout de texte dans les images
        cv2.putText(frame_traitement_1, f"ROI couleur", (2, 18), font, 0.5, (255, 0, 0), 2)
        cv2.putText(frame_traitement_1b, f"X:{config.offset_x_ROI} Y:{config.offset_y_ROI}", (2, 18), font, 0.5, (255, 0, 0), 1)

        cv2.putText(frame_traitement_2, f"ROI Niv gris", (2, 18), font, 0.5, (255, 0, 0), 2)
        # condition d'empilage de l'image
        if config.empilage>1:
            cv2.putText(frame_traitement_2, f"BUFFER ON x{config.empilage}", (2, 95), font, 0.3, (0, 255, 0), 1)

        config.val_BIN_min = min(val_binarisation, config.val_BIN_min)
        cv2.putText(frame_traitement_3, f"Binarisation", (4, 18), font, 0.5, (255, 0, 0), 2)
        cv2.putText(frame_traitement_3b, f"{val_binarisation} pix", (2, 18), font, 0.5, (255, 0, 0), 1)
        cv2.putText(frame_traitement_3b, f"{config.val_BIN_min} min", (2, 36), font, 0.5, (255, 0, 0), 1)
        cv2.putText(frame_traitement_3b, f"{round(config.val_BIN_min/max(val_binarisation,0.001)*100,1)} %", (2, 90), font, 0.5, (255, 0, 0), 1)
        
        config.G_val_pix_bin = val_binarisation

        config.val_FTM_max = max(val_FTM, config.val_FTM_max)
        cv2.putText(frame_traitement_4, f"FTM", (40, 18), font, 0.5, (255, 0, 0), 2)
        cv2.putText(frame_traitement_4b, f"{val_FTM}", (2, 18), font, 0.5, (255, 0, 0), 1)
        cv2.putText(frame_traitement_4b, f"{config.val_FTM_max} max", (2, 36), font, 0.5, (255, 0, 0), 1)
        cv2.putText(frame_traitement_4b, f"{round(val_FTM/max(config.val_FTM_max,0.001)*100,1)} %", (2, 90), font, 0.5, (255, 0, 0), 1)

        config.G_val_FTM = val_FTM

        # Intégrer les frames dans la sous-image
        subimage[0:100, 0:100] = frame_traitement_1
        subimage[0:100, 100:200] = frame_traitement_2
        subimage[0:100, 200:300] = frame_traitement_3
        subimage[0:100, 300:400] = frame_traitement_4

        subimage[100:200, 0:100] = frame_traitement_1b
        subimage[100:200, 100:200] = frame_traitement_2b
        subimage[100:200, 200:300] = frame_traitement_3b
        subimage[100:200, 300:400] = frame_traitement_4b

        original_image = Image.fromarray(subimage)

        # Conversion en format OpenCV pour utiliser cv2.line
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        # Dessiner la ligne verticale rouge à x=100, de y=0 à y=200
        cv2.line(original_image_cv, (0, 100), (400, 100), (64, 64, 64), 1)
        cv2.line(original_image_cv, (100, 0), (100, 200), (255, 255, 255), 1)
        cv2.line(original_image_cv, (200, 0), (200, 200), (255, 255, 255), 1)
        cv2.line(original_image_cv, (300, 0), (300, 200), (255, 255, 255), 1)

        return original_image_cv, img_spectrum, ing_grey

###################################################################################################
## Fenêtre mode Bahtinov
###################################################################################################
def maj_image_mode_bahtinov(val):
        font = cv2.FONT_HERSHEY_SIMPLEX
        subimage = np.zeros((200, 400, 3), dtype=np.uint8)
        # Créer les deux frames de traitement
        frame_traitement_1 = np.zeros((200, 200, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels 
        frame_traitement_2 = np.zeros((200, 200, 3), dtype=np.uint8)  # Créer une image noire de 100x100 pixels


        # traitement d'image
        frame_traitement_1, frame_traitement_2 = traitement_image.Extract_ROI(config.frame, config.val_seuil_BIN, config.offset_x_ROI, config.offset_y_ROI, 400)
        frame_traitement_1, frame_traitement_2 = traitement_image.detect_and_draw_lines(frame_traitement_2, config.HLines_threshold,200)
        frame_traitement_1=traitement_image.zoom_image(frame_traitement_1,config.zoom)
        frame_traitement_2=traitement_image.zoom_image(frame_traitement_2,config.zoom)

        cv2.putText(frame_traitement_1, f"X:{config.offset_x_ROI} Y:{config.offset_y_ROI}", (2, 30), font, 0.3, (255, 0, 0), 1)

        # Intégrer les frames dans la sous-image
        subimage[0:200, 0:200] = frame_traitement_1
        subimage[0:200, 200:400] = frame_traitement_2

        original_image = Image.fromarray(subimage)

        # Conversion en format OpenCV pour utiliser cv2.line
        original_image_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)

        # Dessiner la ligne verticale rouge à x=100, de y=0 à y=200
        cv2.line(original_image_cv, (200, 0), (200, 200), (255, 255, 255), 1)

        return original_image_cv

def maj_image_graphique_bas(img_spectrum, image_grey):
        font = cv2.FONT_HERSHEY_SIMPLEX
        subimage2 = np.zeros((300, 400, 3), dtype=np.uint8)  # Créer une image noire 

        if config.FWHM_mode==1 and config.mode==1:
            subimage2 = img_spectrum

        elif (config.FWHM_mode>1) and config.mode==1:

            config.tab_FWHM[config.index_tab]=config.G_val_FTM
            config.tab_pix_bin[config.index_tab]=config.G_val_pix_bin

            config.tab_FWHM[config.index_tab+1]=0
            config.tab_pix_bin[config.index_tab+1]=0

            if config.FWHM_mode==2 or config.FWHM_mode==4:
                coul=(64,255,64)
                coul_min=(0,64,0)
                cv2.putText(subimage2, f"FWHM : {config.G_val_FTM}", (20, 275), font, 0.5, coul, 1)
                local_tab=config.tab_FWHM

                zoom_factor=traitement_image.zoom_factor_calc(local_tab)
                cv2.putText(subimage2, f"Zoom factor : {zoom_factor}", (170, 275), font, 0.5, coul, 1)

                for i in range(398):
                    height1 = int(local_tab[i]*6*zoom_factor)
                    height2 = int(local_tab[i+1]*6*zoom_factor)
                    if i>config.index_tab:
                        coul_l=coul_min
                    else:
                        coul_l=coul
                    cv2.line(subimage2, (i, 250 - height1), (i+1, 250 - height2), coul_l, 1)

                cv2.line(subimage2, (1, 250 - int(config.val_FTM_max*6*zoom_factor)), (399, 250 - int(config.val_FTM_max*6*zoom_factor)), coul_min, 1)
                cv2.putText(subimage2, f"FWHM Max : {config.val_FTM_max}", (300, 250-int(config.val_FTM_max*6*zoom_factor)-6), font, 0.3, coul_min, 1)

                cv2.line(subimage2, (config.index_tab, 250 - int(local_tab[config.index_tab]*6*zoom_factor)), (config.index_tab+10,  250 - int(local_tab[config.index_tab]*6*zoom_factor)), coul, 1)
                cv2.putText(subimage2, f"{config.G_val_FTM}", (config.index_tab+15, 250-int(local_tab[config.index_tab]*6*zoom_factor)+3), font, 0.3, coul, 1)
            
            if config.FWHM_mode==3 or config.FWHM_mode==4:
                coul=(64,64,255)
                coul_min=(0,0,64)
                cv2.putText(subimage2, f"PIX : {config.G_val_pix_bin}", (20, 290), font, 0.5, coul, 1)
                local_tab=config.tab_pix_bin

                zoom_factor=traitement_image.zoom_factor_calc(local_tab)
                cv2.putText(subimage2, f"Zoom factor : {zoom_factor}", (170, 290), font, 0.5, coul, 1)

                for i in range(398):
                    height1 = int(local_tab[i]*6*zoom_factor)
                    height2 = int(local_tab[i+1]*6*zoom_factor)
                    if i>config.index_tab:
                        coul_l=coul_min
                    else:
                        coul_l=coul
                    cv2.line(subimage2, (i, 250 - height1), (i+1, 250 - height2), coul_l, 1)
                    
                cv2.line(subimage2, (1, 250 - int(config.val_BIN_min*6*zoom_factor)), (399, 250 - int(config.val_BIN_min*6*zoom_factor)), coul_min, 1)

                cv2.line(subimage2, (config.index_tab, 250 - int(local_tab[config.index_tab]*6*zoom_factor)), (config.index_tab+10,  250 - int(local_tab[config.index_tab]*6*zoom_factor)), coul, 1)
                cv2.putText(subimage2, f"{config.G_val_pix_bin}", (config.index_tab+15, 250-int(local_tab[config.index_tab]*6*zoom_factor)+3), font, 0.3, coul, 1)

            cv2.line(subimage2, (1, 251), (400, 251 ), (64, 64, 64), 1)

            if config.FWHM_mode==5:
                subimage2=traitement_image.create_isometric_image(image_grey)
            
            
            

            config.index_tab =config.index_tab+1
            if config.index_tab>398:
                config.index_tab=0
                config.start=0
            
        return subimage2

def Menu_cliquable(resized_frame):      
    # Ajout d'un menu cliquable dans le bas de l'image
    font = cv2.FONT_HERSHEY_SIMPLEX

    if config.zoom==1:
        cv2.rectangle(resized_frame, (120, 370), (200, 390), (128, 24, 24), 1)
        cv2.putText(resized_frame, f"ZOOM X1", (122, 385), font, 0.5, (64, 64, 64), 1)
    else:
        cv2.rectangle(resized_frame, (120, 370), (200, 390), (0, 128, 0), -1)
        cv2.putText(resized_frame, f"ZOOM X{config.zoom}", (122, 385), font, 0.5, (255, 255, 255), 1)        
    
    if config.empilage==1:
        cv2.rectangle(resized_frame, (220, 370), (320, 390), (128, 24, 24), 1)
        cv2.putText(resized_frame, f"BUFFER x{config.empilage}", (222, 385), font, 0.5, (64, 64, 64), 1)
    else:
        cv2.rectangle(resized_frame, (220, 370), (320, 390), (0, 128, 0), -1)
        cv2.putText(resized_frame, f"BUFFER x{config.empilage}", (222, 385), font, 0.5, (255, 255, 255), 1)  

    if config.start==0:
        cv2.rectangle(resized_frame, (340, 370), (390, 390), (128, 24, 24), 1)
        cv2.putText(resized_frame, f"RESET", (342, 385), font, 0.5, (64, 64, 64), 1)
    else:
        cv2.rectangle(resized_frame, (340, 370), (390, 390), (0, 128, 0), -1)
        cv2.putText(resized_frame, f"RESET", (342, 385), font, 0.5, (255, 255, 255), 1) 
    
    if config.mode==1:  #Correspond au mode standard avec FTM et mesure de pixels allumés après binarisation
        cv2.rectangle(resized_frame, (10, 370), (100, 390), (128, 24, 24), 1)
        cv2.putText(resized_frame, f"BAHTINOV", (17, 385), font, 0.5, (64, 64, 64), 1)

    elif config.mode==2:  #Correspond au mode masque de Bahtinov
        cv2.rectangle(resized_frame, (10, 370), (100, 390), (0, 128, 0), -1)
        cv2.putText(resized_frame, f"BAHTINOV", (17, 385), font, 0.5, (255, 255, 255), 1)

    return resized_frame