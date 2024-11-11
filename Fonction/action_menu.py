"""
## Module : interactions

**Description:**
Ce module gère les interactions entre l'utilisateur et l'application via l'interface graphique. Il définit les fonctions liées à la création des menus, à la gestion des événements utilisateur (clics de souris, sélection dans les menus) et à la mise à jour des paramètres de l'application en conséquence.

**Fonctions principales:**

* **update_val_min:** Met à jour les valeurs minimales pour les différents paramètres (FTM, binarisation).
* **update_reglage:** Ouvre les fenêtres de réglages spécifiques (seuils, acquisition).
* **mesure:** Active les différents modes de mesure (FWHM, nombre de pixels, ...).
* **button_click:** Gère les clics sur les boutons pour déplacer la zone d'acquisition.
* **creation_menu_principal:** Crée la barre de menu principale avec les différents sous-menus.

**Dépendances:**
* tkinter : Pour la création de l'interface graphique et la gestion des événements.
* config : Module contenant les paramètres de configuration de l'application.
* Fenetres : Module contenant les fonctions de création des fenêtres spécifiques.

**Auteur:** Thomas GAUTIER
**Date:** 08/11/2024
**Licence:** MIT

**Notes:**
* Ce module est fortement couplé au module `config` pour gérer les paramètres de l'application.
* Les fonctions de ce module sont appelées par l'événement loop de Tkinter pour réagir aux actions de l'utilisateur.
* Le module `Fenetres` contient les définitions des différentes fenêtres de l'application.
"""


import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import Fonction.Fenetres as Fenetres
import Fonction.config as config

def update_val_min(val):
    # Ici, vous mettrez à jour la valeur de Val_FTM_min
    # Par exemple, en appelant votre fonction de calcul de la transformée de Fourier
    #global val_FTM_min, val_FTM_log_min, val_BIN_min
    if val == "FTM":
        config.val_FTM_min = 1000000
        config.val_FTM_log_min = 1000000
    elif val == "Binarisation":
        config.val_BIN_min = 1000000
    elif val == "Tout":
        config.val_FTM_max = 0
        config.val_FTM_log_min = 1000000
        config.val_BIN_min = 1000000
    else:
        a=a

def update_reglage(val):
    # Ici, vous mettrez à jour la valeur de Val_FTM_min
    # Par exemple, en appelant votre fonction de calcul de la transformée de Fourier
    #global val_seuil_BIN
    if val == "Seuils":
        Fenetres.ouvrir_fenetre_reglages()
    elif val == "Acquisition":
        Fenetres.ouvrir_fenetre_Acqui()
    else:
        a=a+1

def mesure(val):
    # Ici, vous mettrez à jour la valeur de Val_FTM_min
    # Par exemple, en appelant votre fonction de calcul de la transformée de Fourier
    #global val_seuil_BIN
    if val == "mesure_FTM":
        config.FWHM_mode=1
    elif val == "mesure_FWHM":
        config.FWHM_mode=2
    elif val == "mesure_pix_bin":
        config.FWHM_mode=3
    elif val == "mesure_all":
        config.FWHM_mode=4
    elif val == "3D":
        config.FWHM_mode=5

# Fonction pour gérer les clics sur les boutons du menu
def button_click(direction):
    #global offset_x, offset_y
    if direction == "X+":
        config.offset_x += 10
    elif direction == "X-":
        config.offset_x -= 10
    elif direction == "Y+":
        config.offset_y += 10
    else:
        config.offset_y -= 10

def creation_menu_principal():
    # Création de la barre de menu
    menu_bar = tk.Menu(config.root)
    # Menu Déplacé

    config.root.config(menu=menu_bar)

    # Menu "Initialisation"
    init_menu = tk.Menu(menu_bar, tearoff=0)
    init_menu.add_command(label="Tous les Min/Max", command=lambda: update_val_min("Tout"))
    init_menu.add_command(label="Reset FWHM Max", command=lambda: update_val_min("FTM"))
    init_menu.add_command(label="Reset NB Pix Min", command=lambda: update_val_min("Binarisation"))
    menu_bar.add_cascade(label="Initialisation", menu=init_menu)

    # Menu "Mesure"
    init_menu = tk.Menu(menu_bar, tearoff=0)
    init_menu.add_command(label="Spectre FTM", command=lambda: mesure("mesure_FTM"))
    init_menu.add_command(label="Graph FWHM", command=lambda: mesure("mesure_FWHM"))
    init_menu.add_command(label="Graph NB Pix", command=lambda: mesure("mesure_pix_bin"))  
    init_menu.add_command(label="Graph FWHM & NB Pix", command=lambda: mesure("mesure_all")) 
    init_menu.add_command(label="Profil 3D", command=lambda: mesure("3D"))
    menu_bar.add_cascade(label="Graphique", menu=init_menu)

    # Menu "Réglage"
    init_menu = tk.Menu(menu_bar, tearoff=0)
    init_menu.add_command(label="Seuils", command=lambda: update_reglage("Seuils"))
    init_menu.add_command(label="Position zone d'acquisition", command=lambda: update_reglage("Acquisition"))
    menu_bar.add_cascade(label="Réglages", menu=init_menu)

    # Création du label pour afficher l'image
    label = tk.Label(config.root)
    label.pack()

