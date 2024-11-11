"""
## Module : config

**Description:**
Ce module contient les variables globales utilisées dans l'ensemble de l'application. Ces variables servent à stocker les paramètres de configuration, les valeurs intermédiaires des calculs et les états de l'application.

**Variables principales:**

* **Offsets:** `offset_x`, `offset_y`, `offset_x_ROI`, `offset_y_ROI` : Coordonnées des zones d'intérêt.
* **Tailles:** `crop_size`, `display_size` : Tailles des images capturées et affichées.
* **Valeurs min/max:** `val_FTM_max`, `val_FTM_log_min`, `val_BIN_min` : Valeurs extrêmes pour les mesures.
* **Seuils:** `val_seuil_BIN`, `HLines_threshold` : Seuils pour la binarisation et la détection de lignes.
* **Données d'image:** `spectrum`, `frame`, `root` : Données liées à l'image en cours de traitement et à l'interface graphique.
* **Modes et états:** `mode`, `zoom`, `empilage`, `start`, `FWHM_mode`, `nb_image` : Indicateurs des différents modes de fonctionnement et états de l'application.
* **Valeurs intermédiaires:** `G_val_FTM`, `G_val_pix_bin`, `index_tab`, `tab_FWHM`, `tab_pix_bin` : Valeurs utilisées pour les calculs et les affichages.
"""

# Variables globales
offset_x = 100
offset_y = 100
offset_x_ROI = 200
offset_y_ROI = 200

crop_size = 800  # Taille de la zone de capture
display_size = 400  # Taille d'affichage dans la fenêtre
val_FTM_max = 0
val_FTM_log_min = 1000000
val_BIN_min = 1000000

val_seuil_BIN = 50
HLines_threshold = 15

spectrum = None

frame = None
root = None

mode=1
zoom=1
empilage=1
start=0

G_val_FTM=0
G_val_pix_bin=0

index_tab=0
tab_FWHM = None
tab_pix_bin = None
FWHM_mode=5

nb_image=0