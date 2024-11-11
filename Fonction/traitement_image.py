"""
## Module : traitement_image

**Description:**
Ce module contient un ensemble de fonctions dédiées au traitement d'images pour l'astrophotographie, 
en particulier pour l'aide à la focalisation d'un appareil photo CANON A7S.

**Fonctions principales:**

* **Recherche_ROI:** Recherche et suit une région d'intérêt (ROI) dans une image.
* **Extract_ROI:** Extrait une région d'intérêt spécifique d'une image.
* **detect_and_draw_lines:** Détecte les lignes dans une image et les dessine.
* **intersection:** Calcule le point d'intersection de deux droites.
* **dessiner_triangle:** Dessine un triangle à partir des intersections de lignes.
* **detecter_cercles:** Détecte les cercles dans une image.
* **traitement_cercle:** Traite les cercles détectés (regroupement, dessin).
* **barycentre:** Calcule le barycentre d'une image.
* **calculate_fourier_transform:** Calcule la transformée de Fourier et la FWHM.
* **process_image_hist2:** Calcule et affiche l'histogramme d'une image.
* **empilage_image:** Effectue un empilage d'images pour améliorer le signal.
* **taille_image:** Détermine la taille d'une image.
* **duplication_image:** Duplique une image en créant une copie avec trois canaux de couleur.
* **binarisation:** Binarisation d'une image en utilisant un seuil donné.
* **calc_binarisation:** Calcule l'image binaire et le nombre de pixels à 255.
* **copy_roi:** Copie une région d'intérêt (ROI) centrée sur les coordonnées données.
* **zoom_image:** Zoom sur une image en conservant la taille d'origine.
* **zoom_factor_calc:** Calcule le facteur de zoom en fonction de la valeur maximale dans un tableau.

**Dépendances:**
* OpenCV : Pour le traitement d'images.
* NumPy : Pour les opérations numériques sur les tableaux.
* Matplotlib : Pour la visualisation (utilisé dans certaines fonctions non présentées ici).
* SciPy : Pour l'optimisation (utilisé dans la fonction calculate_fourier_transform).

**Auteur:** Thomas GAUTIER
**Date:** 08/11/2024
**Licence:** MIT

**Notes:**
* Ce module est spécifiquement conçu pour le traitement d'images d'astrophotographie.
* Certaines fonctions utilisent des paramètres globaux définis dans le module `config`.
* Les fonctions de visualisation (graphiques, affichage de résultats) ne sont pas incluses dans cet extrait.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import Fonction.config as config
import scipy.optimize

def Recherche_ROI(frame, cX, cY,portion_size):
    """
    Traite une image en appliquant des opérations de binarisation, calcul de barycentre et dessin de zones d'intérêt.

    Args:
        frame (numpy.ndarray): L'image d'entrée.
        cX (int): Coordonnée x du centre initial.
        cY (int): Coordonnée y du centre initial.

    Returns:
        tuple: Un tuple contenant l'image traitée, la nouvelle coordonnée x du centre et la nouvelle coordonnée y du centre.
    """

    # Binarisation de l'image
    bin_image = binarisation(frame, 50)

    # Première zone d'intérêt (ROI) centrée sur le premier barycentre
    portion_size = 400
    cX = max(cX, 200)  # Limite les coordonnées pour éviter les débordements
    cY = max(cY, 200)

    roi1 = copy_roi(bin_image, (cX, cY), portion_size)  # Fonction pour copier la zone d'intérêt (non définie ici)
    bin_roi1 = binarisation(roi1, 50)
    cX1, cY1, _, _ = barycentre(bin_roi1)  # Recalcul du barycentre sur la ROI

    # Mise à jour des coordonnées du centre en fonction du nouveau barycentre
    cX = int(cX + (cX1 - portion_size / 2))
    cY = int(cY + (cY1 - portion_size / 2))

    # Limiter les coordonnées du centre pour éviter les débordements
    cX = max(cX, 100)
    cY = max(cY, 100)
    cX = min(cX, 700)
    cY = min(cY, 700)

    # Dupliquer l'image pour éviter de modifier l'original
    frame_processed = duplication_image(frame)

    # Dessiner un carré sur l'image en fonction du mode
    if config.mode == 1:
        cv2.rectangle(frame_processed, (cX - int(portion_size/8), cY - int(portion_size/8)), (cX + int(portion_size/8), cY + int(portion_size/8)), (0, 255, 0), 2)
    elif config.mode == 2:
        cv2.rectangle(frame_processed, (cX - int(portion_size/4), cY - int(portion_size/4)), (cX + int(portion_size/4), cY + int(portion_size/4)), (255, 0, 0), 2)

    # Ajouter le texte des coordonnées du barycentre sur l'image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame_processed, f"{cX}, {cY}", (cX - 47, cY - 34), font, 0.5, (0, 255, 0), 2)

    return frame_processed, cX, cY


def Extract_ROI(frame, niv_binarisation, Cx, Cy, portion_size):
    """
    Cette fonction traite une image pour extraire une région d'intérêt (ROI).

    Args:
        frame: L'image d'entrée.
        niv_binarisation: Le seuil de binarisation.
        Cx: Coordonnée x du centre initial de la première ROI.
        Cy: Coordonnée y du centre initial de la première ROI.
        portion_size: Taille de la première ROI.

    Returns:
        tuple: Un tuple contenant :
            - frame_traitement: La ROI finale (en couleur).
            - frame_traitement_bin: La ROI finale en niveaux de gris.
    """

    # Première ROI centrée sur le premier barycentre
    # On s'assure que le centre de la ROI ne soit pas trop proche du bord
    Cx = int(max(Cx, portion_size / 4))
    Cy = int(max(Cy, portion_size / 4))

    # Extraction de la première ROI et binarisation
    roi1 = copy_roi(frame, (Cx, Cy), portion_size)  # Fonction supposée copier une ROI
    bin_roi1 = binarisation(roi1, niv_binarisation)

    # Recalcul du barycentre sur la ROI binarisée
    Cx, Cy, _, _ = barycentre(bin_roi1)

    # Deuxième ROI centrée sur le nouveau barycentre (plus précise)
    portion_size = int(portion_size / 2)  # Réduction de la taille de la ROI
    roi2 = copy_roi(roi1, (Cx, Cy), portion_size)

    # Conversion en niveaux de gris et re-conversion en couleur (pourquoi ?)
    gray_image = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Binarisation de la deuxième ROI
    bin_roi2 = binarisation(roi2, niv_binarisation)

    # Redimensionnement des images à la taille souhaitée
    roi2 = cv2.resize(roi2, (portion_size, portion_size))
    bin_roi2 = cv2.resize(bin_roi2, (portion_size, portion_size))
    gray_image = cv2.resize(gray_image, (portion_size, portion_size))

    # Retourne la ROI finale en couleur et en niveaux de gris
    return roi2, gray_image















###################################################################################################
# Traitement géométrique
###################################################################################################

def detect_and_draw_lines(frame, HoughtLines_threshold, taille_image, canny_threshold1=50, canny_threshold2=150, min_line_length=10, max_line_gap=100, num_lines=6, angle_threshold=10):
    """
    Détecte et dessine les lignes dans une frame.

    Args:
        frame (numpy.ndarray): L'image sous forme de tableau NumPy.
        canny_threshold1 (int, optional): Seuil inférieur pour l'algorithme de Canny. Defaults to 50.
        canny_threshold2 (int, optional): Seuil supérieur pour l'algorithme de Canny. Defaults to 150.
        min_line_length (int, optional): Longueur minimale d'une ligne. Defaults to 10.
        max_line_gap (int, optional): Distance maximale entre les segments d'une ligne. Defaults to 10.
        num_lines (int, optional): Nombre maximum de lignes à dessiner. Defaults to 3.

    Returns:
        np.ndarray: Image avec les lignes dessinées.
    """

    coul = (0, 255, 0)  # Couleur verte pour les lignes

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des bords avec l'algorithme de Canny
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Détection des lignes avec la transformée de Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, HoughtLines_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)

    # Créer une copie de l'image en couleur pour dessiner les lignes
    color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    color_img2 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Filtrer les lignes pour ne garder que les plus pertinentes
    lines2 = []
    if lines is not None:
        for i in range(min(num_lines, len(lines))):
            line1 = lines[i][0]
            added = False
            for line2 in lines2:
                # Calcul de l'angle entre les deux lignes
                angle = abs(np.arctan2(line2[3]-line2[1], line2[2]-line2[0]) - np.arctan2(line1[3]-line1[1], line1[2]-line1[0])) * 180 / np.pi
                if angle <= angle_threshold or abs(angle - 180) <= angle_threshold:
                    # Si les lignes sont presque parallèles, on fusionne leurs coordonnées
                    x1_med = (line1[0] + line2[0]) // 2
                    y1_med = (line1[1] + line2[1]) // 2
                    x2_med = (line1[2] + line2[2]) // 2
                    y2_med = (line1[3] + line2[3]) // 2
                    line2[0] = x1_med
                    line2[1] = y1_med
                    line2[2] = x2_med
                    line2[3] = y2_med
                    added = True
                    break
            if not added:
                lines2.append(line1.tolist())

    # Limiter le nombre de lignes à 3
    lines2 = lines2[:3]
    lines_triangle = lines2

    # Dessiner les lignes filtrées en vert
    for line in lines2:
        x1, y1, x2, y2 = line
        cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Dessiner toutes les lignes détectées en rouge (pour comparaison)
    if lines is not None:
        for i in range(min(num_lines, len(lines))):
            for x1, y1, x2, y2 in lines[i]:
                cv2.line(color_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Zoomer sur la zone d'intérêt et dessiner le triangle
    zoom_factor = 4
    height, width, _ = color_img.shape
    center_x, center_y = width // 2, height // 2
    new_height, new_width = 200 * zoom_factor, 200 * zoom_factor
    top_left_x = max(0, center_x - new_width // 2)
    top_left_y = max(0, center_y - new_height // 2)
    bottom_right_x = min(width, center_x + new_width // 2)
    bottom_right_y = min(height, center_y + new_height // 2)
    color_img2 = np.zeros((new_height, new_width, 3), dtype=np.uint8)

    for i, line in enumerate(lines2):
        x1, y1, x2, y2 = line
        # Vérifier si la ligne est dans la zone de zoom
        if top_left_x <= x1 <= bottom_right_x and top_left_x <= x2 <= bottom_right_x and \
           top_left_y <= y1 <= bottom_right_y and top_left_y <= y2 <= bottom_right_y:
            # Ajuster les coordonnées pour le zoom
            x1_new = int((x1 - top_left_x) * new_width / (bottom_right_x - top_left_x))
            y1_new = int((y1 - top_left_y) * new_height / (bottom_right_y - top_left_y))
            x2_new = int((x2 - top_left_x) * new_width / (bottom_right_x - top_left_x))
            y2_new = int((y2 - top_left_y) * new_height / (bottom_right_y - top_left_y))

            lines_triangle[i] = x1_new, y1_new, x2_new, y2_new
            cv2.line(color_img2, (x1_new, y1_new), (x2_new, y2_new), coul, 1)

    # Dessiner le triangle sur l'image zoomée
    color_img2 = dessiner_triangle(lines_triangle, color_img2, coul)  

    # Copier la zone zoomée dans une image de taille spécifiée
    color_img2 = copy_roi(color_img2, (100 * zoom_factor, 100 * zoom_factor), taille_image)  

    return color_img, color_img2

def intersection(line1, line2):
    """
    Calcule le point d'intersection de deux droites définies par deux points.

    Args:
        line1: Tuple de coordonnées (x1, y1, x2, y2) de la première ligne.
        line2: Tuple de coordonnées (x1, y1, x2, y2) de la deuxième ligne.

    Returns:
    Le point d'intersection (x, y) ou None si les lignes sont parallèles.
    """

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calcul du dénominateur
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # Les lignes sont parallèles

    # Calcul des coordonnées du point d'intersection
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)

    return x, y

def dessiner_triangle(lines, image,coul):
    """
    Dessine un triangle à partir des intersections de trois lignes sur une image.

    Args:
        lines: Liste de tuples (x1, y1, x2, y2) représentant les lignes.
        image: Image OpenCV sur laquelle dessiner le triangle.

    Returns:
        L'image avec le triangle dessiné.
    """

    # Vérification : il faut au moins 3 lignes
    if len(lines) < 3:
        return image

    # Calcul des points d'intersection
    points = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            inter = intersection(lines[i], lines[j])
            if inter:
                points.append(inter)
                
    # Si on a moins de 3 points d'intersection, on ne peut pas former un triangle
    if len(points) < 3:
        return image

    # Création d'un contour à partir des points d'intersection
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1,1,2))

    # Dessin du contour rempli en vert
    image=cv2.fillPoly(image, [pts], coul)
    
    return image

def detecter_cercles(frame):
    """Détecte les cercles dans une image.

    Args:
        frame: L'image sous forme d'un tableau NumPy.

    Returns:
        Une liste de tuples (x, y, rayon) représentant les centres et les rayons des cercles détectés.
    """

    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre de lissage (optionnel)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détecter les cercles à l'aide de la transformée de Hough
    #circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 10, param1=20, param2=30, minRadius=5, maxRadius=200)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 1)
    # param1=50, param2=30, minRadius=10, maxRadius=200)
    if circles is not None: 
        #circles = circles[:10]
        # Convertir les coordonnées en entiers
        circles = np.round(circles[0, :]).astype("int")

        # Afficher les résultats
        #for (x, y, r) in circles:
            #print("Centre du cercle : (", x, ",", y, "), Rayon :", r)

        return circles
    else:
        print("Aucun cercle détecté.")
        return None
    
def traitement_cercle(image):

    # Détection des cercles
    cercles = detecter_cercles(image)
    
    # Dessiner les cercles sur l'image
    if cercles is not None:
        # Limiter le nombre de cercles à max_cercles
        
        index=0
        for x, y, r in cercles:
            cv2.circle(image, (x, y), r, (255, 0, 0), 1)  # Dessine un cercle rouge
            index+=1
            if index>1000:
                break 

    # Seuil de proximité (à ajuster selon vos besoins)
    seuil_distance = 25
    seuil_rayon = 10

    # Regrouper les cercles proches
    cercles_groupes = []
    cercles = list(cercles)
    while cercles:
        # Prendre le premier cercle
        cercle_courant = cercles.pop(0)
        groupe = [cercle_courant]

        # Trouver les cercles proches
        i = 0
        while i < len(cercles):
            cercle = cercles[i]
            distance = np.sqrt((cercle_courant[0] - cercle[0])**2 + (cercle_courant[1] - cercle[1])**2)
            diff_rayon = abs(cercle_courant[2] - cercle[2])
            if distance <= seuil_distance and diff_rayon <= seuil_rayon:
                groupe.append(cercles.pop(i))
            else:
                i += 1

        if groupe:
            x_moy = np.mean([c[0] for c in groupe])
            y_moy = np.mean([c[1] for c in groupe])
            r_moy = np.mean([c[2] for c in groupe])
            cercles_groupes.append((x_moy, y_moy, r_moy))
        else:
            a=a+1
            #print("Groupe vide")  # Gérer le cas où aucun cercle n'est suffisamment proche

    # Dessiner les cercles moyens
    index=0
    for x, y, r in cercles_groupes:
        #print("Centre du cercle === : (", x, ",", y, "), Rayon :", r)
        cv2.circle(image, (int(x), int(y)), int(r), (0, 0, 255), 2)
        index+=1
        if index>1000:
            break 

    return image


###################################################################################################
# Fonctions de traitement d'images complexe
###################################################################################################
def barycentre(image):
    """
    Calcule les coordonnées du barycentre d'une image.

    Args:
        image (numpy.ndarray): L'image d'entrée.

    Returns:
        tuple: Un tuple contenant les coordonnées x, y du barycentre, la hauteur et la largeur de l'image.
    """

    # Convertir l'image en niveaux de gris
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image_grey.shape

    # Calculer les moments d'ordre 0, 10 et 01 de l'image
    moments = cv2.moments(image_grey)

    # Calculer les coordonnées du barycentre en divisant les moments d'ordre 1 par le moment d'ordre 0
    moments['m00'] = max(moments['m00'], 1)  # Éviter une division par zéro
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])

    return cx, cy, height, width

def calculate_fourier_transform(image, pixel_size_mm=0.0064, output_size=(100, 100)):
    """
    Calcule la transformée de Fourier d'une image et génère des représentations visuelles sous forme d'images NumPy.

    Args:
        image: L'image d'entrée sous forme de tableau NumPy.
        pixel_size_mm: La taille d'un pixel en millimètres.
        output_size: La taille de sortie souhaitée pour les images NumPy.

    Returns:
        Un tuple contenant :
            - img_ft: L'image de la fonction de transfert optique sous forme de tableau NumPy.
            - img_spectrum: L'image du spectre de magnitude sous forme de tableau NumPy.
    """

    # Conversion en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Conversion en format 8 bits pour l'affichage
    try:

        # Calcul de la transformée de Fourier
        f_shift = np.fft.fftshift(np.fft.fft2(gray_image))

        # Calcul de la fonction de transfert optique
        optical_transfer_function = np.abs(f_shift)

        # Normalisation pour une meilleure visualisation
        if np.max(optical_transfer_function)>0:
            optical_transfer_function /= np.max(optical_transfer_function)
        optical_transfer_function *= 4  # Multiplie par 4 pour agrandir
        # Redimensionnement pour la taille de sortie souhaitée
        # On ajoute une dimension pour les canaux de couleur (3 pour RGB)
        img_ft = cv2.resize(optical_transfer_function, output_size)

        # Duplication du canal pour créer une image RGB
        img_ft = np.repeat(img_ft[:, :, np.newaxis], 3, axis=2)

        img_ft = (img_ft * 255).astype(np.uint8)
    except Exception:
        img_ft = gray_image

    # Calcul du spectre de magnitude
    config.spectrum = np.abs(f_shift)   
    middle_row = config.spectrum[config.spectrum.shape[0] // 2, :]
    half_length = len(middle_row) // 2
    middle_row_half = middle_row[half_length+1:]

    #calcul de la normalisation
    spectrum_norm = middle_row_half * 200 // max(np.max(middle_row_half),1)

    # Calcul de la largeur à mi-hauteur
    half_max = spectrum_norm[0] / 2

    def gaussienne(x, amplitude, centre, largeur):
        centre=0
        return amplitude * np.exp(-(x - centre)**2 / (2 * (largeur)**2))

    # Créer un tableau d'abscisses avec un pas de 0.1
    x_data = np.arange(0, len(spectrum_norm)) 

    # Estimation initiale des paramètres
    amplitude_init = np.max(spectrum_norm)
    centre_init = np.argmax(spectrum_norm) 
    largeur_init = 5  # Ajuster cette valeur selon votre estimation de la largeur

    try:
        # Ajustement de la courbe de Gauss
        params, covariance = scipy.optimize.curve_fit(gaussienne, x_data, spectrum_norm, p0=[amplitude_init, centre_init, largeur_init])

        # Création du tableau spectrum_gauss
        x_data = np.arange(0, len(spectrum_norm),1/80)
        spectrum_gauss = gaussienne(x_data, *params)

        # Vérifier si le tableau n'est pas vide
        if len(spectrum_gauss) == 0:
            fwhm_g = 0
        else:
            # Trouver les indices à gauche et à droite de la mi-hauteur avec des contrôles
            right_index = np.argmax(spectrum_gauss[0:] <= half_max)

            # Vérifier si les indices ont été trouvés
            if right_index == len(spectrum_gauss):
                fwhm_g = 0
            else:
                # Calculer la largeur à mi-hauteur
                fwhm_g = right_index /80
    except Exception:
        spectrum_gauss = None  # Ou une autre valeur par défaut

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_spectrum = np.zeros((300, 400, 3), dtype=np.uint8)
    # Dessin des lignes du spectre
    cv2.line(img_spectrum, (10, 250), (390, 250), (64, 64, 64), 2)
    cv2.line(img_spectrum, (10, 250), (10, 30), (64, 64, 64), 2)
    for i in range(min(len(spectrum_gauss), 380*10) - 30):
        height1 = min(int(spectrum_gauss[i]), 300)  # Multiplier l'indice par 10
        height2 = min(int(spectrum_gauss[(i+1)]), 300)
        cv2.line(img_spectrum, (10+int(1 + i/10), 250 - height1), (10+int(1 + (i/10+1)), 250 - height2), (0, 255, 0), 1)

    cv2.putText(img_spectrum, f"FWHM : {fwhm_g}", (20, 280), font, 0.5, (64, 255, 64), 1)    

    return img_ft, fwhm_g, img_spectrum

def process_image_hist2(frame):
    """
    Calcule l'histogramme d'une image et le dessine sous forme de graphique.

    Args:
        frame: L'image d'entrée à traiter.

    Returns:
        Une image représentant le graphique de l'histogramme.

    Cette fonction réalise les étapes suivantes :
    1. **Calcul de l'histogramme:** Calcule la distribution des niveaux de gris dans l'image.
    2. **Création du canvas:** Crée une image noire vierge sur laquelle le graphique sera dessiné.
    3. **Dessin des lignes:** Pour chaque niveau de gris, dessine une ligne verticale dont la hauteur correspond au nombre de pixels de ce niveau.
    4. **Redimensionnement:** Réduit la taille du graphique pour l'adapter aux besoins.
    """

    # Calcul de l'histogramme (canal gris uniquement)
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])

    # Création d'une image noire pour le graphique
    img_graph = np.zeros((100, 300, 3), dtype=np.uint8)

    # Dessin des lignes de l'histogramme
    for i in range(256):
        # Limiter la hauteur de la ligne à 100 pour ne pas dépasser le graphique
        height = min(int(hist[i]), 100)
        cv2.line(img_graph, (25 + i, 100), (25 + i, 100 - height), (255, 255, 255), 1)

    # Redimensionnement de l'image du graphique
    img_hist = cv2.resize(img_graph, (100, 100))

    return img_hist

def create_isometric_image(img, output_size=(300, 400)):


    # Handle potential image resizing
    if img.shape != (100, 100):
        img = cv2.resize(img, (100, 100))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create empty output image
    output = np.zeros(output_size, dtype=np.uint8)

    # Isometric projection parameters
    angle = np.pi / 4
    scale_v = 0.9
    scale_h = 1.4

    # Iterate through image pixels
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # 3D coordinates
            x = j
            y = i
            z = img[i, j]

            # Isometric projection calculations (element-wise)
            x_proj = int((np.round((x - y) * np.cos(angle) * scale_h + output_size[0] / 1.5)))
            y_proj = int((np.round((x + y) * np.sin(angle) * scale_v / 2 - z * scale_v + output_size[1]/2 +30)))

            # Ensure coordinates are within output image bounds
            x_proj = max(0, min(x_proj, 399))
            y_proj = max(0, min(y_proj, 299))

            # Draw pixel on output image
            output[y_proj, x_proj] = 255

    cv2.line(output, (100, 0), (100, 400 ), (32, 32, 32), 1)
    cv2.line(output, (300, 0), (300, 400 ), (32, 32, 32), 1)

    cv2.line(output, (50, 280), (50, 280-255 ), (64, 64, 64), 1)
    cv2.line(output, (350, 280), (350, 280-255 ), (64, 64, 64), 1)
    cv2.line(output, (0, 280), (100, 280 ), (64, 64, 64), 1)
    cv2.line(output, (300, 280), (400, 280 ), (64, 64, 64), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, f"Profil X", (30, 20), font, 0.4, (128,0,0), 1)
    cv2.putText(output, f"Profil Y", (330, 20), font, 0.4, (128,0,0), 1)
    cv2.putText(output, f"3D", (190, 20), font, 0.4, (128,0,0), 1)

    for i in range(99):
        z1=int(img[50,i])
        z1b=int(img[50,i+1])        
        z2=int(img[i,50])
        z2b=int(img[i+1,50])

        cv2.line(output, (i,280-z1),(i+1,280-z1b),(255,255,255))
        cv2.line(output, (i+300,280-z2),(i+301,280-z2b),(255,255,255))
    
        
        

    return output


###################################################################################################
# Fonctions d'empilage
###################################################################################################

previous_frames = []

def calculate_average_frame(frames):
    """
    Calcule l'image moyenne à partir d'une liste d'images.

    Args:
        frames (list): Une liste d'images au format NumPy.

    Returns:
        np.ndarray: L'image moyenne.
    """

    # Convertit la liste d'images en un tableau NumPy 3D
    # Chaque dimension représente (hauteur, largeur, canaux de couleur)
    frames_array = np.array(frames)

    # Calcule la moyenne de chaque pixel en considérant toutes les images
    # L'axe 0 correspond à la dimension des images (la profondeur du tableau)
    average_frame = np.mean(frames_array, axis=0).astype(np.uint8)

    return average_frame

def empilage_image(frames, max_frames):
    """
    Effectue un empilage d'images en calculant une moyenne glissante.

    Args:
        frames (numpy.ndarray): L'image à ajouter à l'empilement.

    Returns:
        np.ndarray: L'image moyenne si le nombre d'images est suffisant, sinon l'image d'entrée.
    """

    global previous_frames  # Utilise la variable globale pour stocker les images précédentes

    # Ajoute l'image courante à la liste des images précédentes
    previous_frames.append(frames)

    # Supprime l'image la plus ancienne si la liste est pleine
    if len(previous_frames) > max_frames:
        previous_frames.pop(0)

    # Si la liste contient suffisamment d'images, calcule l'image moyenne
    if len(previous_frames) == max_frames:
        average_frame = calculate_average_frame(previous_frames)
        return average_frame
    else:
        # Sinon, retourne simplement l'image d'entrée
        return frames

###################################################################################################
# Fonctions de traitement d'images simple
###################################################################################################  
def taille_image(image):
    """
    Détermine la hauteur et la largeur d'une image.

    Args:
        image (numpy.ndarray): L'image au format NumPy.

    Returns:
        tuple: Un tuple contenant la hauteur et la largeur de l'image.
    """

    if image is None:
        print("Impossible de charger l'image.")
    elif len(image.shape) == 2:  # Image en niveaux de gris
        height, width = image.shape
    elif image.shape[2] == 3:  # Image en couleur RGB
        height, width, channels = image.shape
    elif image.shape[2] == 4:  # Image en couleur RGBA
        height, width, channels = image.shape[:3]
    else:
        print("Le type de couleur de l'image est inconnu.")

    return height, width

def duplication_image(image):
    """
    Duplique une image en créant une copie avec trois canaux de couleur.

    Args:
        image (numpy.ndarray): L'image d'entrée.

    Returns:
        numpy.ndarray: L'image dupliquée avec trois canaux de couleur.
    """

    height, width = taille_image(image)
    output_image = np.zeros((height, width, 3), dtype=np.uint8)
    output_image[:, :] = image
    return output_image

def binarisation(image, seuil):
    """
    Binarisation d'une image en utilisant un seuil donné.

    Args:
        image (numpy.ndarray): L'image d'entrée.
        seuil (int): Le seuil de binarisation.

    Returns:
        numpy.ndarray: L'image binaire.
    """

    output_image = duplication_image(image)
    _, output_image = cv2.threshold(output_image, seuil, 255, cv2.THRESH_BINARY)
    return output_image

def calc_binarisation(image, seuil):
    """
    Calcule l'image binaire et le nombre de pixels à 255.

    Args:
        image: L'image d'entrée.
        seuil: Le seuil de binarisation.

    Returns:
        Un tuple contenant :
            - output_image: L'image binaire.
            - nb_pixel: Le nombre de pixels à 255 dans l'image binaire.
    """

    output_image = duplication_image(image)  # Assurez-vous que cette fonction duplique correctement l'image
    _, output_image = cv2.threshold(output_image, seuil, 255, cv2.THRESH_BINARY)

    # Calcul du nombre de pixels à 255
    nb_pixel = int(np.count_nonzero(output_image == 255)/3)

    return output_image, nb_pixel

def copy_roi(image, center, size):
    """
    Copie une région d'intérêt (ROI) centrée sur les coordonnées données.
     Args:
        image: L'image source.
        center: Le tuple (x, y) des coordonnées du centre de la ROI.
        size: La taille de la ROI (carré).
     Returns:
        La ROI extraite.
    """
    x_start = max(0, center[0] - int(size // 2))
    y_start = max(0, center[1] - int(size // 2))
    x_end = min(image.shape[1], center[0] + int(size // 2))
    y_end = min(image.shape[0], center[1] + int(size // 2))

    return image[y_start:y_end, x_start:x_end].copy()

def zoom_image(image, zoom):
    """
    Zoom sur une image en conservant la taille d'origine.

    Args:
        image (numpy.ndarray): L'image d'entrée.
        zoom (float): Facteur de zoom.

    Returns:
        numpy.ndarray: L'image zoomée.
    """

    output_image = duplication_image(image)  # Duplique l'image pour éviter de modifier l'original
    height, width = taille_image(image)
    taille = height  # Utilise la hauteur comme référence pour la taille de la zone de zoom

    # Applique le zoom à l'image en utilisant l'interpolation la plus proche
    image = cv2.resize(image, (int(height * zoom), int(width * zoom)), interpolation=cv2.INTER_NEAREST)

    # Calcule les coordonnées du centre et les dimensions de la zone à copier
    height = int(height * zoom // 2)
    width = int(width * zoom // 2)

    # Copie la zone zoomée centrée dans l'image de sortie
    output_image = copy_roi(image, (height, width), taille)  # Fonction pour copier la zone d'intérêt (non définie ici)

    return output_image

def zoom_factor_calc(local_tab):
    """
    Calcule le facteur de zoom en fonction de la valeur maximale dans le tableau.

    Args:
        local_tab (numpy.ndarray): Tableau de valeurs.

    Returns:
        float: Facteur de zoom.
    """

    max_value = np.max(local_tab)

    if max_value < 2.5:
        zoom_factor = 16
    elif max_value < 5:
        zoom_factor = 8
    elif max_value < 10:
        zoom_factor = 4
    elif max_value < 20:
        zoom_factor = 2
    elif max_value < 40:
        zoom_factor = 1
    elif max_value < 80:
        zoom_factor = 0.5
    elif max_value < 160:
        zoom_factor = 0.25
    elif max_value < 320:
        zoom_factor = 0.125
    elif max_value < 640:
        zoom_factor = 0.125 / 2
    elif max_value < 1080:
        zoom_factor = 0.125 / 4
    elif max_value < 2160:
        zoom_factor = 0.125 / 8
    else:
        zoom_factor = 0.125 / 16

    return zoom_factor
