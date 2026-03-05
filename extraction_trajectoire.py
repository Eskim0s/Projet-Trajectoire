"""
Script d'extraction de la trajectoire d'un projectile depuis une video.
Utilise OpenCV pour detecter l'objet par seuillage HSV et calcul du centroide.
Retourne une liste de tuples (t, x, y) en metres avec y oriente vers le haut.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Parametres de seuillage HSV -- a ajuster selon la couleur du projectile
# ---------------------------------------------------------------------------
HSV_LOWER = np.array([20, 100, 100])   # borne basse (ex: balle orange/jaune)
HSV_UPPER = np.array([35, 255, 255])   # borne haute

# Taille minimale du contour en pixels pour filtrer le bruit
MIN_CONTOUR_AREA = 50


# ---------------------------------------------------------------------------
# Calibration : conversion pixels -> metres
# ---------------------------------------------------------------------------
def calibrer(longueur_reference_m, longueur_reference_px):
    """
    Calcule le facteur de conversion pixel -> metre.

    Parameters
    ----------
    longueur_reference_m : float
        Longueur connue dans la scene, exprimee en metres.
    longueur_reference_px : float
        Meme longueur mesuree en pixels sur l'image.

    Returns
    -------
    float
        Facteur d'echelle (metres par pixel).
    """
    if longueur_reference_px <= 0:
        raise ValueError("La longueur de reference en pixels doit etre > 0")
    return longueur_reference_m / longueur_reference_px


def pixels_vers_metres(x_px, y_px, echelle, hauteur_image):
    """
    Convertit des coordonnees pixel en coordonnees metriques.
    L'axe y est inverse : l'origine est placee en bas a gauche.

    Parameters
    ----------
    x_px, y_px : float
        Coordonnees en pixels (origine en haut a gauche).
    echelle : float
        Facteur metres/pixel issu de calibrer().
    hauteur_image : int
        Hauteur de l'image en pixels.

    Returns
    -------
    (float, float)
        Coordonnees (x, y) en metres, y oriente vers le haut.
    """
    x_m = x_px * echelle
    y_m = (hauteur_image - y_px) * echelle
    return x_m, y_m


# ---------------------------------------------------------------------------
# Extraction des positions
# ---------------------------------------------------------------------------
def extraire_positions(chemin_video, echelle,
                       hsv_lower=None, hsv_upper=None,
                       afficher=False):
    """
    Parcourt la video image par image, detecte le projectile par seuillage
    HSV et retourne la liste des positions calibrees.

    Parameters
    ----------
    chemin_video : str
        Chemin vers le fichier video.
    echelle : float
        Facteur metres/pixel (resultat de calibrer()).
    hsv_lower : np.ndarray, optional
        Borne basse HSV. Par defaut HSV_LOWER global.
    hsv_upper : np.ndarray, optional
        Borne haute HSV. Par defaut HSV_UPPER global.
    afficher : bool
        Si True, affiche la detection en temps reel (appuyer sur 'q' pour quitter).

    Returns
    -------
    list[tuple[float, float, float]]
        Liste de tuples (t, x, y) ou t est en secondes et x, y en metres.
    """
    if hsv_lower is None:
        hsv_lower = HSV_LOWER
    if hsv_upper is None:
        hsv_upper = HSV_UPPER

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la video : {chemin_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Impossible de lire le FPS de la video")

    hauteur_image = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    positions = []
    index_frame = 0

    noyau = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = index_frame / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, hsv_lower, hsv_upper)

        masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)
        masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, noyau)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            plus_grand = max(contours, key=cv2.contourArea)
            aire = cv2.contourArea(plus_grand)

            if aire >= MIN_CONTOUR_AREA:
                M = cv2.moments(plus_grand)
                if M["m00"] > 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]

                    x_m, y_m = pixels_vers_metres(cx, cy, echelle,
                                                  hauteur_image)
                    positions.append((t, x_m, y_m))

                    if afficher:
                        centre = (int(cx), int(cy))
                        cv2.circle(frame, centre, 6, (0, 255, 0), -1)
                        cv2.putText(frame,
                                    f"({x_m:.3f}, {y_m:.3f}) m",
                                    (centre[0] + 10, centre[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 1)

        if afficher:
            cv2.imshow("Detection", frame)
            cv2.imshow("Masque", masque)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        index_frame += 1

    cap.release()
    if afficher:
        cv2.destroyAllWindows()

    return positions


# ---------------------------------------------------------------------------
# Utilitaire : sauvegarde CSV
# ---------------------------------------------------------------------------
def sauvegarder_csv(positions, chemin_csv):
    """Ecrit les positions dans un fichier CSV (t, x, y)."""
    with open(chemin_csv, 'w') as f:
        f.write("t,x,y\n")
        for t, x, y in positions:
            f.write(f"{t:.6f},{x:.6f},{y:.6f}\n")


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # -- Configuration utilisateur ------------------------------------------

    CHEMIN_VIDEO = "video_projectile.mp4"

    # Calibration : mesurer un objet de reference dans la video
    # Exemple : un objet de 1.0 m mesure 320 pixels sur l'image
    LONGUEUR_REF_M = 1.0
    LONGUEUR_REF_PX = 320.0

    echelle = calibrer(LONGUEUR_REF_M, LONGUEUR_REF_PX)
    print(f"Echelle de calibration : {echelle:.6f} m/px")

    # -- Extraction ---------------------------------------------------------

    positions = extraire_positions(CHEMIN_VIDEO, echelle, afficher=True)

    print(f"\n{len(positions)} positions extraites.")
    for t, x, y in positions[:10]:
        print(f"  t={t:.4f} s  |  x={x:.4f} m  |  y={y:.4f} m")

    if len(positions) > 10:
        print(f"  ... ({len(positions) - 10} positions supplementaires)")

    # -- Sauvegarde ---------------------------------------------------------

    sauvegarder_csv(positions, "positions_trajectoire.csv")
    print("Resultats sauvegardes dans positions_trajectoire.csv")
