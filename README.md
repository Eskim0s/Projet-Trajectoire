# Projet Trajectoire - MA266

Ce projet implémente une suite de scripts Python permettant de reconstruire la trajectoire d'un projectile à partir de données vidéo ou simulées.

## Installation

Les dépendances requises s'installent via pip :

```bash
pip install numpy scipy matplotlib torch torchdiffeq
pip install opencv-python  # pour l'extraction video uniquement
```

## Fonctionnalites

1. **Extraction video** : Détection d'un projectile par seuillage HSV et calcul du centroide avec OpenCV, calibration pixel-mètre avec inversion de l'axe y.


2. **Simulation** : Génération de trajectoires théoriques (sans frottement, trainee lineaire ou quadratique) avec ajout de bruit gaussien pour simuler l'imprecision d'une camera.


3. **Probleme inverse sans frottement** : Identification de la vitesse initiale v0 et de l'angle theta par minimisation des moindres carres sous contraintes physiques (L-BFGS-B).


4. **Probleme inverse avec trainee lineaire** : Résolution de l'EDO paramétrique par différenciation automatique (torchdiffeq) et comparaison de trois méthodes de gestion de la contrainte gamma >= 0 : pénalisation, projection et barrière logarithmique.


5. **Analyse de sensibilité** : Etude de la variance des paramètres estimés en fonction du niveau de bruit, avec regularisation de Tikhonov pour stabiliser l'optimisation.


6. **Extension - trainee quadratique** : Comparaison numérique et visuelle du modèle linéaire avec le modèle quadratique.


## Structure du projet

| Script | Description |
|---|---|
| `extraction_trajectoire.py` | Extraction de positions depuis une vidéo (OpenCV) |
| `simulation_trajectoire.py` | Simulation sans frottement ou avec trainée linéaire |
| `simulation_trajectoire_quad.py` | Simulation avec trainée quadratique |
| `methode_simple.py` | Identification v0, theta (modele sans trainée) |
| `methode_frottements.py` | Identification avec trainée linéaire, comparaison des contraintes |
| `methode_drag_quad.py` | Comparaison modèle linéaire vs quadratique |
| `sensibilite.py` | Analyse de sensibilité et régularisation de Tikhonov |

## Exemples d'utilisation

Voici un workflow simple pour tester les fonctionnalités principales :

```bash
# 1. Générer des observations simulees avec trainée linéaire
#    (modifier GAMMA dans le script pour choisir le modèle)
# $ python3 simulation_trajectoire.py

# 2. Identifier les paramètres sans frottement
# $ python3 methode_simple.py

# 3. Identifier les paramètres avec frottement (3 méthodes de contrainte)
# $ python3 methode_frottements.py

# 4. Analyser la sensibilité au bruit
# $ python3 sensibilite.py
```

Les scripts de methode chargent automatiquement `observations.csv`. Si le fichier est absent, ils lancent la simulation correspondante. Tous les CSV partagent le meme format interchangeable :

```
t,x,y
0.000000,0.238100,0.168900
...
```

## References

Les constantes physiques et modèles implémentés se basent sur les standards suivants :

* **Mouvement de projectile** : [Wikipedia - Projectile motion](https://en.wikipedia.org/wiki/Projectile_motion)
* **OpenCV documentation** : [https://docs.opencv.org/](https://docs.opencv.org/)
* **torchdiffeq (Neural ODE)** : [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Auteur

Liam ADGH
Aurélien de Almeida
