"""
Simulation d'une trajectoire balistique (sans ou avec frottement).
Genere des donnees theoriques et des observations bruitees
exploitables par un algorithme d'optimisation.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

G = 9.81


# ---------------------------------------------------------------------------
# Génère une trajectoire
# ---------------------------------------------------------------------------
def generate_trajectory(v0, theta, t_max, num_points):
    """
    Calcule la trajectoire théorique d'un projectile sans frottement.

    Parameters
    ----------
    v0 : float
        Vitesse initiale en m/s.
    theta : float
        Angle de tir en radians.
    t_max : float
        Durée maximale de la simulation en secondes.
    num_points : int
        Nombre de points temporels.

    Returns
    -------
    t : np.ndarray
        Vecteur temps de taille num_points.
    x : np.ndarray
        Positions horizontales théoriques (m).
    y : np.ndarray
        Positions verticales théoriques (m).
    """
    t = np.linspace(0, t_max, num_points)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * G * t**2
    return t, x, y

# ---------------------------------------------------------------------------
# Génère une trajectoire avec trainée linéaire
# ---------------------------------------------------------------------------
def generate_trajectory_drag(v0, theta, gamma, t_max, num_points):
    """
    Calcule la trajectoire avec trainée linéaire (gamma = k/m).
    EDO : dvx/dt = -gamma*vx, dvy/dt = -g - gamma*vy
    Integration par RK45.

    Parameters
    ----------
    v0 : float
        Vitesse initiale (m/s).
    theta : float
        Angle de tir (rad).
    gamma : float
        Coéfficient de trainée k/m (s^-1).
    t_max : float
        Duree maximale (s).
    num_points : int
        Nombre de points temporels.

    Returns
    -------
    t, x, y : np.ndarray
    """
    def edo(t, state):
        _, _, vx, vy = state
        return [vx, vy, -gamma * vx, -G - gamma * vy]

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    etat0 = [0.0, 0.0, vx0, vy0]

    t_eval = np.linspace(0, t_max, num_points)
    sol = solve_ivp(edo, [0, t_max], etat0, t_eval=t_eval, method='RK45',
                    rtol=1e-10, atol=1e-12)

    return sol.t, sol.y[0], sol.y[1]

# ---------------------------------------------------------------------------
# Ajout du bruit
# ---------------------------------------------------------------------------
def ajouter_bruit(x, y, sigma=0.05):
    """
    Ajoute un bruit blanc gaussien aux coordonnées pour simuler
    l'imprecision d'une camera de smartphone.

    Parameters
    ----------
    x, y : np.ndarray
        Coordonnées theoriques.
    sigma : float
        Ecart-type du bruit en mètres.

    Returns
    -------
    x_obs, y_obs : np.ndarray
        Coordonnées bruitées.
    """
    bruit_x = np.random.normal(0, sigma, size=x.shape)
    bruit_y = np.random.normal(0, sigma, size=y.shape)
    return x + bruit_x, y + bruit_y

# ---------------------------------------------------------------------------
# Graphe de la trajectoire
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 8

def tracer_trajectoire(t, x_th, y_th, x_obs, y_obs, v0, theta_deg):
    """
    Trace la trajectoire théorique et les observations bruitées
    sur un même graphique.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_facecolor('#F8F9FA')

    ax.plot(x_th, y_th, '-', color='#1D4E89', linewidth=1, label='Trajectoire théorique')
    ax.scatter(x_obs, y_obs, c='#E07A5F', s=10, alpha=0.7, zorder=5,
               label='Observations bruitees')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(f'Trajectoire balistique  |  v0 = {v0} m/s, '
                 f'theta = {theta_deg} deg')
    ax.legend()
    ax.set_ylim(bottom=-0.5)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectoire_simulee.png', dpi=1200)
    plt.show()


if __name__ == '__main__':

    # -- Parametres de simulation -------------------------------------------
    v0 = 15.0                   # vitesse initiale (m/s)
    theta = np.pi / 4           # angle de tir (rad)
    theta_deg = np.degrees(theta)
    num_points = 100
    sigma_bruit = 0.08          # ecart-type du bruit (m)

    # Mettre a 0 pour le modele sans frottement,
    # ou a une valeur > 0 pour simuler la trainee (ex: 0.3)
    GAMMA = 0.5

    # -- Generation ---------------------------------------------------------
    if GAMMA > 0:
        t_vol = 2 * v0 * np.sin(theta) / G  # estimation initiale
        t_max = t_vol * 1.2
        t, x_th, y_th = generate_trajectory_drag(v0, theta, GAMMA,
                                                  t_max, num_points)
        print(f"Mode : AVEC frottement (gamma={GAMMA})")
    else:
        t_vol = 2 * v0 * np.sin(theta) / G
        t_max = t_vol * 1.05
        t, x_th, y_th = generate_trajectory(v0, theta, t_max, num_points)
        print(f"Mode : SANS frottement")

    x_obs, y_obs = ajouter_bruit(x_th, y_th, sigma=sigma_bruit)

    # Tronquer les points sous le sol
    masque = y_th >= 0
    t = t[masque]
    x_th = x_th[masque]
    y_th = y_th[masque]
    x_obs = x_obs[masque]
    y_obs = y_obs[masque]

    # -- Affichage ----------------------------------------------------------
    print(f"Parametres : v0={v0} m/s, theta={theta_deg:.1f} deg"
          f"{f', gamma={GAMMA}' if GAMMA > 0 else ''}")
    print(f"Points generes : {len(t)}")
    print(f"Bruit (sigma) : {sigma_bruit} m\n")

    print("Apercu des donnees (t, x_obs, y_obs) :")
    for i in range(min(8, len(t))):
        print(f"  t={t[i]:.4f} s  |  x={x_obs[i]:.4f} m  |  y={y_obs[i]:.4f} m")
    if len(t) > 8:
        print(f"  ... ({len(t) - 8} points supplementaires)")

    # -- Sauvegarde CSV (meme format que extraction_trajectoire.py) ----------
    with open("observations.csv", "w") as f:
        f.write("t,x,y\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.6f},{x_obs[i]:.6f},{y_obs[i]:.6f}\n")
    print("Donnees bruitees sauvegardees dans observations.csv")

    with open("trajectoire_theorique.csv", "w") as f:
        f.write("t,x,y\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.6f},{x_th[i]:.6f},{y_th[i]:.6f}\n")
    print("Trajectoire theorique sauvegardee dans trajectoire_theorique.csv")

    tracer_trajectoire(t, x_th, y_th, x_obs, y_obs, v0, theta_deg)
