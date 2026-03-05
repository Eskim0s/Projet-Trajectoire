"""
Probleme inverse : identification de v0 et theta.
Minimise l'ecart entre les observations et le modele balistique sans frottement
en utilisant scipy.optimize.minimize avec contraintes physiques.
"""

import os
import subprocess
import sys

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

G = 9.81


# ---------------------------------------------------------------------------
# Modele balistique sans frottement
# ---------------------------------------------------------------------------
def modele_trajectoire(t, v0, theta, x0=0.0, y0=0.0):
    """
    Calcule les positions (x, y) du projectile pour chaque instant t.

    Parameters
    ----------
    t : np.ndarray
        Vecteur temps (s).
    v0 : float
        Vitesse initiale (m/s).
    theta : float
        Angle de tir (rad).
    x0, y0 : float
        Position initiale (m).

    Returns
    -------
    x, y : np.ndarray
    """
    x = x0 + v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * G * t**2
    return x, y


# ---------------------------------------------------------------------------
# Fonction de cout (somme des residus au carre)
# ---------------------------------------------------------------------------
def fonction_cout(params, t, x_obs, y_obs):
    """
    J(v0, theta, x0, y0) = sum_i [(x_obs_i - x(t_i))^2 + (y_obs_i - y(t_i))^2]
    """
    v0, theta, x0, y0 = params
    x_mod, y_mod = modele_trajectoire(t, v0, theta, x0, y0)
    residus = (x_obs - x_mod)**2 + (y_obs - y_mod)**2
    return np.sum(residus)


# ---------------------------------------------------------------------------
# Optimisation avec contraintes
# ---------------------------------------------------------------------------
def identifier_parametres(t, x_obs, y_obs,
                          v0_init=10.0, theta_init=0.7,
                          x0_init=None, y0_init=None):
    """
    Resout le probleme inverse par minimisation sous contraintes.

    Parameters
    ----------
    t, x_obs, y_obs : np.ndarray
        Observations au format standard (t, x, y).
    v0_init, theta_init : float
        Estimations initiales.
    x0_init, y0_init : float, optional
        Estimations initiales de la position de depart.
        Par defaut, prend le premier point observe.

    Returns
    -------
    dict
        Resultats : v0, theta (deg et rad), x0, y0, cout final.
    """
    if x0_init is None:
        x0_init = x_obs[0]
    if y0_init is None:
        y0_init = y_obs[0]

    p0 = [v0_init, theta_init, x0_init, y0_init]

    bornes = [
        (0.1, None),         # v0 > 0
        (0.0, np.pi / 2),    # theta in [0, pi/2]
        (None, None),        # x0 libre
        (None, None),        # y0 libre
    ]

    resultat = minimize(
        fonction_cout, p0,
        args=(t, x_obs, y_obs),
        method='L-BFGS-B',
        bounds=bornes,
        options={'maxiter': 10000, 'ftol': 1e-15}
    )

    v0_opt, theta_opt, x0_opt, y0_opt = resultat.x

    return {
        'v0': v0_opt,
        'theta_rad': theta_opt,
        'theta_deg': np.degrees(theta_opt),
        'x0': x0_opt,
        'y0': y0_opt,
        'cout': resultat.fun,
        'succes': resultat.success,
        'message': resultat.message,
        'resultat_brut': resultat
    }





# ---------------------------------------------------------------------------
# Chargement depuis CSV (format commun t,x,y)
# ---------------------------------------------------------------------------
def charger_csv(chemin):
    """Charge un fichier CSV au format t,x,y."""
    data = np.loadtxt(chemin, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------
# Visualisation des resultats
# ---------------------------------------------------------------------------
def tracer_resultats(t, x_obs, y_obs, res, params_vrais=None):
    """Trace les observations, le modele identifie et optionnellement le modele vrai."""
    x_fit, y_fit = modele_trajectoire(t, res['v0'], res['theta_rad'],
                                       res['x0'], res['y0'])

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(x_obs, y_obs, c='r', s=20, alpha=0.6, label='Observations bruitees')
    ax.plot(x_fit, y_fit, 'b-', linewidth=2,
            label=f'Modele identifie (v0={res["v0"]:.2f}, '
                  f'theta={res["theta_deg"]:.2f} deg)')

    if params_vrais is not None:
        x_vrai, y_vrai = modele_trajectoire(t, params_vrais['v0'],
                                             params_vrais['theta_rad'])
        ax.plot(x_vrai, y_vrai, 'g--', linewidth=1.5,
                label=f'Verite (v0={params_vrais["v0"]:.2f}, '
                      f'theta={params_vrais["theta_deg"]:.2f} deg)')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Identification des parametres balistiques')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('resultats.png', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    CSV_PATH = "observations.csv"
    SCRIPT_SIMULATION = "simulation_trajectoire.py"

    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} introuvable, lancement de {SCRIPT_SIMULATION}...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chemin_script = os.path.join(script_dir, SCRIPT_SIMULATION)
        if not os.path.exists(chemin_script):
            sys.exit(f"Erreur : {SCRIPT_SIMULATION} introuvable dans {script_dir}")
        subprocess.run([sys.executable, chemin_script], check=True)

    print("Chargement des observations")
    t, x_obs, y_obs = charger_csv(CSV_PATH)

    # -- Identification -----------------------------------------------------
    res = identifier_parametres(t, x_obs, y_obs)

    print(f"\n--- Resultats de l'optimisation ---")
    print(f"v0    = {res['v0']:.4f} m/s")
    print(f"theta = {res['theta_deg']:.4f} deg  ({res['theta_rad']:.6f} rad)")
    print(f"x0    = {res['x0']:.4f} m")
    print(f"y0    = {res['y0']:.4f} m")
    print(f"Cout  = {res['cout']:.6e}")
    print(f"Convergence : {res['succes']} ({res['message']})")

    # -- Comparaison avec la trajectoire theorique si disponible -------------
    params_vrais = None
    if os.path.exists("trajectoire_theorique.csv"):
        t_th, x_th, y_th = charger_csv("trajectoire_theorique.csv")
        v0_sim = x_th[1] / (t_th[1] * np.cos(np.arctan2(
            y_th[1] + 0.5 * G * t_th[1]**2, x_th[1])))
        theta_sim = np.arctan2(y_th[1] + 0.5 * G * t_th[1]**2, x_th[1])
        params_vrais = {
            'v0': v0_sim,
            'theta_rad': theta_sim,
            'theta_deg': np.degrees(theta_sim)
        }
        print(f"\n--- Comparaison avec les vraies valeurs ---")
        print(f"v0    : vrai={params_vrais['v0']:.4f}  "
              f"estime={res['v0']:.4f}  "
              f"erreur={abs(res['v0'] - params_vrais['v0']):.4f} m/s")
        print(f"theta : vrai={params_vrais['theta_deg']:.4f}  "
              f"estime={res['theta_deg']:.4f}  "
              f"erreur={abs(res['theta_deg'] - params_vrais['theta_deg']):.4f} deg")

    # -- Graphique ----------------------------------------------------------
    tracer_resultats(t, x_obs, y_obs, res, params_vrais)
