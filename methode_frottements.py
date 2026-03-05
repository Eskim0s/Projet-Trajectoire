"""
Modele balistique avec trainee lineaire.
EDO : m*x'' = -k*x', m*y'' = -m*g - k*y'  =>  gamma = k/m
Resolution par torchdiffeq (differentiation automatique).
Comparaison de trois methodes de contrainte gamma >= 0 :
penalisation, projection, barriere.
"""

import os
import sys
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt

from torchdiffeq import odeint


G = 9.81
DEVICE = torch.device('cpu')


# ---------------------------------------------------------------------------
# EDO du projectile avec trainee lineaire
# ---------------------------------------------------------------------------
class DynamiqueProjectile(torch.nn.Module):
    """
    Etat : [x, y, vx, vy]
    Equations :
        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = -gamma * vx
        dvy/dt = -g - gamma * vy
    """
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, t, state):
        vx = state[2]
        vy = state[3]
        ax = -self.gamma * vx
        ay = -G - self.gamma * vy
        return torch.stack([vx, vy, ax, ay])


# ---------------------------------------------------------------------------
# Resolution de l'EDO et calcul du cout
# ---------------------------------------------------------------------------
def integrer_trajectoire(params, t_tensor):
    """
    Integre l'EDO pour les parametres donnes.

    Parameters
    ----------
    params : dict de tenseurs torch
        v0, theta, x0, y0, gamma
    t_tensor : torch.Tensor

    Returns
    -------
    x_pred, y_pred : torch.Tensor
    """
    v0 = params['v0']
    theta = params['theta']
    x0 = params['x0']
    y0 = params['y0']
    gamma = params['gamma']

    vx0 = v0 * torch.cos(theta)
    vy0 = v0 * torch.sin(theta)
    etat_init = torch.stack([x0, y0, vx0, vy0])

    dynamique = DynamiqueProjectile(gamma)
    solution = odeint(dynamique, etat_init, t_tensor, method='rk4')

    return solution[:, 0], solution[:, 1]


def cout_moindres_carres(params, t_tensor, x_obs, y_obs):
    x_pred, y_pred = integrer_trajectoire(params, t_tensor)
    return torch.sum((x_obs - x_pred)**2 + (y_obs - y_pred)**2)


# ---------------------------------------------------------------------------
# Trois methodes de gestion de la contrainte gamma >= 0
# ---------------------------------------------------------------------------

def optimiser_penalisation(t_tensor, x_obs, y_obs, n_iter=2000, lr=0.01,
                           mu=100.0):
    """
    Methode de penalisation : J_pen = J + mu * max(0, -gamma)^2
    """
    params = _creer_params()
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    historique = []

    for i in range(n_iter):
        optimizer.zero_grad()
        J = cout_moindres_carres(params, t_tensor, x_obs, y_obs)
        penalite = mu * torch.relu(-params['gamma'])**2
        loss = J + penalite
        loss.backward()
        optimizer.step()
        historique.append(J.item())

    return _extraire_resultats(params, historique, "Penalisation")


def optimiser_projection(t_tensor, x_obs, y_obs, n_iter=2000, lr=0.01):
    """
    Methode de projection : apres chaque pas de gradient, gamma = max(0, gamma)
    """
    params = _creer_params()
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    historique = []

    for i in range(n_iter):
        optimizer.zero_grad()
        J = cout_moindres_carres(params, t_tensor, x_obs, y_obs)
        J.backward()
        optimizer.step()
        with torch.no_grad():
            params['gamma'].clamp_(min=0.0)
        historique.append(J.item())

    return _extraire_resultats(params, historique, "Projection")


def optimiser_barriere(t_tensor, x_obs, y_obs, n_iter=2000, lr=0.01,
                       mu_init=1.0, mu_decay=0.995):
    """
    Methode de barriere logarithmique : J_bar = J - mu * ln(gamma)
    mu decroit progressivement pour resserrer la contrainte.
    """
    params = _creer_params(gamma_init=0.5)
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    historique = []
    mu = mu_init

    for i in range(n_iter):
        optimizer.zero_grad()
        J = cout_moindres_carres(params, t_tensor, x_obs, y_obs)

        gamma_val = params['gamma']
        barriere = -mu * torch.log(torch.clamp(params['gamma'], min=1e-8))

        loss = J + barriere
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            params['gamma'].clamp_(min=1e-8)

        mu *= mu_decay
        historique.append(J.item())

    return _extraire_resultats(params, historique, "Barriere")


# ---------------------------------------------------------------------------
# Utilitaires internes
# ---------------------------------------------------------------------------
def _creer_params(v0_init=12.0, theta_init=0.75, gamma_init=0.2):
    return {
        'v0': torch.tensor(v0_init, dtype=torch.float64, requires_grad=True),
        'theta': torch.tensor(theta_init, dtype=torch.float64, requires_grad=True),
        'x0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'y0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'gamma': torch.tensor(gamma_init, dtype=torch.float64, requires_grad=True),
    }


def _extraire_resultats(params, historique, nom):
    return {
        'nom': nom,
        'v0': params['v0'].item(),
        'theta_rad': params['theta'].item(),
        'theta_deg': np.degrees(params['theta'].item()),
        'x0': params['x0'].item(),
        'y0': params['y0'].item(),
        'gamma': params['gamma'].item(),
        'cout_final': historique[-1] if historique else float('inf'),
        'historique': historique,
    }


# ---------------------------------------------------------------------------
# Chargement CSV (format commun t,x,y)
# ---------------------------------------------------------------------------
def charger_csv(chemin):
    data = np.loadtxt(chemin, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def tracer_convergence(resultats_list):
    """Trace les courbes de convergence des trois methodes."""
    fig, ax = plt.subplots(figsize=(10, 5))
    couleurs = ['#e74c3c', '#2ecc71', '#3498db']

    for res, c in zip(resultats_list, couleurs):
        hist = res['historique']
        ax.plot(hist, label=f'{res["nom"]} (final={res["cout_final"]:.4f})',
                color=c, linewidth=1.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cout J (moindres carres)')
    ax.set_title('Convergence des trois methodes de contrainte')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    plt.show()


def tracer_trajectoires(t_np, x_obs_np, y_obs_np, resultats_list):
    """Trace les trajectoires identifiees par chaque methode."""
    fig, ax = plt.subplots(figsize=(10, 5))
    couleurs = ['#e74c3c', '#2ecc71', '#3498db']

    ax.scatter(x_obs_np, y_obs_np, c='gray', s=15, alpha=0.5,
               label='Observations', zorder=1)

    t_dense = torch.linspace(0, float(t_np[-1]), 200, dtype=torch.float64)

    for res, c in zip(resultats_list, couleurs):
        params = {
            'v0': torch.tensor(res['v0'], dtype=torch.float64),
            'theta': torch.tensor(res['theta_rad'], dtype=torch.float64),
            'x0': torch.tensor(res['x0'], dtype=torch.float64),
            'y0': torch.tensor(res['y0'], dtype=torch.float64),
            'gamma': torch.tensor(res['gamma'], dtype=torch.float64),
        }
        with torch.no_grad():
            x_fit, y_fit = integrer_trajectoire(params, t_dense)
        ax.plot(x_fit.numpy(), y_fit.numpy(), color=c, linewidth=2,
                label=f'{res["nom"]} (gamma={res["gamma"]:.4f})', zorder=2)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Trajectoires identifiees avec frottement')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('trajectoires.png', dpi=150)
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
    t_np, x_obs_np, y_obs_np = charger_csv(CSV_PATH)

    t_tensor = torch.tensor(t_np, dtype=torch.float64)
    x_obs = torch.tensor(x_obs_np, dtype=torch.float64)
    y_obs = torch.tensor(y_obs_np, dtype=torch.float64)

    N_ITER = 2000
    LR = 0.01

    # -- Methode 1 : Penalisation -------------------------------------------
    print("\n[1/3] Optimisation par penalisation...")
    res_pen = optimiser_penalisation(t_tensor, x_obs, y_obs,
                                     n_iter=N_ITER, lr=LR)

    # -- Methode 2 : Projection ---------------------------------------------
    print("[2/3] Optimisation par projection...")
    res_proj = optimiser_projection(t_tensor, x_obs, y_obs,
                                    n_iter=N_ITER, lr=LR)

    # -- Methode 3 : Barriere -----------------------------------------------
    print("[3/3] Optimisation par barriere logarithmique...")
    res_bar = optimiser_barriere(t_tensor, x_obs, y_obs,
                                 n_iter=N_ITER, lr=LR)

    resultats = [res_pen, res_proj, res_bar]

    # -- Affichage des resultats --------------------------------------------
    print("\n" + "=" * 65)
    print(f"{'Methode':<16} {'v0':>8} {'theta':>8} {'gamma':>8} {'Cout':>12}")
    print("-" * 65)
    for r in resultats:
        print(f"{r['nom']:<16} {r['v0']:>8.3f} {r['theta_deg']:>8.3f} "
              f"{r['gamma']:>8.5f} {r['cout_final']:>12.4e}")
    print("=" * 65)

    # -- Graphiques ---------------------------------------------------------
    tracer_convergence(resultats)
    tracer_trajectoires(t_np, x_obs_np, y_obs_np, resultats)
