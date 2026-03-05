"""
Extension - Modele balistique avec trainee quadratique.
Force de trainee proportionnelle au carre de la vitesse :
    F_drag = -c * |v| * v   =>  parametre gamma_q = c/m
Compare la reconstruction avec le modele lineaire et le modele quadratique.
"""

import os
import sys
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

G = 9.81


# ===========================================================================
# EDO : modele lineaire (rappel)
# ===========================================================================
class DragLineaire(torch.nn.Module):
    """dvx/dt = -gamma*vx, dvy/dt = -g - gamma*vy"""
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, t, state):
        vx, vy = state[2], state[3]
        return torch.stack([vx, vy,
                            -self.gamma * vx,
                            -G - self.gamma * vy])


# ===========================================================================
# EDO : modele quadratique
# ===========================================================================
class DragQuadratique(torch.nn.Module):
    """
    dvx/dt = -gamma_q * |v| * vx
    dvy/dt = -g - gamma_q * |v| * vy
    ou |v| = sqrt(vx^2 + vy^2)
    """
    def __init__(self, gamma_q):
        super().__init__()
        self.gamma_q = gamma_q

    def forward(self, t, state):
        vx, vy = state[2], state[3]
        vitesse = torch.sqrt(vx**2 + vy**2 + 1e-12)
        return torch.stack([vx, vy,
                            -self.gamma_q * vitesse * vx,
                            -G - self.gamma_q * vitesse * vy])


# ===========================================================================
# Integration generique
# ===========================================================================
def integrer(params, t_tensor, modele_class):
    v0 = params['v0']
    theta = params['theta']
    x0 = params['x0']
    y0 = params['y0']
    gamma = params['gamma']

    vx0 = v0 * torch.cos(theta)
    vy0 = v0 * torch.sin(theta)
    etat = torch.stack([x0, y0, vx0, vy0])

    dynamique = modele_class(gamma)
    sol = odeint(dynamique, etat, t_tensor, method='rk4')
    return sol[:, 0], sol[:, 1]


def cout(params, t_tensor, x_obs, y_obs, modele_class):
    x_p, y_p = integrer(params, t_tensor, modele_class)
    return torch.sum((x_obs - x_p)**2 + (y_obs - y_p)**2)


# ===========================================================================
# Optimisation (projection)
# ===========================================================================
def _creer_params(v0_init=35.0, theta_init=0.75, gamma_init=0.2):
    return {
        'v0': torch.tensor(v0_init, dtype=torch.float64, requires_grad=True),
        'theta': torch.tensor(theta_init, dtype=torch.float64, requires_grad=True),
        'x0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'y0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'gamma': torch.tensor(gamma_init, dtype=torch.float64, requires_grad=True),
    }


def optimiser(t_tensor, x_obs, y_obs, modele_class, nom,
              n_iter=2000, lr=0.01):
    params = _creer_params()
    optimizer = torch.optim.Adam(params.values(), lr=lr)
    historique = []

    for _ in range(n_iter):
        optimizer.zero_grad()
        J = cout(params, t_tensor, x_obs, y_obs, modele_class)
        J.backward()
        optimizer.step()
        with torch.no_grad():
            params['gamma'].clamp_(min=0.0)
        historique.append(J.item())

    return {
        'nom': nom,
        'v0': params['v0'].item(),
        'theta_rad': params['theta'].item(),
        'theta_deg': np.degrees(params['theta'].item()),
        'x0': params['x0'].item(),
        'y0': params['y0'].item(),
        'gamma': params['gamma'].item(),
        'cout_final': historique[-1],
        'historique': historique,
        'modele_class': modele_class,
    }


# ===========================================================================
# Chargement CSV
# ===========================================================================
def charger_csv(chemin):
    data = np.loadtxt(chemin, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2]


# ===========================================================================
# Visualisation
# ===========================================================================
def tracer_comparaison(t_np, x_obs_np, y_obs_np, res_lin, res_quad):
    """Superpose les trajectoires reconstruites par les deux modeles."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(x_obs_np, y_obs_np, c='gray', s=15, alpha=0.5,
               label='Observations', zorder=1)

    t_dense = torch.linspace(0, float(t_np[-1]), 300, dtype=torch.float64)

    for res, couleur, style in [(res_lin, '#e74c3c', '-'),
                                 (res_quad, '#2ecc71', '--')]:
        params = {
            'v0': torch.tensor(res['v0'], dtype=torch.float64),
            'theta': torch.tensor(res['theta_rad'], dtype=torch.float64),
            'x0': torch.tensor(res['x0'], dtype=torch.float64),
            'y0': torch.tensor(res['y0'], dtype=torch.float64),
            'gamma': torch.tensor(res['gamma'], dtype=torch.float64),
        }
        with torch.no_grad():
            x_fit, y_fit = integrer(params, t_dense, res['modele_class'])
        ax.plot(x_fit.numpy(), y_fit.numpy(), color=couleur, linewidth=2,
                linestyle=style, zorder=2,
                label=f'{res["nom"]} (gamma={res["gamma"]:.4f})')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Comparaison : trainee lineaire vs quadratique')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('drag_comparaison.png', dpi=150)
    plt.show()


def tracer_convergence(res_lin, res_quad):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(res_lin['historique'], color='#e74c3c', linewidth=1.5,
            label=f'Lineaire (final={res_lin["cout_final"]:.4f})')
    ax.plot(res_quad['historique'], color='#2ecc71', linewidth=1.5,
            label=f'Quadratique (final={res_quad["cout_final"]:.4f})')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cout J')
    ax.set_title('Convergence : lineaire vs quadratique')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('drag_convergence.png', dpi=150)
    plt.show()


# ===========================================================================
# Point d'entree
# ===========================================================================
if __name__ == '__main__':

    CSV_PATH = "observations_quad.csv"
    SCRIPT_SIMULATION = "simulation_trajectoire_quad.py"
    N_ITER = 2000
    LR = 0.01

    if not os.path.exists(CSV_PATH):
        print(f"{CSV_PATH} introuvable, lancement de {SCRIPT_SIMULATION}...")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        chemin_script = os.path.join(script_dir, SCRIPT_SIMULATION)
        if not os.path.exists(chemin_script):
            sys.exit(f"Erreur : {SCRIPT_SIMULATION} introuvable dans {script_dir}")
        subprocess.run([sys.executable, chemin_script], check=True)

    print("Chargement des observations")
    t_np, x_obs_np, y_obs_np = charger_csv(CSV_PATH)

    t_t = torch.tensor(t_np, dtype=torch.float64)
    x_t = torch.tensor(x_obs_np, dtype=torch.float64)
    y_t = torch.tensor(y_obs_np, dtype=torch.float64)

    # -- Reconstruction avec le modele lineaire -----------------------------
    print("\n[1/2] Reconstruction par modele lineaire...")
    res_lin = optimiser(t_t, x_t, y_t, DragLineaire, "Lineaire",
                        n_iter=N_ITER, lr=LR)

    # -- Reconstruction avec le modele quadratique --------------------------
    print("[2/2] Reconstruction par modele quadratique...")
    res_quad = optimiser(t_t, x_t, y_t, DragQuadratique, "Quadratique",
                         n_iter=N_ITER, lr=LR)

    # -- Tableau comparatif -------------------------------------------------
    print("\n" + "=" * 70)
    print(f"{'Modele':<14} {'v0':>8} {'theta':>8} {'gamma':>8} {'Cout':>12}")
    print("-" * 70)
    for r in [res_lin, res_quad]:
        print(f"{r['nom']:<14} {r['v0']:>8.3f} {r['theta_deg']:>8.3f} "
              f"{r['gamma']:>8.5f} {r['cout_final']:>12.4e}")
    print("=" * 70)

    # -- Graphiques ---------------------------------------------------------
    tracer_comparaison(t_np, x_obs_np, y_obs_np, res_lin, res_quad)
    tracer_convergence(res_lin, res_quad)
