"""
Analyse de sensibilite et regularisation de Tikhonov.
Evalue la variance des parametres estimes (v0, theta, gamma) en fonction
du niveau de bruit, et compare l'optimisation avec et sans regularisation.
"""

import os
import sys
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint

G = 9.81


# ---------------------------------------------------------------------------
# Modele ODE (identique a methode_frottements.py)
# ---------------------------------------------------------------------------
class DynamiqueProjectile(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, t, state):
        vx, vy = state[2], state[3]
        return torch.stack([vx, vy, -self.gamma * vx, -G - self.gamma * vy])


def integrer(params, t_tensor):
    v0, theta = params['v0'], params['theta']
    x0, y0 = params['x0'], params['y0']
    gamma = params['gamma']

    vx0 = v0 * torch.cos(theta)
    vy0 = v0 * torch.sin(theta)
    etat = torch.stack([x0, y0, vx0, vy0])

    sol = odeint(DynamiqueProjectile(gamma), etat, t_tensor, method='rk4')
    return sol[:, 0], sol[:, 1]


# ---------------------------------------------------------------------------
# Fonctions de cout
# ---------------------------------------------------------------------------
def cout_brut(params, t_tensor, x_obs, y_obs):
    x_p, y_p = integrer(params, t_tensor)
    return torch.sum((x_obs - x_p)**2 + (y_obs - y_p)**2)


def cout_tikhonov(params, t_tensor, x_obs, y_obs, alpha, p_ref):
    """Cout moindres carres + regularisation de Tikhonov (Ridge)."""
    J = cout_brut(params, t_tensor, x_obs, y_obs)
    reg = alpha * (
        (params['v0'] - p_ref['v0'])**2 +
        (params['theta'] - p_ref['theta'])**2 +
        (params['gamma'] - p_ref['gamma'])**2
    )
    return J + reg


# ---------------------------------------------------------------------------
# Optimisation (projection pour gamma >= 0)
# ---------------------------------------------------------------------------
def _creer_params(v0_init=12.0, theta_init=0.75, gamma_init=0.2):
    return {
        'v0': torch.tensor(v0_init, dtype=torch.float64, requires_grad=True),
        'theta': torch.tensor(theta_init, dtype=torch.float64, requires_grad=True),
        'x0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'y0': torch.tensor(0.0, dtype=torch.float64, requires_grad=True),
        'gamma': torch.tensor(gamma_init, dtype=torch.float64, requires_grad=True),
    }


def optimiser(t_tensor, x_obs, y_obs, n_iter=1500, lr=0.01,
              alpha=0.0, p_ref=None):
    """
    Optimise les parametres par descente de gradient projetee.
    Si alpha > 0, applique la regularisation de Tikhonov.
    """
    params = _creer_params()
    optimizer = torch.optim.Adam(params.values(), lr=lr)

    if p_ref is None:
        p_ref = {'v0': torch.tensor(12.0, dtype=torch.float64),
                 'theta': torch.tensor(0.75, dtype=torch.float64),
                 'gamma': torch.tensor(0.2, dtype=torch.float64)}

    for _ in range(n_iter):
        optimizer.zero_grad()
        if alpha > 0:
            loss = cout_tikhonov(params, t_tensor, x_obs, y_obs, alpha, p_ref)
        else:
            loss = cout_brut(params, t_tensor, x_obs, y_obs)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            params['gamma'].clamp_(min=0.0)

    return {
        'v0': params['v0'].item(),
        'theta_deg': np.degrees(params['theta'].item()),
        'gamma': params['gamma'].item(),
    }


# ---------------------------------------------------------------------------
# Generation de donnees avec frottement (via scipy, pas de dependance torch)
# ---------------------------------------------------------------------------
def generer_donnees(v0, theta, gamma, sigma, num_points=80):
    from scipy.integrate import solve_ivp

    def edo(t, s):
        return [s[2], s[3], -gamma * s[2], -G - gamma * s[3]]

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    t_vol = 2 * v0 * np.sin(theta) / G
    t_max = t_vol * 1.2
    t_eval = np.linspace(0, t_max, num_points)
    sol = solve_ivp(edo, [0, t_max], [0, 0, vx0, vy0],
                    t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)

    t, x, y = sol.t, sol.y[0], sol.y[1]
    masque = y >= 0
    t, x, y = t[masque], x[masque], y[masque]

    x_obs = x + np.random.normal(0, sigma, size=x.shape)
    y_obs = y + np.random.normal(0, sigma, size=y.shape)
    return t, x_obs, y_obs


# ---------------------------------------------------------------------------
# Analyse de sensibilite
# ---------------------------------------------------------------------------
import concurrent.futures

def run_single_experiment(args):
    """
    Fonction executee par chaque coeur du processeur (worker).
    """
    sigma, v0_vrai, theta_vrai, gamma_vrai, n_iter, lr, alpha_tikhonov = args
    
    # 1. Generation des donnees
    t_np, x_np, y_np = generer_donnees(v0_vrai, theta_vrai, gamma_vrai, sigma)
    t_t = torch.tensor(t_np, dtype=torch.float64)
    x_t = torch.tensor(x_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)

    p_ref = {
        'v0': torch.tensor(v0_vrai, dtype=torch.float64),
        'theta': torch.tensor(theta_vrai, dtype=torch.float64),
        'gamma': torch.tensor(gamma_vrai, dtype=torch.float64),
    }

    # 2. Optimisation Brut
    res_brut = optimiser(t_t, x_t, y_t, n_iter=n_iter, lr=lr, alpha=0.0)
    
    # 3. Optimisation Tikhonov
    res_tikh = optimiser(t_t, x_t, y_t, n_iter=n_iter, lr=lr, 
                         alpha=alpha_tikhonov, p_ref=p_ref)
                         
    return sigma, res_brut, res_tikh


def analyse_sensibilite(v0_vrai, theta_vrai, gamma_vrai,
                        sigmas, n_essais=20, n_iter=1500, lr=0.01,
                        alpha_tikhonov=5.0):
    """
    Analyse de sensibilite parallelisee sur tous les coeurs du CPU.
    """
    resultats = {'brut': {s: [] for s in sigmas},
                 'tikhonov': {s: [] for s in sigmas}}

    # Preparation de la liste des taches a distribuer
    taches = []
    for sigma in sigmas:
        for _ in range(n_essais):
            taches.append((sigma, v0_vrai, theta_vrai, gamma_vrai, 
                           n_iter, lr, alpha_tikhonov))

    total = len(taches)
    print(f"Lancement de {total} taches")

    # Utilisation du Pool de processus
    # max_workers=None utilise automatiquement tous les cœurs disponibles
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # map() distribue les taches et recupere les resultats
        for i, (sigma, res_brut, res_tikh) in enumerate(executor.map(run_single_experiment, taches), 1):
            resultats['brut'][sigma].append(res_brut)
            resultats['tikhonov'][sigma].append(res_tikh)
            
            # Affichage de la progression
            print(f"\r  Progression : {i}/{total} taches terminees", end="", flush=True)

    print("\nTermine !")
    return resultats


def calculer_stats(resultats, sigmas):
    """Calcule moyenne et ecart-type pour chaque sigma et chaque methode."""
    stats = {}
    for methode in ['brut', 'tikhonov']:
        stats[methode] = {'sigma': [], 'v0_moy': [], 'v0_std': [],
                          'theta_moy': [], 'theta_std': [],
                          'gamma_moy': [], 'gamma_std': []}
        for s in sigmas:
            vals = resultats[methode][s]
            v0s = [r['v0'] for r in vals]
            thetas = [r['theta_deg'] for r in vals]
            gammas = [r['gamma'] for r in vals]

            stats[methode]['sigma'].append(s)
            stats[methode]['v0_moy'].append(np.mean(v0s))
            stats[methode]['v0_std'].append(np.std(v0s))
            stats[methode]['theta_moy'].append(np.mean(thetas))
            stats[methode]['theta_std'].append(np.std(thetas))
            stats[methode]['gamma_moy'].append(np.mean(gammas))
            stats[methode]['gamma_std'].append(np.std(gammas))
    return stats


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def tracer_sensibilite(stats, v0_vrai, theta_vrai_deg, gamma_vrai):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    params_info = [
        ('v0', 'v0 (m/s)', v0_vrai),
        ('theta', 'theta (deg)', theta_vrai_deg),
        ('gamma', 'gamma (s^-1)', gamma_vrai),
    ]

    for ax, (key, label, vrai) in zip(axes, params_info):
        for methode, couleur, style in [('brut', '#e74c3c', 'o-'),
                                         ('tikhonov', '#2ecc71', 's--')]:
            s = stats[methode]
            moy = np.array(s[f'{key}_moy'])
            std = np.array(s[f'{key}_std'])
            sigma = np.array(s['sigma'])

            ax.errorbar(sigma, moy, yerr=std, fmt=style, color=couleur,
                        capsize=4, linewidth=1.5, markersize=5,
                        label=f'{methode.capitalize()}')

        ax.axhline(vrai, color='gray', linestyle=':', linewidth=1,
                   label=f'Vrai ({vrai})')
        ax.set_xlabel('Sigma bruit (m)')
        ax.set_ylabel(label)
        ax.set_title(f'Sensibilite de {label}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sensibilite et regularisation de Tikhonov',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sensibilite.png', dpi=150)
    plt.show()


def tracer_variance(stats):
    fig, ax = plt.subplots(figsize=(10, 5))

    for methode, couleur, style in [('brut', '#e74c3c', 'o-'),
                                     ('tikhonov', '#2ecc71', 's--')]:
        s = stats[methode]
        sigma = np.array(s['sigma'])
        var_totale = (np.array(s['v0_std'])**2 +
                      np.array(s['theta_std'])**2 +
                      np.array(s['gamma_std'])**2)
        ax.plot(sigma, var_totale, style, color=couleur, linewidth=1.5,
                markersize=5, label=f'{methode.capitalize()}')

    ax.set_xlabel('Sigma bruit (m)')
    ax.set_ylabel('Variance totale cumulee')
    ax.set_title('Variance totale vs niveau de bruit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig('variance.png', dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Point d'entree
# ---------------------------------------------------------------------------
if __name__ == '__main__':

    V0_VRAI = 15.0
    THETA_VRAI = np.pi / 4
    GAMMA_VRAI = 0.5

    SIGMAS = [0.01, 0.03, 0.05, 0.08, 0.12, 0.20]
    N_ESSAIS = 15
    N_ITER = 1500
    LR = 0.01
    ALPHA_TIKHONOV = 5.0

    print(f"Parametres vrais : v0={V0_VRAI}, theta={np.degrees(THETA_VRAI):.1f} deg, "
          f"gamma={GAMMA_VRAI}")
    print(f"Niveaux de bruit : {SIGMAS}")
    print(f"Essais par sigma : {N_ESSAIS}")
    print(f"Alpha Tikhonov : {ALPHA_TIKHONOV}")
    print(f"Total optimisations : {len(SIGMAS) * N_ESSAIS * 2}\n")

    resultats = analyse_sensibilite(
        V0_VRAI, THETA_VRAI, GAMMA_VRAI,
        SIGMAS, n_essais=N_ESSAIS, n_iter=N_ITER, lr=LR,
        alpha_tikhonov=ALPHA_TIKHONOV
    )

    stats = calculer_stats(resultats, SIGMAS)

    # -- Tableau recapitulatif ----------------------------------------------
    print(f"\n{'sigma':>6} | {'--- Sans regul ---':^30} | {'--- Tikhonov ---':^30}")
    print(f"{'':>6} | {'v0':>8} {'theta':>8} {'gamma':>8}  | {'v0':>8} {'theta':>8} {'gamma':>8}")
    print("-" * 75)
    for i, s in enumerate(SIGMAS):
        sb = stats['brut']
        st = stats['tikhonov']
        print(f"{s:>6.3f} | "
              f"{sb['v0_moy'][i]:>6.2f}+/-{sb['v0_std'][i]:<5.3f} "
              f"{sb['theta_moy'][i]:>6.2f}+/-{sb['theta_std'][i]:<5.3f} "
              f"{sb['gamma_moy'][i]:>6.3f}+/-{sb['gamma_std'][i]:<5.4f} | "
              f"{st['v0_moy'][i]:>6.2f}+/-{st['v0_std'][i]:<5.3f} "
              f"{st['theta_moy'][i]:>6.2f}+/-{st['theta_std'][i]:<5.3f} "
              f"{st['gamma_moy'][i]:>6.3f}+/-{st['gamma_std'][i]:<5.4f}")

    # -- Graphiques ---------------------------------------------------------
    tracer_sensibilite(stats, V0_VRAI, np.degrees(THETA_VRAI), GAMMA_VRAI)
    tracer_variance(stats)
