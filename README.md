# AdvVoter
This repository contains data and code used in our studies: *[Adversarial attacks on voter model dynamics in complex networks](https://doi.org/10.1103/PhysRevE.106.014301)* and *[Mitigation of adversarial attacks on voter model dynamics by network heterogeneity](https://doi.org/10.1088/2632-072X/acd296)*.

## Terms of use

MIT licensed. Happy if you cite our papers when using the codes:

* Chiyomaru K and Takemoto K (2022) **Adversarial attacks on voter model dynamics in complex networks**. Phys. Rev. E 106, 014301. doi:10.1103/PhysRevE.106.014301.
* Chiyomaru K and Takemoto K (2023) **Mitigation of adversarial attacks on voter model dynamics by network heterogeneity**. JPhys Complexity 4, 025009. doi:10.1088/2632-072X/acd296.

## Usage
### Requirements
Python 3.9
```
pip install -r requirements.txt
```

### Vote model dynamics in model networks
e.g., in Erdos-Renyi networks
```
python run.py --network ER
```

Note that $N=t_{\max}=400$, $\langle k \rangle = 6$, $\rho_{\mathrm{init}}=0.8$, and $\epsilon=0.01$ are in default configuration (see `run.py` for details).

$\rho$ distibutions for no perturbation (at $\epsilon=0$), adversarial attacks, and random attacks are displayed.
Each $\rho$ distribution is obtained from 100 realizations in default configuration (see `run.py` for details).

![rho_distributions](rho_distribution.png)

The following network models can be also considered:
* Watts-Strogatz model (`--network WS`)
* Barabasi-Albert model (`--network BA`)
* Goh-Kahng-Kim model (`--network GKK`)
* Holme-Kim model (`--network HK`)

For a negative `tmax` value, the voter model dynamics are terminated when a steady state (consensus) is reached.

e.g., in Erdos-Renyi networks
```
python run.py --network ER --tmax -1
```

### For maximally disassortative and assortative model networks
e.g., for maximally disassortative ER networks
```
python run.py --network ER --correlation disassort
```

for maximally assortative GKK networks with the degree exponent of 3.0
```
python run.py --network GKK --gamma 3.0 --correlation assort
```

### Vote model dynamics in real-world networks
Note that computation is time-consuming.

e.g., Facebook
```
python run.py --network facebook_combined
```

The following real-world networks are also available.
* Advogato (`--network soc-advogato`)
* AnyBeat (`--network soc-anybeat`)
* HAMSTERster (`--network soc-hamsterster`)

### Vote model dynamics in degree-preserving random networks for real-world networks
Note that computation is time-consuming.

e.g., Facebook
```
python run_randomlized.py --network facebook_combined
```

Use `run.py` when considering ER networks as null model networks.
