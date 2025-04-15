# Active Learning  - Plasma Surrogate Model



Introduction 



Problem Formulation and Previous Work 



We assume that we don't have a previous collected dataset and we want to create a surrogate model as sample efficient as possible.

Furthermore, the dataset we collect help the subsequent offline models 



## Mathematical Formalization in an RL Framework 



The problem of choosing the next samples (actions) in an active learning context can be framed as a sequential decision-making process as follows:

#### State Space ($\mathcal{S}$)

The state $s_t$ at time $t$ represents the current "knowledge" about the the input space as encoded by the uncertainty function.
$$
s_t = \{u_t(x_1), u_t(x_2), \dots, u_t(x_n), \phi_t  \}
$$
where $\{x_1, \dots, x_n \}$ are representative points and $\phi_t$ corresponds to the global uncertainty (e.g. average of all representative points)

#### Action Space ($\mathcal{A}$)

An action $a_t$ involves selecting a subset of new sample points $\{x_{a_1}, x_{a_2}, \dots, x_{a_k} \}$ from a candidate pool. These are the new points to be queried (or simulated) in order to gather additional information data.
$$
a_t \subset \Chi_{candidates}
$$
where constraints like ensuring coverage (e.g. maintining diversity in the set $a_t$) may be imposed.

#### Transition Dynamics ($\mathcal{P}$)

The transiton from $s_t$ to $s_{t+1}$ is governed by the acquisition of new data and subsequent update of the predictive model. The update effects the uncertainty function:
$$
s_{t+1} = f(s_t, a_t, o_t)
$$
Where $o_t$ represents the observed outcomes (e.g. simulator calls) at the chosen points $a_t$ and $f$ denotes the update of the surrogate model.

#### Reward Function ($\mathcal{R}$)

The reward $r_t$ is designed to capture the value of red



#### Objective 









Problem Seen in the probabilistic sense





Model Architecture and Pipeline 





Proof of concept with synthetic data and Results









Next Work







