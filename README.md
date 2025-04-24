# Active Learning  - Plasma Surrogate Model

José Filipe Afonso 15/04/2025



## Introduction and Problem Formulation

Creating efficient surrogate models that modulate the output of difficult-to-compute physical simulators that model the surface plasma-surface interactions is highly desirable.

So far, this approach has been widely explored from a **supervised learning** perspective. A dataset is collected using a design filling algorithm like the Latin hypercube sampling. Then, as a posterior and independent task, a model is created to predict the dataset as closely as possible (given that we define the loss criteria). In this second stage, different architectures and hyperparameters are tested and fine-tuned.

This work pretends to follow a different direction. Instead of treating the sample collection and the surrogate model developing as two independent tasks, we pretend to combine them, following a perspective of **active learning**. By taking advantage of the uncertainties and expected changes in the surrogate model, we hope to enhance its accuracy and make it more sample-efficient without changing the previously defined model architectures. Indeed, this last task has been studied in a different work, which is out of scope here. However, future development could involve applying the methods developed here to enhance *offline* models.

In a different context, the active learning approach has been successfully applied to other systems and tasks, such as Lewis et al. *A Sequential Algorithm for Training Text Classifiers*, which motivates the present study.

Additionally, it would be beneficial if the dataset we collected improved future offline models, as this could facilitate the learning process.



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

The reward $r_t$ is designed to capture the value of reducing degree of uncertainty or the increment of the expected improvement. For each point $x$ in the candidate pool  (or the list that composes a state) is computed the reward or also called the acquisition function $A(\cdot)$
$$
r_t = A(\cdot)
$$
The $A(\cdot)$ considered function will be widely explained in a following section

#### Objective 

The goal is to select a sequence of action(s) (i.e. sample points) that maximizes the expected cumlative reward over a horizon $T$:
$$
\max_{\{a_t\}} \mathbb{E} \left[\sum^{T-1}_{t=0} \gamma^t r_t \right]
$$
However, in the case of our system considering $T > 1$ implies to simulate the evolution of the ensemble, which quickly becomes intractable. As a result, we simplify the objective and adopt a myopic and greedy strategy given as follows:
$$
T = 1 ~~~~ \max_{a_t} A(a_t)
$$
So, at each step, we pick the action that maximizes $A(\cdot)$, without having in mind the long term effects. In this case, our approach is more related to a model-based RL algorithm with a one-step planning rollout 



## Problem seen in a probabilistic view

#### State - Candidate Pool - Prior Sampling

The definition of the state described previously corresponds to a belief over the input space given by the **prior** distribution $p(x)$.

We use the space the space-filling design to sample a set of candidate points:
$$
\{x_1, x_2, \dots, x_N \} \sim p(x)
$$
If we further add a Gaussian perturbation (jitter) to each sample:
$$
x'_i = x_i + \epsilon_i, ~~~~\epsilon_i \sim \mathcal{N}(0, \sigma^2_j I) \newline
x'_i \sim \overline{p}(x) = (p * \mathcal{N}(0, \sigma^2_j I)) (x)
$$
The jitter amount $\sigma_j$ can be scheduled and represents a change in the prior belief.

#### Action and Rewards - Select the samples from the candiadate pool

We observe the data $D = \{ x_i, y_i \}$ from the physical simulator, we train the ensemble of nets that provide a predictive distribution for a new input $x$:
$$
p(f(x)|x,D) = \mathcal{N}(\mu(x), \sigma^2(x)I)
$$
The acquisition function $A(x)$  allows to select the points of the candidate pool that maximize a giving score. As a result, the **posterior** for selecting candidates, given a state $s_t$, can be written as:
$$
p(x|D) \propto p(x) l(D|x) \propto p(x) A(x)
$$


## Acquisiton (Reward) function definition

The proposed acquisition function pretends select the most *informative* samples for training the neural ensemble. This function simultaneously accounts for three critical criteria: the model's uncertainty about the ouput, $U(x)$, the expected change in the model's parameters (gradient update magnitude), $G(x)$,  and the coverage of the candidate sample with respect to the current data distribution, $\rho(x)$. They are defined as follows:

- **Uncertainty measure** $U(x)$:

  This metric quantifies the ensemble's uncertainty. In our implementation, it is computed as the standard deviation of the predictions $\{ y_i\}^M_{i=1}$ provided by $M$ ensemble members:
  $$
  U(x) = \text{std}\left(\{ y_i\}^M_{i=1} \right)
  $$

- **Gradient Update sxEstimate** $G(x)$:

  This term quantifies the expected magnitude of the update of the model parameters if the point $x$ was added to the training set. For each ensemble member $i$ with parameters $\theta$ and loss function $L(\cdot,\cdot)$, let
  $$
  g_i(x, y) = \nabla_{\theta_i} L(f_i(x;\theta_i), y)
  $$
  be the gradient vector when. $x$ is associated with a target $y$.  Since $y$ is unknown at the query time, we define an expected gradient norm over the predictive distribution $p(y|x)$:
  $$
  G(x) = \frac{1}{N} \sum^M_{i=1} \int ||g_{i}(x, y) ||^2 p(y|x) dy \approx \newline 
  \approx \frac{1}{N} \sum^M_{i=1} ||g_i(x, \hat{y}_i)||^2 ~~~\text{where} ~~ \hat{y}_i\sim \mathcal{N}(\mu_i(x), \sigma^2_i(x))
  $$

- **Coverage Measure** $\rho(x)$:

  To favor samples that are representative of regions with high density, we define the measure $\rho(x)$ that corresponds to a function that increases with the representativeness of $x$:

$$
\rho(x) = 1/\min(d(x, D))
$$

To combine these heterogeneous metrics into a unified acquisition function, we first normalize each component so that they take values in the interval $[0,1]$. So, we define the normalized version $\tilde{F}(x)$ via *min-max* normalization over the set of candidates $\Chi$:
$$
\tilde{F}(x) = \frac{F(x) - \min_{x' \in \Chi} F(x')}{\max_{x' \in \Chi} F(x') - \min_{x' \in \Chi} F(x')}
$$
and so the acquisition function $A(x)$ is given by:
$$
A(x) = \alpha \tilde{U}(x) + \beta \tilde{G}(x) + \gamma \tilde{\rho}(x)
$$
with the weights satifying: $\alpha, \beta, \gamma \geq 0$ and $\alpha + \beta + \gamma = 1$

In this way the Acquisition function $A(x)$ is designed to select candidate samples that are simultaneously:

- **Uncertain**: The ensemble exhibits a high variance in its predictions for $x$;
- **Informative**: The expected gradient update (and thereby potential improvement) is high
- **Representative**: The sample $x$ lies in an underexplored region of the input space.





## Pipeline and Model Architecture

The complete pipeline is available on GitHub [https://github.com/joseAf28/PlasmaActiveLearning]. 

We use the LisbOn KInetics (LoKI) simulation tool as our physical simulator for experimental purposes. This tool intends to model non-equilibrium low-temperature plasmas (LTPs) produced from different gases or gas mixtures under a wide range of working conditions. It comprises two modules: LoKI-B, available on [https://github.com/IST-Lisbon/LoKI], and LoKI-C, which is not publicly available.

We consider a simple system that effectively corresponds to a function $F: \mathbb{R}^3 \to \mathbb{R}^3$, the function that, for a given change in the input experimental conditions to assumed default values - tuple of the changed experimental conditions - computes the equilibrium concentrations of the chemical species.

The surrogate model comprised an ensemble of neural networks with the following hyperparameter configuration:

```python
config = {
    'ensemble_size': 10,		# ensemble composed by 10 neural nets
    'input_size': 3,
    'hidden_size': 256,			# For each ensemble's neural net
    'output_size': 10,
  	'num_epochs_init': 400,	# For the initial state
    'num_epochs': 70,       # For each state
    'batch_size': 32,
    'learning_rate': 1e-2,
    'candidate_pool_size': 150, # Candidate Pool size
    'n_samples': 150,       # Initial buffer size
    'subset_frac': 0.7,     # Memory buffer fraction to mitigate catastrophic forgetting
    'n_actions': 15,        # Number of samples taken at each action
    'lambda_vec': np.array([0.12, 0.85, 0.03])  # Weights for the acquisition function: (α, β, γ)
}
```

This configuration represents a trade-off between model complexity and computational feasibility.



## Experimental Results and Comparison 

#### 

A neural network with an identical architecture and training regimen as each ensemble member aside the number of epochs (600). Its dataset is generated using traditional Latin hypercube sampling. 



Our experiments reveal that the ensemble predictor, enhanced through active learning, almost always consistently outperforms the baseline models for a buffer size larger than $\sim 300$ samples.

![](/Users/joseafonso/Desktop/PlasmaActiveLearning/figures/panel.png)



**Figure 1a** presents the RMSE for the ensemble models, the one that include active learning (blue) line and the one trained on the latin hypercube domain discretization (orange) as a function of the buffer size. 

**Figure 1b** shows the RMSE percentage gain (or loss) of the active learning case when compared to the baseline case for Figure 1a as a function of the buffer size.

As a result, we can conclude for a buffer size larger than $\sim 300$ samples this approach almost always outperforms the baseline case with gains between $10-20 \%$.  At very small budgets $(N≲200)$, the active‐learning ensemble (blue) actually starts off slightly worse than the latin‐hypercube baseline (orange), owing to the cold‐start noise in the first few acquisition steps. 

**Figure 1c** shows that, while both schemes scale essentially linearly with N, the active-learning strategy, which warm-starts from the previous model and only fine-tunes on the $\Delta N$ new points, cuts out the cost of re-processing all $N$ samples at every iteration. By contrast, the baseline at each buffer size $N$ generates  $N$ samples from the physical simulator and then trains the model from scratch on all $N$ points (for 600 epochs). As a result, at $N = 500$ the active-learning loop is already over $100 s$ faster than the baseline.

**Figure 1d** plots the time ratio (active / baseline) as a function of buffer size. You can see that beyond $\sim300$ samples the ratio drops below $0.75$ and continues to fall toward $\sim0.6$ at our largest buffer size, confirming that our incremental‐retraining scheme becomes increasingly efficient as more data are collected.



## Discussion and Next Steps

Our active learning framework illustrates a strategy for improving surrogate modeling in simulation tasks. The integration of uncertainty quantification, expected gradient updates, and coverage metrics into the acquisition function aids in efficiently directing the sampling process. Future work may involve:

- **Hyperparameter Fine-Tuning:**
  Refining the model hyperparameters and acquisition function weights to further enhance performance and sample efficiency.
- **Extension to Offline Models:**
  Investigating whether the active learning gains can be transferred to improve offline models, potentially through a two-stage training process.



## References 

Shin et al., *Neural Bootstrapper*

Gramacy. *Gaussian process modelig, design and optimization for applied sciences*

Osband et al.,  *Deep Exploration via Bootstrapped DQN*

Lewis et al., *A Sequential Algorithm for Training Text Classifers*

Sergey Levine , Lecture 11 and 13 - CS 285 RAIL 

Smith, Freddie Bickford, et al. *Prediction-Oriented Bayesian Active Learning*. 17 Apr. 2023. arXiv:2304.08550, https://arxiv.org/abs/2304.08550.