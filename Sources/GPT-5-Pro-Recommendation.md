## 0) Derived Requirements (from the two reports)

- Continuous-time, solver-native handling of irregular sampling and adaptive computation; memory-efficient training via adjoint [A§1.2].  
- Physics consistency and grey-box modeling: embed known laws (Arrhenius scaling, monotonic SOH decrease with rare regeneration, resistance non-decrease), enforce/penalize residuals (PINN/UDE) [A§5.1–5.3].  
- Robustness and stability on noisy/stochastic data; ANODE instability risk must be mitigated (regularization, better parameterizations, gated dynamics) [A§4.3]; use NODE-native regularization: contractivity, STEER, TA-BN [B§6.1–6.3].  
- Generalization across chemistries/conditions via conditioning on context α(t) (T, C‑rate, SoC window, chemistry), multi-battery training, inter-cell difference features [A§7.2–7.3].  
- Interpretability: inspect/visualize learned vector field; use interpretable dynamics parameterizations (KAN) and/or symbolic distillation (SNODE); enable regime-aware gating [A§6.2, B§2.1, B§4.1, B§7.1].  
- Stiffness tolerance and efficiency: learned time reparameterization, stiff-capable solvers (TR‑BDF2‑family), event handling; optional collocation-based simultaneous optimization for faster training [B§5.1–5.3].  
- Multi-modal/topological fusion optional (graphs) for heterogeneous data, but with clear interpretability and parameter efficiency [A§3.2].  
- Safety: conservative EOL bias acceptable/desired [A§8.3].  
- Knee-region adaptivity: fast dynamics around capacity knee must be captured (adaptive solvers, event detection) [A§1.2].

---

## 1) Candidate Design Space (3–5 options)

1) Physics-Gated UDE with KAN residuals  
- Pros: Physics consistency (UDE), interpretable residuals (KAN), regime gating stabilizes dissipative vs. conservative behavior [A§5.3, A§6.2, B§4.1].  
- Pros: Handles irregular sampling natively; robust via contractivity/STEER/TA‑BN [A§1.2, B§6.x].  
- Cons: Added complexity (gates + KAN + losses) and tuning.  
- Cons: If physics priors are misspecified, residual must compensate.

2) Characteristic-Curve UDE (C‑NODE‑style)  
- Pros: Overcomes trajectory topology limits; fewer NFEs for same budget [B§2.3].  
- Pros: Still physics-informed via UDE residuals.  
- Cons: More complex latent construction; PDE-characteristics design overhead.  
- Cons: Less off-the-shelf interpretability vs. KAN unless combined.

3) Probabilistic B‑NODE-inspired UDE (mean/variance flows)  
- Pros: Native uncertainty propagation; robust to time-varying inputs [B§3.3].  
- Pros: Provides calibrated RUL/EOL distributions.  
- Cons: Heavier training; complexity in stability control.  
- Cons: Interpretability depends on dynamics parameterization.

4) GDE (Graph ODE) fusion + UDE core  
- Pros: Fuses Euclidean and topological signals; strong for fleet/cohort learning [A§3.2, B§3.1].  
- Pros: Graph priors help generalization.  
- Cons: Graph construction cost; potential opacity unless KAN/SNODE layered in.  
- Cons: May be overkill for single-cell scenarios.

5) SNODE-UDE (Symbolic core + residual NN)  
- Pros: High interpretability/extrapolation; equation discovery [B§2.1].  
- Pros: Physics residuals become readable rules.  
- Cons: Training complexity (3-stage pipeline); sensitive to noise.  
- Cons: May need post-hoc residual NN to match SOTA accuracy.

---

## 2) Final Choice & Name

Chosen design: CHARM‑Batt UDE (Characteristic‑Reparameterized, Mechanism‑gated, KAN‑residual Universal Differential Equation)

Novelty: Unlike ACLA’s heavy feature frontend with ANODE backend [A§3.1], CHARM‑Batt centers the ODE itself: physics-embedded UDE with an attention-based dissipation gate [B§4.1], KAN-interpretable residuals [A§6.2], and a learned time reparameterization for stiffness [B§5.1]. It unifies physics consistency, expressivity, and robustness in the dynamics core, rather than relying on a large pre-ODE feature stack or graph fusion alone [A§3.2].

---

## 3) Mathematical Specification

State and inputs:  
- Latent health state y(t) ∈ R^d (d small), partitioned as y = [q, r, s]^T, where  
  - q(t) ∈ [0,1]: normalized capacity/SOH,  
  - r(t) ≥ 0: normalized DC resistance index,  
  - s(t) ∈ R^{d_s}: latent “mechanism intensities” (e.g., SEI/plating/cracking abstractions).  
- Augmented dimensions a(t) ∈ R^{d_a} initialized to 0 (ANODE-style augmentation) [A§2.2].  
- Observations x(t): compact per-cycle/per-segment features from CC/CV traces (e.g., time-in-segment, slope proxies).  
- Context α(t): temperature T, C-rate/current proxy, DoD/SoC window descriptors, rest duration, and a chemistry/domain embedding [A§7.2].  
- Optional cohort topology G: graph over cells or cycle-features if used [A§3.2].

Core dynamics (UDE with learned time reparameterization):  
Let τ be learned non-stiff time [B§5.1], dτ/dt = χφ(t,α) with χφ(t,α) > 0 (monotone reparameterization). Define z = [y;a].  
- Dynamics in τ:  
  - dz/dτ = Fphys(z, α, p) + gψ(z, x, α, τ) ⊙ Nθ(z, x, α, τ)  
  - dt/dτ = 1 / χφ(t,α)  
- Back to physical time: dz/dt = χφ(t,α)[Fphys + gψ ⊙ Nθ].  
Here, Fphys encodes priors; Nθ is the residual (KAN-parameterized); gψ ∈ [0,1] is an attention gate for dissipative evolution [B§4.1].

Physics priors and constraints:  
- Capacity fade monotonicity with rare regeneration: dq/dt ≤ εrec, with εrec ≥ 0 small, penalized if exceeded [A§5.1, A§5.2].  
- Resistance non-decreasing: dr/dt ≥ −εr (εr small slack), penalized [A§5.2].  
- Arrhenius-like temperature factors: Each nonnegative rate ρ_i scales as ρ_i(T) = κ_i exp(−E_i/(R·T)) [A§5.3].  
- Bounded rates: |dq/dt| ≤ Bq, |dr/dt| ≤ Br, ||ds/dt|| ≤ Bs; enforced by residual penalties and/or parameterization.  
- Non-negativity: q∈[0,1], r≥0, mechanism rates ≥0 when physically required; enforce via activations or barrier penalties.  
- Optional conservation-like constraints on latent flows (e.g., total side-reaction budget), if desired.

Concrete Fphys template (grey-box, minimal):  
- dq/dτ = −χ_SEI(T,α)·σ_SEI(s,α) − χ_pl(T,α)·σ_pl(s,α) − χ_mech(T,α)·σ_mech(s,α)  
- dr/dτ = χ_R(T,α)·σ_R(s,α)  
- ds/dτ = A_s(α)·s + b_s(α) with A_s constrained stable (e.g., negative diagonal)  
Each χ_•(T,α) are Arrhenius-like scalers; σ_• are nonnegative mechanism intensities from s; A_s,b_s simple linear priors. Residual Nθ shapes unknown dependencies on z,x,α.

Dissipation gate gψ (attention-based):  
- gψ(z,x,α,τ) = sigmoid(w^T·att(z,x,α,τ)), learned to suppress dissipation in conservative regimes and activate in dissipative/onset-of-knee regimes [B§4.1].

Outputs and event heads:  
- SOH: ŷ_SOH(t) = q(t).  
- Resistance: ŷ_R(t) = r(t).  
- EOL distribution: hazard head hη(y,α) ≥ 0; survival S(t) = exp(−∫_{t0}^t hη(y(ξ),α(ξ)) dξ); RUL distribution from S [A§4.1].  
- Knee-onset probability: π_knee(t) = sigmoid(kη(y,α)) and/or curvature-based auxiliary metric κ(t) ≈ −d²q/dt².  
- Optional auxiliary heads: dQ/dt, ΔQ over horizon, etc.

---

## 4) Architecture Diagram (Mermaid)

```mermaid
flowchart LR
    A[Raw charge/discharge segments] -->|minimal features| B[Segment Feature Encoder]
    C[Context α(t): T, C-rate, DoD, rest, chemistry token] --> D[Context Embedding]
    E[Inter-cell differences (optional)] --> D
    B --> F[Initializer f_init(z0|x,α)]
    D --> F
    F --> G{{Time Reparam χφ(t,α)}}
    B --> H[Attention Gate gψ(z,x,α,τ)]
    D --> H
    G --> I[[UDE Core]]
    H --> I
    I -->|Fphys + gψ⊙Nθ (KAN)| I
    I --> J[SOH/Resistance Heads]
    I --> K[Hazard/Survival Head]
    I --> L[Knee Head]
    J --> M[SOH(t), R(t)]
    K --> N[EOL/RUL distribution]
    L --> O[Knee probability]
```

---

## 5) Dynamics Parameterization

- Residual Nθ as a Kolmogorov–Arnold Network (KAN): spline-based learnable univariate functions on edges; parameter-efficient and interpretable mappings z,x,α → residual rates [A§6.2].  
- Gate gψ as attention-based mechanism over z and segment features x, enabling sharp switching between conservative and dissipative regimes (elastic vs. plastic analogue) [B§4.1].  
- Low-rank linear priors in Fphys (A_s,b_s) stabilize latent mechanism evolution; Arrhenius scalers encode T/C-rate dependence [A§5.3].  
- ANODE-style augmentation a(t) to bypass homeomorphism limits; small d_a to avoid instability [A§2.2, A§4.3].  
- Optional graph operator (lightweight): if G used, apply a single message-passing step to context embeddings before entering UDE core, preserving transparency [A§3.2].  
- Contractivity-promoting weight penalties on KAN edge weights/linear maps to bound Lipschitz constants and enhance robustness [B§6.2].  
- Post-hoc symbolic distillation: fit a SNODE SymNet to the trained KAN residuals for equation-like summaries (optional) [B§2.1].

Why this advances the goals:  
- Interpretability: KAN edges expose univariate response curves; gate trajectories reveal regime activation [A§6.2, B§4.1].  
- Stability/robustness: gate suppresses unnecessary dissipation; contractivity regularization reduces noise amplification [B§6.2]; Arrhenius/priors anchor extrapolation [A§5.3].  
- Expressivity: augmentation + KAN handle complex nonlinearities without massive parameters [A§2.2, A§6.2].

---

## 6) Solver Strategy & Numerical Choices

- Stiffness: Expected around knee onset and at low/high T (fast/slow processes) [A§1.2, B§5.1].  
- Primary approach: learn time map τ via χφ(t,α) to de-stiffen dynamics; integrate in τ (non-stiff) [B§5.1].  
- Solvers:  
  - Non-stiff segments: adaptive explicit RK with absolute/relative tolerances (ε_abs, ε_rel).  
  - Stiff segments or when χφ fails: switch to implicit TR‑BDF2/I‑TR‑BDF2; 3 function evaluations/step, better stability [B§5.3].  
  - Event handling: detect q(t)=q_EOL threshold and curvature crossings for knee; solver with root-finding events [A§1.2].  
- Gradients: adjoint sensitivity for memory efficiency [A§1.2], with safety checkpointing near events if required.  
- NFE budget: target median NFE ∈ [8, 32] per sequence; cap at NFE_max for embedded deployment.  
- Tolerances: ε_abs ∈ [1e−5, 1e−3], ε_rel ∈ [1e−6, 1e−4]; anneal during training to stabilize early phases.  
- Fallback discretization: predictor–corrector (PC‑RNN) with fixed Δt for ultra-light inference [A§2.3].

---

## 7) Training Objective

Let D be observations at irregular times {t_i}. Loss = L_data + L_phys + L_reg + L_unc.

- Data fit:  
  - SOH/Resistance: L_SOH = Σ_i w_i ||q(t_i) − Q_obs(t_i)||²; L_R similarly.  
  - EOL/RUL: negative log-likelihood of observed failure/censoring via survival S(t) = exp(−∫ hη dt); standard time-to-event NLL.  
  - Knee classification: binary cross-entropy for knee labels over window(s), or AUC-optimized surrogate.

- Physics residuals (weights λ•):  
  - Monotone SOH: L_mono = Σ_i max(0, dq/dt|_{t_i} − εrec)² [A§5.2].  
  - Non-decreasing R: L_res = Σ_i max(0, −dr/dt|_{t_i} − εr)².  
  - Arrhenius scaling coherence: for pairs at T1,T2, penalize deviations from ρ(T2)/ρ(T1) ≈ exp(−E/R·(1/T2−1/T1)) [A§5.3].  
  - Rate bounds: L_rate = Σ_i (softplus(|dq/dt|−Bq)+softplus(|dr/dt|−Br)+...).  
  - State feasibility: barriers for q∉[0,1], r<0, or s violating sign constraints.

- Continuous-time regularization:  
  - STEER: randomize final integration time T ∼ U(t1−b,t1+b) during training [B§6.1].  
  - Contractivity: L_contr = Σ layers ||W||² with slopes-restricted activations; promotes contraction [B§6.2].  
  - TA‑BN: apply temporal adaptive normalization within the ODE block to stabilize deep continuous flows [B§6.3].

- Uncertainty:  
  - Aleatoric: heteroscedastic heads predict σ_q(t), σ_r(t); L_NLL for Gaussian/Student‑t.  
  - Epistemic: ensemble over K seeds; temperature scaling to calibrate; calibration loss L_cal (e.g., ECE proxy) on validation.  
  - Optional B‑NODE-inspired mean/variance flow: propagate (μ_y, log σ_y) with a parallel ODE; include KL regularizer to stabilize [B§3.3].

Total loss:  
L = L_SOH + L_R + L_EOL + λ_knee L_knee + Σ_j λ_j L_phys,j + λ_contr L_contr + λ_cal L_cal + λ_rate L_rate.

Optional training acceleration:  
- Phase‑1 simultaneous optimization via direct collocation (states+parameters) to warm-start [B§5.2, B§4.3], then Phase‑2 adjoint fine-tune.

---

## 8) Data Interface & Irregular Sampling

- No imputation: integrate from t0 to each observed t_i; solver natively handles irregular grids [A§1.2].  
- Feature preprocessing:  
  - From CC/CV segments, compute a compact set of normalized segment/time/slope descriptors (z‑score or min‑max per dataset) [A§8.3(3)].  
  - Scale context α(t); encode chemistry token to an embedding [A§7.2].  
- Optional topology:  
  - Build a cohort graph with nodes = cells; edges weighted by similarity of early-life features; one-step message passing to enrich α at initialization [A§3.2].  
- Inter-cell difference features Δf enter α to boost cross-condition robustness [A§7.3].

---

## 9) Generalization Plan

- Conditioning: α(t) includes T, C‑rate, DoD/SoC window, rest, and chemistry/domain embeddings so F(·|α) adapts [A§7.2].  
- Multi-battery training: include many fully aged cells across conditions to learn entire degradation manifold [A§7.3].  
- Inter-cell differences: inject Δf = features(cell) − cohort_mean(features) into α(t) [A§7.3].  
- Protocols:  
  - Leave‑chemistry‑out: train on all but one chemistry; test directly on held‑out chemistry [A§4.2].  
  - Cross‑condition: hold out temperature/C‑rate bands; evaluate zero‑shot generalization.

---

## 10) Evaluation Protocol & Metrics

- RMSE_SOH: sqrt(mean((q̂−q)^2)) in %.  
- AE_EOL: |t̂_EOL − t_EOL| / t_EOL in %.  
- Calibration: NLL on held‑out; Expected Calibration Error (ECE) for predictive intervals.  
- Knee detection: AUPRC and F1 for knee labels within ±Δ cycles.  
- Constraint violation rate: fraction of samples with dq/dt > εrec, dr/dt < −εr, or state bound breaches.  
- Robustness-to-noise: ΔRMSE_SOH and ΔAE_EOL under input perturbations (Gaussian, salt‑and‑pepper) [B§6.2].  
- Efficiency: median NFE, tail NFE95, latency per sequence at set tolerances.  
- Stability: solver failure rate, event detection accuracy.

---

## 11) Ablation & Stress Tests

Ablations:  
1) Remove physics gate gψ (set to 1) vs. full gate.  
2) Replace KAN with MLP; compare interpretability/accuracy [A§6.2].  
3) No time reparameterization (χφ≡1) vs. learned χφ [B§5.1].  
4) No augmentation (a≡0) vs. ANODE dims [A§2.2].  
5) Disable contractivity penalty vs. enabled [B§6.2].  
6) Remove Arrhenius scalers from Fphys vs. included [A§5.3].  
7) Collocation warm-start off vs. on [B§5.2].  
8) Ensemble off vs. on; evaluate calibration.

Stress tests:  
- Noise spikes in segments; missing segments; partial cycles; abrupt T/C‑rate shifts; long rest periods; synthetic knee sharpness variations.

---

## 12) Interpretability & Diagnostics

- Vector-field probes: visualize dq/dt, dr/dt over (q,r) planes at fixed α; show flow lines [B§7.1].  
- Gate activity maps gψ over life: when/why dissipation activates [B§4.1].  
- KAN edge functions: plot learned univariate splines vs. inputs to explain residual physics [A§6.2].  
- Sensitivities: ∂q(t*)/∂α_k and ∂EOL/∂α_k to quantify condition effects.  
- Symbolic distillation (optional): SNODE regression of residual terms; report candidate symbolic forms [B§2.1].  
- High-order flow (optional): Event Transition Tensors to approximate uncertainty propagation to EOL events [B§2.2].  
- Automatic reports: constraint violations over time, NFE histograms, calibration curves, knee localization error.

---

## 13) Risk Register & Mitigations

- Solver instability near knees: use event-aware step control; fallback to implicit TR‑BDF2; raise ε_abs early, anneal later [B§5.3].  
- Non-physical trends (SOH increase): strong λ_mono, small εrec, clamp via activation; curriculum to introduce physics losses progressively [A§5.2].  
- Overfitting with augmentation: limit d_a; contractivity penalty; STEER [A§4.3, B§6.1–6.2].  
- Poor calibration: use ensembles, temperature scaling, NLL training; ECE-based tune.  
- Label leakage (future context): enforce causal feature construction per cycle; audits.  
- Conservative EOL bias excessive: calibrate hazard prior from validation; allow small regeneration εrec [A§8.3].  
- Misspecified Arrhenius: allow residual to correct; schedule λ_Arrhenius.

---

## 14) Deployment Profile

- Parameters:  
  - KAN residual with small width/degree yields |θ| ≈ 0.2–1.0M; gate/priors negligible overhead.  
- NFE:  
  - Training median NFE ≈ 16–40 (adaptive); inference target 8–24.  
- Latency: proportional to NFE and solver tolerances; provide a fixed-step PC‑RNN surrogate with 2nd‑order predictor–corrector for edge devices [A§2.3].  
- Safety mode: event-based early stopping when conservative EOL reached [A§8.3].

---

## 15) Minimal Prototype Plan (7–10 steps)

1) Define minimal segment features x(t) and context α(t); normalize; create chemistry/domain tokens.  
2) Build Fphys template with Arrhenius scalers and simple stable latent s dynamics.  
3) Implement KAN residual Nθ and attention gate gψ; add ANODE augmentation a(t)=0 init.  
4) Add learned time map χφ(t,α) ensuring positivity (e.g., softplus).  
5) Implement UDE ODE system in τ with event functions for q= q_EOL and knee curvature.  
6) Choose adaptive solver with events; configure ε_abs, ε_rel; enable adjoint gradients.  
7) Define losses: data, physics, regularizers (STEER, contractivity), uncertainty heads; set λ weights.  
8) Optional Phase‑1 collocation warm-start on a subset; then adjoint fine-tune full model [B§5.2].  
9) Train with multi-battery batches; apply STEER; monitor constraint violations, NFE, metrics.  
10) Evaluate: LCO‑chemistry‑out, cross‑condition; run ablations; generate interpretability report.

---

## 16) Pseudocode (framework-agnostic)

Forward solve (irregular time grid):
```
function forward_solve(x_seq, alpha_seq, t_grid, z0):
    # z = [y; a], initialize from initializer using first x, alpha
    z = z0
    outputs = []
    for t_i in t_grid:
        # integrate from current time to t_i with learned time map
        define ODE in tau:
            dz_dtau = F_phys(z, alpha(t)) + g_gate(z, x(t), alpha(t), tau) * N_resid(z, x(t), alpha(t), tau)
            dt_dtau = 1 / chi_time(t, alpha(t))
        integrate [z, t] in tau until t reaches t_i (event on t - t_i = 0)
        record heads: SOH=q(z), R=r(z), hazard=h(z, alpha), knee=pi(z, alpha)
        outputs.append(heads)
    return outputs, trajectory
```

Loss computation:
```
function compute_loss(outputs, labels, traj, alpha_seq):
    L_data = mse(outputs.SOH, labels.SOH) + mse(outputs.R, labels.R) + survival_nll(outputs.hazard, labels.EOL)
    # compute derivatives along trajectory by solver-provided slopes or finite diff
    dq_dt, dr_dt = derivatives(traj)
    L_phys = mono_penalty(dq_dt, eps_rec) + nondec_penalty(dr_dt, eps_r)
           + arrhenius_penalty(traj, alpha_seq) + rate_bound_penalty(dq_dt, dr_dt)
           + state_barriers(traj)
    L_reg = steer_time_warp() + contractivity_penalty(params) + TA_BN_norm(traj)
    L_unc = nll_aleatoric(outputs) + calibration_penalty(outputs, labels)
    return L_data + sum(lambda_i * L_phys_i) + L_reg + L_unc
```

Backward/gradient approach:
```
function train_step(batch):
    # Option A: adjoint-based gradients
    outputs, traj = forward_solve(...)
    L = compute_loss(outputs, labels, traj, alpha_seq)
    grads = adjoint_backprop(L)  # memory efficient
    update_parameters(grads)

    # Option B (warm-start): collocation simultaneous optimization
    # discretize time; set z(t_k) as decision vars; enforce ODE residuals at collocation points
    # solve nonlinear program to minimize L subject to residual constraints
```

Evaluation loop:
```
function evaluate(dataset, thresholds):
    metrics = init_metrics()
    for sample in dataset:
        outputs, _ = forward_solve(...)
        update_RMSE_SOH(outputs.SOH, labels.SOH)
        update_AE_EOL(predict_EOL(outputs.hazard), labels.EOL)
        update_knee_metrics(outputs.knee, knee_labels)
        update_calibration(outputs, labels)
        count_constraint_violations(...)
        record_NFE_latency(...)
    return summarize_metrics(metrics)
```

---

## 17) Expected Novelty & Impact

- First battery UDE to jointly combine: (i) attention-based dissipation gating [B§4.1], (ii) KAN-interpretable residual dynamics [A§6.2], and (iii) learned time reparameterization for stiffness [B§5.1], all inside the ODE core.  
- Physics is primary, not a post-hoc loss: Arrhenius-scaled priors plus strict monotonic constraints steer learning toward plausible flows [A§5.2–5.3].  
- Variational/ensemble uncertainty and a survival head yield calibrated EOL/RUL distributions for safety-critical use [A§6.1, B§3.3, B§7.1].  
- Robustness by design via contractivity penalties, STEER temporal regularization, and TA‑BN normalization in continuous depth [B§6.1–6.3].  
- Generalization via explicit conditioning on α(t) and inter‑cell differences, aligned with universal modeling strategies [A§7.2–7.3].  
- Event-aware integration captures knee dynamics and supports conservative EOL triggering [A§1.2, A§8.3].  
- Interpretability toolkit: vector-field maps, gate activity, KAN edge plots, optional symbolic distillation and high‑order flow analysis [A§6.2, B§2.1–2.2, B§7.1].  
- Rapid testing path: start with non-stiff solver + adjoint, then enable χφ and gate; ablate physics residuals to quantify gains.

---