

# **The Continuous-Time Revolution in Battery Prognostics: A Synthesis of Neural Ordinary Differential Equations and Physics-Informed Learning**

### **Executive Summary**

This report provides a comprehensive analysis of a transformative paradigm in lithium-ion battery prognostics: the use of Neural Ordinary Differential Equations (NODEs) to model degradation. This approach reframes battery aging from a discrete sequence of health states into the continuous-time evolution of a dynamical system, offering fundamental advantages over traditional discrete-time models like Recurrent Neural Networks (RNNs). The analysis traces the architectural evolution from canonical NODEs and Augmented NODEs (ANODEs) to the current state of the art, which is characterized by two divergent but powerful philosophies. The first, exemplified by the ACLA framework, focuses on sophisticated feature abstraction through a synergistic frontend of attention mechanisms, CNNs, and LSTMs to achieve superior accuracy and generalization to unseen battery chemistries.1 The second, represented by architectures like the GAT-KAN-ODE, prioritizes the fusion of heterogeneous data sources and enhanced model interpretability through novel components like Graph Attention Networks and Kolmogorov-Arnold Networks (KANs).1

Quantitative benchmarking consistently demonstrates that NODE-based models outperform traditional RNNs, with advanced variants achieving state-of-health estimation errors below 2.5% on entirely new datasets.1 However, the limitations of purely data-driven models—namely, their potential for non-physical predictions and poor generalization—have catalyzed the emergence of a new frontier: Physics-Informed Machine Learning, also known as Scientific Machine Learning (SciML). Frameworks such as Physics-Informed Neural Networks (PINNs) and Universal Differential Equations (UDEs) embed physical laws directly into the learning process, promising to create "grey-box" models that are more robust, data-efficient, and trustworthy.2 This convergence of data-driven flexibility and physics-based rigor is poised to become the cornerstone of next-generation Battery Management Systems (BMS). Strategic recommendations for research and development include prioritizing these physics-informed models, investing in interpretable architectures, and developing universal training strategies. For industrial application, a phased implementation approach, leveraging the conservative nature of ODE-based predictions for safety, is advised.

## **1\. A Paradigm Shift: Modeling Battery Degradation as a Continuous Dynamical System**

### **1.1. Theoretical Underpinnings of the ODE-Based Approach**

The traditional approach to modeling lithium-ion battery degradation treats a battery's lifecycle as a discrete sequence of charge-discharge cycles. A more physically analogous and powerful paradigm, however, frames the evolution of a battery's State of Health (SOH) as the continuous-time trajectory of a dynamical system.1 This conceptual shift is the foundation for applying Neural Ordinary Differential Equations to battery prognostics.

The core principle is to represent the battery's state vector, y, which includes SOH and other relevant features, as evolving according to a governing ordinary differential equation (ODE):

$$\\dot{y} \= F(y, t)$$

In this equation, y˙​ represents the instantaneous rate of change of the state vector with respect to time or cycle number, and F is an unknown, generally non-linear function that encapsulates the complex electrochemical and physical dynamics of degradation, such as solid-electrolyte interphase (SEI) growth and lithium plating.1 The central challenge is to learn this unknown function directly from observational data.  
Neural networks, as universal function approximators, provide a solution. By parameterizing the unknown dynamics function with a neural network, F(y,t,θ), the prognostics problem is transformed into a learning problem: finding the network parameters θ that best describe the observed degradation data.1 Once this function F is learned, predicting the future SOH from a known initial state y(t0​) becomes an initial value problem (IVP). The future state y(t1​) is found by integrating the learned dynamics over the desired time horizon:

$$y(t\_1) \= y(t\_0) \+ \\int\_{t\_0}^{t\_1} F(y(\\tau), \\tau, \\theta) d\\tau$$

This integration is performed by a numerical ODE solver, which effectively becomes a layer within the deep learning architecture.1 This approach is not merely a methodological choice; it represents a fundamental alignment of the modeling framework with the underlying physical reality. Degradation mechanisms are continuous physical and chemical processes, not discrete events that occur only at the end of a cycle.3 By modeling the rate of change of the system's health state, NODEs adopt the language of differential equations, which is native to the physical sciences and provides a natural scaffolding for the future integration of physical laws.2

### **1.2. Fundamental Advantages Over Discrete-Time Models (RNNs, LSTMs)**

This continuous-depth formulation offers several profound advantages over conventional discrete-layer architectures like RNNs, Long Short-Term Memory (LSTM), and Gated Recurrent Units (GRU).1 The architecture of Residual Networks (ResNets), with updates of the form $h\_{t+1} \= h\_t \+ f(h\_t, \\theta\_t)$, can be viewed as a discrete Euler discretization of an underlying differential equation. NODEs emerge from taking the continuous limit of these steps, resulting in the ODE $dh(t)/dt \= f(h(t), t, \\theta)$.1 In this framework, passing an input through the "network" is equivalent to solving this ODE over a time interval $\[t\_0, t\_1\]$. This reframing yields significant practical benefits:

* **Handling Irregularly Sampled Data:** Real-world battery usage data from vehicles or grid storage is often sampled at irregular intervals due to varying operational patterns or data logging constraints. Standard RNNs are designed for sequences with fixed, regular time steps and struggle with such data. NODEs are naturally equipped to handle this challenge, as the numerical ODE solver can evaluate the system's dynamics and state at any required time point, seamlessly integrating over non-uniform intervals.1  
* **Adaptive Computation:** Modern ODE solvers are adaptive, meaning they can dynamically adjust their step size based on the complexity of the learned dynamics. They can take larger, more computationally efficient steps when the degradation trajectory is smooth and linear, and smaller, more precise steps during periods of rapid change, such as the onset of a capacity "knee".1 This allows for an explicit and principled trade-off between computational cost and numerical precision, a feature absent in fixed-step RNNs.  
* **Memory Efficiency:** A significant bottleneck in training very deep neural networks is the memory required to store the activations of every layer for backpropagation. NODEs circumvent this problem by using the adjoint sensitivity method. This technique computes gradients by solving a second, augmented ODE backward in time. This method has a constant memory cost with respect to the depth (i.e., the integration time), enabling the training of models with effectively infinite depth without the risk of memory overflow.1

## **2\. An Architectural Taxonomy of Foundational Neural ODE Models**

The application of ODE principles to battery degradation has spurred the development of several foundational architectures, each embodying different trade-offs between theoretical power, computational efficiency, and practical robustness.

### **2.1. The Canonical Neural ODE (NODE)**

The standard NODE model directly parameterizes the degradation dynamics function $F(y, t, \\theta)$ using a relatively simple neural network, such as a Multi-Layer Perceptron (MLP). The model's forward pass involves using a numerical ODE solver, like the Dormand-Prince 5(4) method (Dopri5), to integrate the learned dynamics from an initial state to produce a predicted trajectory. The backward pass for training leverages the computationally efficient adjoint sensitivity method to calculate parameter gradients, treating the ODE solver as a black box and avoiding the need to backpropagate through its internal operations.1 This architecture serves as the essential baseline for all subsequent developments.

### **2.2. Augmented Neural ODEs (ANODE): Overcoming Representational Bottlenecks**

A critical theoretical limitation of the canonical NODE is that the continuous transformation it learns must be a homeomorphism. This mathematical constraint means it cannot change the topology of the data space; for instance, it cannot map a high-dimensional space to a lower-dimensional one or cause trajectories to cross.1 This can be a restrictive constraint when modeling complex degradation phenomena.

Augmented Neural ODEs (ANODEs) were introduced to overcome this limitation. The solution is to augment the state vector $y$ with additional, artificial dimensions $a$ that are initialized to zero. The dynamics are then learned in this higher-dimensional space 1:

$$\\frac{d}{dt}\\begin{bmatrix}y \\\\ a\\end{bmatrix} \= F\\left(\\begin{bmatrix}y \\\\ a\\end{bmatrix}, \\theta(t)\\right), \\quad \\begin{bmatrix}y(0) \\\\ a(0)\\end{bmatrix} \= \\begin{bmatrix}y\_0 \\\\ 0\\end{bmatrix}$$  
By allowing the dynamics to evolve in this augmented space, the model can learn far more complex and expressive transformations, effectively resolving the representational bottleneck of standard NODEs. This has been shown to lower training losses and improve predictive performance in battery SOH forecasting.1

### **2.3. Predictor-Corrector RNNs (PC-RNN): A Discretized Approximation**

The Predictor-Corrector RNN (PC-RNN) serves as a conceptual bridge between continuous-time ODEs and discrete-time RNNs. Instead of relying on a general-purpose, black-box ODE solver, it implements a specific numerical integration scheme analogous to a second-order Runge-Kutta method.1 The process for advancing from cycle $k$ to $k+1$ involves two explicit steps:

1. **Predictor Step:** An initial estimate of the next state, $\\hat{y}\_{k+1}$, is calculated using a simple forward Euler step: $\\hat{y}\_{k+1} \= y\_k \+ \\Delta t \\cdot F(y\_k, \\theta)$.  
2. **Corrector Step:** This initial estimate is used to evaluate the gradient at the predicted future point. The final state, $y\_{k+1}$, is then computed by averaging the gradient at the current state and the gradient at the predicted state: $y\_{k+1} \= y\_k \+ \\frac{1}{2}\\Delta t \\cdot (F(y\_k, \\theta) \+ F(\\hat{y}\_{k+1}, \\theta))$.

This structure provides a computationally simpler and more explicit alternative to a full ODE solver. The evolution from NODE to ANODE to PC-RNN reveals a fundamental design tension between theoretical expressiveness, computational complexity, and practical stability. The canonical NODE is simple but representationally limited. ANODE overcomes this limitation, offering superior theoretical power, but this increased freedom can lead to instability on noisy data, as later performance analysis shows.1 PC-RNN moves in the opposite direction, sacrificing the adaptive computation of a true ODE solver for a fixed, computationally cheaper, and potentially more stable update rule. This creates a spectrum of choices for model architects, where the optimal selection is contingent on the specific application's data quality and hardware constraints.

## **3\. The State of the Art in Hybrid Architectures: Feature Extraction vs. Data Fusion**

Building upon the foundational NODE family, recent research has focused on developing sophisticated hybrid architectures. These models combine the continuous-time dynamics of NODEs with powerful feature extraction modules, leading to a divergence in prognostic philosophies: one centered on perfecting feature abstraction from a single data source, and another focused on fusing heterogeneous data and enhancing the transparency of the core model.

### **3.1. The ACLA Framework: A Deep Dive into Synergistic Feature Abstraction**

The ACLA model represents a significant advance in the "powerful frontend" philosophy, integrating an Attention mechanism, a Convolutional Neural Network (CNN), and an LSTM network into the ANODE framework.1 This architecture features a multi-stage pipeline designed to produce a highly informative latent representation of the battery's state, which is then evolved in continuous time by the ANODE backend. The roles of the components are highly synergistic:

* **Attention Layer:** The model first applies an attention mechanism to the input features (normalized charging times), allowing it to dynamically assign importance weights to different parts of the charging curve at each cycle and focus on the most salient indicators of degradation.1  
* **CNN Layers:** The attention-weighted features are processed by 1D convolutional layers, which excel at extracting local, time-invariant patterns from the charging curve, such as the shape and slope of different voltage segments.1  
* **LSTM Unit:** The sequence of feature maps from the CNN is fed into an LSTM, whose role is to model the long-range temporal dependencies across many cycles, capturing how the degradation patterns identified by the CNN evolve over the battery's lifetime.1  
* **ANODE Solver:** Finally, the rich, context-aware hidden state vector from the LSTM is used as the initial condition for the ANODE solver, which models the continuous-time evolution of this latent health state to predict future degradation.1

The innovation of ACLA lies in its specialized feature engineering pipeline that precedes the ODE solver. This approach is predicated on the principle that all necessary information is contained within the primary signal (the charging curve), provided it is processed with sufficient sophistication before being handed to the dynamic model.

### **3.2. The GAT-KAN-ODE Framework: Fusing Topological and Euclidean Data**

A distinct and novel approach, representing the "expressive core" philosophy, employs a dual-stream architecture that processes degradation information from both Euclidean and non-Euclidean spaces, leveraging a Kolmogorov-Arnold Network (KAN) to parameterize the neuronal ODE.1

* **Non-Euclidean Branch (GAT):** This branch models the battery's health indicators as a graph. Nodes represent measurements (e.g., voltage at different cycles), and edges represent their relationships (e.g., correlation). A Graph Attention Network (GAT) is then used to learn the implicit topological relationships, capturing complex interdependencies that a standard sequential model would overlook.1  
* **Euclidean Branch (CNN):** In parallel, a 2D CNN processes raw time-series data as a grid or image, extracting local spatial and temporal features in the conventional manner.1  
* **Neuronal ODE with KAN:** Feature vectors from both branches are concatenated to form the initial state for a neuronal ODE. A crucial innovation is that the ODE's dynamics are parameterized not by a standard MLP, but by a KAN. KANs feature learnable activation functions on the network edges rather than fixed activations on the nodes, making them more parameter-efficient and significantly more interpretable than traditional MLPs.1

### **3.3. Analysis of Divergent Philosophies: Powerful Frontends vs. Expressive Cores**

The emergence of these two state-of-the-art architectures reflects a maturation of the field, moving beyond a monolithic approach to specialized solutions. Early models like NODE and ANODE used a generic MLP for the dynamics and fed it relatively raw features; the innovation was the continuous-time solver itself.1 ACLA represents the pinnacle of the "feature engineering" school, implicitly arguing that the quality of the initial state vector is more important than the specific parameterization of the dynamics function. In contrast, the GAT-KAN-ODE model represents the "model architecture" school, arguing that fusing disparate data types and enhancing the transparency of the dynamics function are paramount. This is not just a technical difference but a strategic one. An organization focused on maximizing accuracy on a specific benchmark with a single data source might favor the ACLA approach, while one building a safety-critical BMS that must fuse sensor data and be certifiable by regulators would gravitate towards the GAT-KAN-ODE approach.

**Table 1: Architectural Comparison of Foundational and Hybrid NODE Models**

| Model Name | Base Framework | Feature Extractor | Dynamics Parameterization | Key Innovation | Targeted Problem |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **NODE** | ODE | None (Direct Input) | MLP | Continuous-depth model using ODE solver | Basic time-series forecasting |
| **ANODE** | ODE | None (Direct Input) | MLP | State augmentation to overcome topological constraints | Learning complex, non-homeomorphic transformations |
| **PC-RNN** | RNN | None (Direct Input) | MLP | Discretized ODE using a predictor-corrector scheme | Computationally efficient ODE approximation |
| **ACLA** | ODE (ANODE) | Attention \+ 1D CNN \+ LSTM | MLP | Synergistic feature extraction frontend for ANODE | Generalization across diverse battery datasets |
| **GAT-CNN-KAN-ODE** | ODE | GAT (Topological) \+ 2D CNN (Euclidean) | KAN | Fusion of Euclidean/non-Euclidean features; Interpretable dynamics | Multi-modal data fusion and model interpretability |

## **4\. Empirical Validation and Performance Benchmarking**

Quantitative evaluation across standardized datasets reveals the consistent advantages of NODE-based models and highlights the performance gains achieved by advanced hybrid architectures.

### **4.1. Quantitative Analysis on Standardized Datasets (NASA, Oxford)**

Multiple studies have benchmarked NODE-based models against traditional RNNs on the widely used Oxford and NASA battery datasets, using Root Mean Squared Error on SOH prediction ($RMSE\_{SOH}$) and error in End-of-Life (EOL) prediction as key metrics.1 Foundational work demonstrated that ANODE and PC-RNN generally outperformed LSTM and GRU, with ANODE achieving an average $RMSE\_{SOH}$ below 3% on the Oxford dataset.1 On the more challenging NASA dataset, known for its irregular degradation patterns and capacity regeneration phenomena, the simpler NODE and PC-RNN models proved more robust.1

Subsequent work demonstrated the superiority of the ACLA architecture. When trained on 90% of the NASA dataset, ACLA achieved an $RMSE\_{SOH}$ of 1.19%, a remarkable 74.1% error reduction compared to the baseline ANODE model.1 Similarly, the GAT-KAN-ODE model was shown to achieve optimal minimum RMSE while reducing the number of model parameters by 39-49% compared to benchmarks like stacked CNNs and LSTMs.1

**Table 2: Consolidated Performance Metrics on Benchmark Datasets (NASA & Oxford)**

| Data Split | GRU | LSTM | NODE | PC-RNN | ANODE | ACL | ACLA |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Oxford-50%** | 3.99 | 4.73 | 2.37 | 1.74 | 2.02 | 2.12 | 1.74 |
| **Oxford-70%** | 1.66 | 2.01 | 1.13 | 1.05 | 1.77 | 1.89 | 1.74 |
| **Oxford-90%** | 1.79 | 1.40 | 1.46 | 1.08 | 1.17 | 0.97 | 0.93 |
| **NASA-50%** | 7.57 | 4.06 | 4.49 | 9.54 | 10.88 | 7.41 | 8.87 |
| **NASA-70%** | 4.56 | 5.88 | 3.97 | 3.81 | 5.36 | 3.96 | 3.54 |
| **NASA-90%** | 2.39 | 4.97 | 1.85 | 1.87 | 4.59 | 1.41 | 1.19 |
| Note: Data synthesized from.1 $RMSE\_{SOH}$ shown in %; lower values are better. |  |  |  |  |  |  |  |

### **4.2. The Critical Challenge of Generalization: Performance on Unseen Chemistries (TJU, HUST)**

A critical weakness of many data-driven models is their inability to generalize to new data that differs from the training set in chemistry, manufacturing, or operating conditions.2 The ACLA framework was explicitly designed and tested to address this challenge. After being trained exclusively on the NASA (LCO/NCA) and Oxford (LCO/NMC) datasets, the ACLA model was evaluated directly on two entirely independent datasets: Tianjin University (TJU, NCM+NCA) and Huazhong University of Science and Technology (HUST, LFP).1

The results were compelling. On the challenging HUST dataset, ACLA achieved a low average $RMSE\_{SOH}$ of 2.24% and an average EOL absolute error ($AE\_{EOL}$) of 5.33%. This performance represents a 57% and 54.7% reduction in error, respectively, compared to the baseline NODE model.1 This provides strong evidence that the sophisticated, hierarchical feature extraction within ACLA allows it to learn fundamental, abstract degradation signatures that are transferable across different battery types, rather than simply memorizing dataset-specific patterns. By learning an abstract "language" of degradation—for example, how the slope of a particular voltage region evolves over time—ACLA can recognize these patterns even in batteries with different chemistries, which may manifest them at different absolute voltage levels or time scales. This capability is a clear directive for R\&D: achieving universal models requires moving up the ladder of abstraction to find invariant representations of degradation.

**Table 3: Generalization Performance on Unseen Datasets (TJU & HUST)**

| Dataset | Metric | NODE | ANODE | ACL | ACLA |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **HUST** | $RMSE\_{SOH}$ % (Mean ± Std) | $5.22 \\pm 0.64$ | $3.45 \\pm 0.68$ | $2.34 \\pm 0.91$ | **$2.24 \\pm 0.99$** |
|  | $AE\_{EOL}$ % (Mean ± Std) | $11.76 \\pm 1.62$ | $8.05 \\pm 1.55$ | $5.60 \\pm 2.20$ | **$5.33 \\pm 2.45$** |
| **TJU** | $RMSE\_{SOH}$ % (Mean ± Std) | $0.94 \\pm 0.24$ | $1.17 \\pm 0.25$ | $1.04 \\pm 0.27$ | **$1.04 \\pm 0.27$** |
|  | $AE\_{EOL}$ % (Mean ± Std) | $1.23 \\pm 0.59$ | $1.28 \\pm 0.76$ | $1.24 \\pm 0.76$ | **$1.01 \\pm 0.72$** |
| Note: Data from.1 Best-performing model for each metric is in bold. |  |  |  |  |  |

### **4.3. Deconstructing Performance Trade-offs: Expressiveness vs. Robustness on Noisy Data**

The quantitative data reveals a subtle but critical trade-off. While ANODE is theoretically more powerful than NODE, its performance on the stochastic NASA dataset was highly unstable, with an $RMSE\_{SOH}$ exceeding 10% in one scenario.1 The simpler NODE and discretized PC-RNN models, however, remained more stable. This suggests that the increased flexibility afforded by ANODE's augmented dimensions can lead to overfitting or learning unstable dynamics when the underlying degradation process is noisy or irregular—a critical consideration for real-world deployment.

The success of the ACLA model, which uses an ANODE backend, indicates that this instability can be mitigated. Its powerful Attention-CNN-LSTM frontend acts as a sophisticated filter, transforming the raw, potentially noisy input data into a smoother, more informative latent representation. This frontend effectively regularizes the problem, allowing the powerful ANODE solver to model the core degradation trend without being derailed by noise. This highlights a key design principle: the more complex and expressive the ODE solver, the more critical the quality and stability of its initial state vector become.

## **5\. The Next Frontier: Physics-Informed Scientific Machine Learning (SciML)**

### **5.1. From "Black-Box" to "Grey-Box": The Rationale for Integrating Domain Knowledge**

A primary limitation of all purely data-driven models is that their predictions are not constrained by the laws of physics. This can lead to non-physical forecasts, such as SOH increasing over time, which erodes trust and presents a barrier to adoption in safety-critical systems.1 Scientific Machine Learning (SciML) has emerged as a framework to address this by fusing the flexibility of data-driven methods with the rigor of domain knowledge, creating more interpretable, robust, and scientifically grounded solutions.3 This hybrid approach aims to capture both known and unknown degradation dynamics, leveraging the strengths of both modeling paradigms.5 The emergence of SciML represents a fundamental acknowledgment of the limits of pure data-driven modeling and signals a convergence of the machine learning and electrochemical modeling communities, driven by the industrial need for prognostic tools that are not just accurate, but also trustworthy and data-efficient.

### **5.2. Physics-Informed Neural Networks (PINNs): Constraining Learning with Physical Laws**

PINNs are a prominent implementation of SciML where the model's loss function is augmented with a residual term that penalizes violations of known physical laws, which are often expressed as differential equations.1 For batteries, these physical laws could include simplified electrochemical models, conservation laws, or empirical degradation equations describing mechanisms like SEI growth.7 This physics-based residual acts as a powerful form of regularization, guiding the learning process towards solutions that are not only consistent with the data but also physically plausible. This approach can significantly improve generalization and reduce the amount of training data required to achieve high accuracy.16 The continuous-time framework of NODEs is exceptionally well-suited for this fusion, providing a natural structure onto which these differential constraints can be applied.1

### **5.3. Universal Differential Equations (UDEs): Embedding Known Physics into Model Architecture**

UDEs represent a related but distinct SciML approach. Instead of adding physics to the loss function, UDEs embed known physical principles directly into the model's architecture.2 The governing ODE is formulated as a combination of a known, physics-based component and an unknown component parameterized by a neural network:

$$\\dot{y} \= F\_{known}(y, t, p) \+ NN(y, t, \\theta)$$

Here, Fknown​ represents a simplified physical model (e.g., an Arrhenius relationship for calendar aging), while the neural network NN learns the "missing physics"—the discrepancies between the simplified model and the complex reality observed in the data.3 Studies using this framework have demonstrated high precision, achieving a mean squared error of 9.90 on synthetic battery degradation data.2 This approach effectively breaks through the respective ceilings of purely data-driven and purely physics-based models, combining the generalizable foundation of physics with the flexibility of machine learning.  
**Table 4: A Comparative Framework of Prognostic Paradigms**

| Paradigm | Example Architectures | Core Principle | Predictive Accuracy | Data Requirement | Generalizability | Interpretability | Computational Cost |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Discrete Data-Driven** | RNN, LSTM, GRU | Model degradation as a discrete sequence of states. | Moderate | High | Low | Low (Black-Box) | Moderate |
| **Continuous Data-Driven** | NODE, ANODE, ACLA | Model the continuous-time rate of change of degradation. | High | Moderate-High | Moderate-High | Low (Black-Box) | Moderate-High |
| **Physics-Informed (SciML)** | PINN, UDE | Fuse known physical laws (ODEs) with neural networks. | High | Low-Moderate | High | High (Grey-Box) | High |

## **6\. The Imperative for Interpretability in Safety-Critical Systems**

### **6.1. The "Black-Box" Problem as a Barrier to BMS Adoption**

The opaque, "black-box" nature of many deep learning models remains a significant barrier to their adoption in safety-critical systems like automotive or grid-scale BMS.1 In these applications, understanding *why* a model makes a certain prediction is crucial for validation, certification, failure analysis, and building trust with operators and regulators.13 A model that predicts battery failure with high accuracy is of limited practical use if its decision-making process cannot be audited and verified. This has created a parallel research track focused not just on reducing error, but on building models whose reasoning is transparent.

### **6.2. Model-Inherent Interpretability: The Promise of Kolmogorov-Arnold Networks (KANs)**

The introduction of KANs as a parameterization for the ODE dynamics in the GAT-KAN-ODE model is a prime example of the trend towards building interpretability directly into the model architecture.1 Unlike MLPs, which have fixed non-linearities at the nodes, KANs have learnable, univariate spline activation functions on the edges. This structure offers a potential pathway to visualizing and even symbolically analyzing the learned degradation dynamics. By decomposing a complex, multivariate function into a series of simpler, one-dimensional functions, KANs can transform the core of the ODE model from an opaque black box into a more transparent grey box.1

### **6.3. Post-Hoc Explainability and Feature Importance**

In addition to designing inherently interpretable models, post-hoc methods can be applied to existing black-box models to gain insight. Model-agnostic techniques like SHAP (SHapley Additive exPlanations) can be used to rank the importance of various input features, revealing which factors most influence the model's predictions.13 Furthermore, architectures like the Temporal Fusion Transformer (TFT), which has been applied to battery degradation, include specialized components such as attention mechanisms that provide interpretable results, shedding light on path-dependency and identifying which historical time steps are most critical for a given forecast.18

## **7\. The Quest for Universal Battery Models**

### **7.1. Defining the Grand Challenge: A Single Model for All Chemistries and Conditions**

The ultimate goal of battery prognostics is to develop a single, universal, or "foundation" model that can accurately predict the lifetime of any battery, regardless of its chemistry, design, or operating conditions, using minimal cell-specific training data.1 Achieving this would dramatically accelerate battery development, optimize fleet management, and enable a reliable second-life battery market. This pursuit mirrors the development of foundation models in other AI domains, where the strategy is shifting from training bespoke models for specific tasks to pre-training massive, generalist models on all available data, which can then be fine-tuned for specific applications.

### **7.2. Architectural Pathways to Universality**

Model architectures can be explicitly designed to facilitate universality. One key approach, particularly suited to the ODE framework, is to reformulate the governing equation to be conditioned on an input vector, α(t), that represents contextual variables like temperature, C-rate, and chemistry-specific parameters 1:

$$\\dot{y} \= F(y, \\theta(t), \\alpha(t), t)$$

This allows a single learned dynamics function F to adapt its predictions based on these contextual inputs. In parallel, graph neural network-based models like BatteryFormer, which learn from fundamental material composition and structural prototypes rather than time-series data, point towards a future of predicting properties from first principles, a method that is inherently more generalizable across different materials.20

### **7.3. Data-Centric Pathways: Learning from Inter-Cell Variability and Diverse Datasets**

Data-driven strategies are also crucial for achieving universality. Research has shown that a multi-battery training approach, which includes data from many fully aged cells in the training set, allows a model to learn a more general representation of the entire degradation trajectory, significantly improving early-life prediction accuracy.1 A recent key finding is that incorporating *inter-cell feature differences* as a model input, rather than relying solely on single-cell characteristics, significantly increases prediction accuracy and cross-condition robustness.22 This technique allows the model to learn from the variability itself, breaking the learning boundaries between different aging conditions. The success of these data-centric approaches is critically dependent on the availability of large-scale, high-quality, and diverse public datasets.17

## **8\. Strategic Synthesis and Future Outlook**

### **8.1. A Consolidated View of the Technology Landscape**

The analysis of battery degradation modeling reveals a clear and rapid evolutionary trajectory. The field has progressed from discrete black-box models (RNNs), which impose an artificial structure on a continuous process, to continuous black-box models (NODEs), which offer a more physically analogous framework with superior performance. The current state of the art is now advancing towards continuous "grey-box" models that are physics-informed, interpretable, and designed for universality. This progression reflects a growing acknowledgment that while data is powerful, it is insufficient for building the truly robust and trustworthy prognostic tools required for safety-critical industrial applications.

### **8.2. Recommendations for Research & Development Priorities**

1. **Prioritize Physics-Informed Models:** The primary research focus should shift from purely data-driven models to the development of universal, physics-informed neural ODEs. Integrating physical constraints via PINN and UDE methodologies is the most promising path to creating models that are robust, data-efficient, and can generalize across a wide range of battery chemistries and operating conditions.1  
2. **Invest in Interpretable Architectures:** To build trust and facilitate adoption in safety-critical applications, research into inherently interpretable dynamics learners, such as KANs and attention-based transformers, should be accelerated. The ability to inspect and understand the learned degradation function is a crucial next step for the field.1  
3. **Build Foundational Datasets:** Foster the creation of large, diverse, and standardized public datasets. These are essential for training the next generation of universal "foundation" models and should ideally include not just operational data but also material properties and post-mortem analysis to link macro-level behavior to micro-level mechanisms.13

### **8.3. Recommendations for Industrial Application and BMS Integration**

1. **Adopt a Phased Implementation Approach:** While complex models like ACLA demonstrate state-of-the-art performance, simpler and more computationally efficient variants like PC-RNN or a well-regularized NODE may be more suitable for initial deployment on resource-constrained embedded systems in current-generation BMS.1 A technology roadmap should be planned to incorporate more complex hybrid and SciML models as edge computing capabilities improve.  
2. **Leverage Conservative EOL Predictions as a Safety Feature:** A key operational advantage of some ODE-based models is their tendency to provide conservative (pessimistic) EOL estimates. This should be treated as a valuable safety feature, enabling more prudent maintenance scheduling and reducing the risk of unexpected battery failure.1  
3. **Standardize Data Collection and Feature Engineering:** The performance of all models is highly dependent on the quality of the input data. Industrial efforts should focus on standardized data collection protocols, especially for the information-rich constant-current (CC) charging phase, as this provides the consistent, high-quality features required for the most advanced prognostic techniques.1

#### **Works cited**

1. NODE Battery Degradation Modeling Research.pdf  
2. A Scientific Machine Learning Approach for Predicting and Forecasting Battery Degradation in Electric Vehicles \- arXiv, accessed October 17, 2025, [https://arxiv.org/html/2410.14347v1](https://arxiv.org/html/2410.14347v1)  
3. A Scientific Machine Learning Approach for Predicting and Forecasting Battery Degradation in Electric Vehicles \- arXiv, accessed October 17, 2025, [https://arxiv.org/pdf/2410.14347](https://arxiv.org/pdf/2410.14347)  
4. Physics-Informed Neural Networks for Prognostics and Health Management of Lithium-Ion Batteries \- Semantic Scholar, accessed October 17, 2025, [https://www.semanticscholar.org/paper/Physics-Informed-Neural-Networks-for-Prognostics-of-Wen-Ye/93ce4360bdfe3ccbba5aaffa5a5691f6130f292f](https://www.semanticscholar.org/paper/Physics-Informed-Neural-Networks-for-Prognostics-of-Wen-Ye/93ce4360bdfe3ccbba5aaffa5a5691f6130f292f)  
5. A Scientific Machine Learning Approach for Predicting and ..., accessed October 17, 2025, [https://arxiv.org/abs/2410.14347](https://arxiv.org/abs/2410.14347)  
6. Neural ordinary differential equations and recurrent Neural Networks for Predicting the State of Health of Batteries \- ResearchGate, accessed October 17, 2025, [https://www.researchgate.net/publication/358626669\_Neural\_ordinary\_differential\_equations\_and\_recurrent\_Neural\_Networks\_for\_Predicting\_the\_State\_of\_Health\_of\_Batteries](https://www.researchgate.net/publication/358626669_Neural_ordinary_differential_equations_and_recurrent_Neural_Networks_for_Predicting_the_State_of_Health_of_Batteries)  
7. Diagnostic forecasting of battery degradation through contrastive learning \- PMC, accessed October 17, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12480837/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12480837/)  
8. Modeling of Lithium-Ion Battery Degradation for Cell Life Assessment \- ResearchGate, accessed October 17, 2025, [https://www.researchgate.net/publication/303890624\_Modeling\_of\_Lithium-Ion\_Battery\_Degradation\_for\_Cell\_Life\_Assessment](https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment)  
9. Physics-informed neural networks for a Lithium-ion batteries model: A case of study, accessed October 17, 2025, [https://www.aimsciences.org/article/doi/10.3934/acse.2024018](https://www.aimsciences.org/article/doi/10.3934/acse.2024018)  
10. Interpretable Battery Lifetime Prediction Using Early Degradation Data \- Chalmers Research, accessed October 17, 2025, [https://research.chalmers.se/publication/535712/file/535712\_Fulltext.pdf](https://research.chalmers.se/publication/535712/file/535712_Fulltext.pdf)  
11. A Novel Neural-Ode Model for the State of Health Estimation of Lithium-Ion Battery Using Charging Curve \- ResearchGate, accessed October 17, 2025, [https://www.researchgate.net/publication/391581592\_A\_Novel\_Neural-Ode\_Model\_for\_the\_State\_of\_Health\_Estimation\_of\_Lithium-Ion\_Battery\_Using\_Charging\_Curve](https://www.researchgate.net/publication/391581592_A_Novel_Neural-Ode_Model_for_the_State_of_Health_Estimation_of_Lithium-Ion_Battery_Using_Charging_Curve)  
12. ACCEPT: Diagnostic Forecasting of Battery Degradation Through Contrastive Learning, accessed October 17, 2025, [https://arxiv.org/html/2501.10492v1](https://arxiv.org/html/2501.10492v1)  
13. (PDF) Interpretable Data-Driven Modeling Reveals Complexity of Battery Aging, accessed October 17, 2025, [https://www.researchgate.net/publication/370484095\_Interpretable\_Data-Driven\_Modeling\_Reveals\_Complexity\_of\_Battery\_Aging](https://www.researchgate.net/publication/370484095_Interpretable_Data-Driven_Modeling_Reveals_Complexity_of_Battery_Aging)  
14. (PDF) A Scientific Machine Learning Approach for Predicting and Forecasting Battery Degradation in Electric Vehicles \- ResearchGate, accessed October 17, 2025, [https://www.researchgate.net/publication/385091466\_A\_Scientific\_Machine\_Learning\_Approach\_for\_Predicting\_and\_Forecasting\_Battery\_Degradation\_in\_Electric\_Vehicles](https://www.researchgate.net/publication/385091466_A_Scientific_Machine_Learning_Approach_for_Predicting_and_Forecasting_Battery_Degradation_in_Electric_Vehicles)  
15. Physics-informed neural network for lithium-ion battery degradation ..., accessed October 17, 2025, [https://ideas.repec.org/a/nat/natcom/v15y2024i1d10.1038\_s41467-024-48779-z.html](https://ideas.repec.org/a/nat/natcom/v15y2024i1d10.1038_s41467-024-48779-z.html)  
16. Physics-Informed Neural Networks for Advanced Thermal Management in Electronics and Battery Systems: A Review of Recent Developments and Future Prospects \- MDPI, accessed October 17, 2025, [https://www.mdpi.com/2313-0105/11/6/204](https://www.mdpi.com/2313-0105/11/6/204)  
17. Battery Prognostics and Health Management: AI and Big Data \- MDPI, accessed October 17, 2025, [https://www.mdpi.com/2032-6653/16/1/10](https://www.mdpi.com/2032-6653/16/1/10)  
18. Interpretable Deep Learning Using Temporal Transformers for Battery Degradation Prediction \- Preprints.org, accessed October 17, 2025, [https://www.preprints.org/manuscript/202505.1044/v1](https://www.preprints.org/manuscript/202505.1044/v1)  
19. A Perspective on Inverse Design of Battery Interphases using Multi-scale Modelling, Experiments and Generative Deep Learning, accessed October 17, 2025, [https://orbit.dtu.dk/files/195524959/1\_s2.0\_S2405829719302193\_main.pdf](https://orbit.dtu.dk/files/195524959/1_s2.0_S2405829719302193_main.pdf)  
20. A Universal Machine Learning Framework Driven by Artificial Intelligence for Ion Battery Cathode Material Design | JACS Au \- ACS Publications, accessed October 17, 2025, [https://pubs.acs.org/doi/10.1021/jacsau.5c00526](https://pubs.acs.org/doi/10.1021/jacsau.5c00526)  
21. A Universal Machine Learning Framework Driven by Artificial Intelligence for Ion Battery Cathode Material Design | JACS Au \- ACS Publications, accessed October 17, 2025, [https://pubs.acs.org/doi/abs/10.1021/jacsau.5c00526](https://pubs.acs.org/doi/abs/10.1021/jacsau.5c00526)  
22. \[2310.05052\] Accurate battery lifetime prediction across diverse aging conditions with deep learning \- arXiv, accessed October 17, 2025, [https://arxiv.org/abs/2310.05052](https://arxiv.org/abs/2310.05052)  
23. End-to-End Framework for Predicting the Remaining Useful Life of Lithium-Ion Batteries, accessed October 17, 2025, [https://arxiv.org/html/2505.16664v2](https://arxiv.org/html/2505.16664v2)  
24. State of Health Prognostics for Series Battery Packs: A Universal Deep Learning Method, accessed October 17, 2025, [https://www.researchgate.net/publication/354079952\_State\_of\_Health\_Prognostics\_for\_Series\_Battery\_Packs\_A\_Universal\_Deep\_Learning\_Method](https://www.researchgate.net/publication/354079952_State_of_Health_Prognostics_for_Series_Battery_Packs_A_Universal_Deep_Learning_Method)