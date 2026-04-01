# Math4AI Capstone: Project Roadmap

This roadmap outlines the 7-day execution plan for the Math4AI Final Capstone. Each phase is mapped to the requirements specified in `deliverables/math4ai_capstone_assignment.tex`.

---

## 📅 Day 1: Data Foundations & Environment Setup
- [ ] **Data Verification:** Use `starter_pack/scripts/visualize_data.py` to confirm shapes of all `.npz` files.
- [ ] **Split Logic:** Confirm `digits_split_indices.npz` correctly maps to `digits_data.npz`.
- [ ] **Repository Setup:** Initialize your team's Git workflow (feature branches for each member).

---

## 📅 Day 2: The Baseline — Softmax Regression
- [ ] **Implementation (from scratch):**
    - [ ] Numerically stable `softmax(z)`.
    - [ ] Forward pass: $s = Wx + b$.
    - [ ] Loss: Cross-entropy $-\sum y \log(p)$.
    - [ ] Gradients: $\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n}(P-Y)^\top X$.
- [ ] **Sanity Check:** Successfully overfit 5 examples of the Gaussian task.
- [ ] **Task:** Train on `linear_gaussian.npz` and plot the decision boundary.

---

## 📅 Day 3: Representation Learning — One-Hidden-Layer NN
- [ ] **Mathematical Derivation:** Complete the manual derivation of gradients for the $W_1, b_1, W_2, b_2$ parameters.
- [ ] **Implementation:**
    - [ ] `tanh(z)` and its derivative $1 - \tanh^2(z)$.
    - [ ] Vectorized backpropagation.
    - [ ] $L_2$ regularization ($\lambda = 10^{-4}$).
- [ ] **Sanity Check:** Perform a gradient check comparing analytical vs. finite-difference gradients.
- [ ] **Task:** Train on `moons.npz` and compare with the linear baseline.

---

## 📅 Day 4: Core Experiments & Visualization
- [ ] **Standardized Protocol:**
    - [ ] Batch size: 64.
    - [ ] Learning rate: 0.05 (SGD).
    - [ ] Epoch budget: 200.
- [ ] **Visualization:**
    - [ ] Decision boundary plots for both synthetic tasks.
    - [ ] Training dynamics (Loss vs. Epoch) for the Digits benchmark.
- [ ] **Goal:** Collect evidence showing where the hidden layer changes the problem's geometry.

---

## 📅 Day 5: Required Ablations & Statistics
- [ ] **Capacity Ablation (Moons):** Train NN with widths $\{2, 8, 32\}$.
- [ ] **Optimizer Study (Digits):** 
    - [ ] Implement **Momentum** ($\mu=0.9$).
    - [ ] Implement **Adam** ($\alpha=0.001, \beta_1=0.9, \beta_2=0.999$).
- [ ] **Repeated-Seed Protocol:** 
    - [ ] Run the final chosen configuration 5 times with different seeds.
    - [ ] Calculate means and 95% Confidence Intervals for Accuracy and Cross-Entropy.

---

## 📅 Day 6: Advanced Track & Failure Analysis
- [ ] **Advanced Track (Choose One):**
    - [ ] **Track A:** PCA/SVD analysis (Scree plot, 2D projection, accuracy vs. dimensions).
    - [ ] **Track B:** Prediction Reliability (Confidence vs. Accuracy, predictive entropy).
- [ ] **Failure Case Analysis:** Identify one specific failure (e.g., instability, under-capacity) and explain the underlying mechanism.

---

## 📅 Day 7: Final Synthesis & Deliverables
- [ ] **Report Drafting (6-8 pages):** 
    - [ ] Focus on the "Interpretive Questions" in the handout.
    - [ ] Document all required implementation sanity checks.
- [ ] **Presentation:** Prepare 10-minute technical pitch and slides.
- [ ] **Final Repository Polish:** Update `README.md` with exact reproduction instructions.

---

## 🛠 Required Implementation Checklist (NumPy Only)
- [ ] Numerically stable softmax
- [ ] Cross-entropy loss
- [ ] Forward/Backward propagation
- [ ] Mini-batch SGD/Momentum/Adam
- [ ] Evaluation metrics (Accuracy, Mean Cross-Entropy)
- [ ] Decision boundary plotting logic
