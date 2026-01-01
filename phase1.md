# Phase-1: Protein Function Prediction Model

## Problem Formulation

Phase-1 is formulated as a **binary protein classification task**, where each protein sequence is categorized into one of two classes:

- **Enzyme (positive class):** Proteins associated with one or more Enzyme Commission (EC) numbers
- **Non-enzyme (negative class):** Proteins without any EC annotations

This formulation is biologically meaningful, well-established in protein function prediction literature, and suitable for evaluating sequence-based neural architectures.

---

## Dataset

The dataset is derived from **Swiss-Prot**, the manually reviewed subset of UniProt, ensuring high annotation quality and minimal label noise.

**Dataset characteristics:**

- **Input:** Raw amino-acid sequences
- **Labels:** Binary enzyme labels derived from curated EC annotations
- **Labeling rule:**
  - Proteins with a non-null EC number are labeled as **enzymes**
  - Proteins without EC annotations are labeled as **non-enzymes**

**Dataset properties:**

- Balanced class distribution (~50% enzyme / ~50% non-enzyme)
- Sequence length filtering and truncation applied for computational stability
- Final dataset sampled to ensure efficient and stable training

This dataset choice avoids annotation sparsity issues common in large multi-label datasets while maintaining realistic biological complexity.

---

## Model Architecture

Phase-1 employs a **hybrid CNN–BiLSTM architecture** designed to capture both local sequence motifs and long-range dependencies in protein sequences.

**Architecture components:**

1. **Embedding Layer** – learns dense representations of amino acids
2. **1D Convolutional Layer** – captures local biochemical motifs
3. **Max Pooling** – reduces sequence dimensionality
4. **Bidirectional LSTM** – models long-range contextual dependencies
5. **Global Max Pooling** – aggregates sequence-level features
6. **Fully Connected Layers** – perform final binary classification

The model outputs a single probability value representing the likelihood of a protein being an enzyme.

---

## Training Strategy

- **Loss function:** Binary cross-entropy
- **Optimizer:** Adam
- **Primary evaluation metric:** Accuracy (appropriate due to balanced classes)
- **Validation strategy:** Held-out test split for generalization assessment

The model exhibits rapid and stable convergence, achieving over **80% validation accuracy within the first epoch** and exceeding **85% validation accuracy after a small number of epochs**, indicating effective learning of biologically relevant sequence patterns.

---

## Motivation for Phase-1 Design

Phase-1 functions as a **validation stage for sequence representation learning**. Strong performance on curated Swiss-Prot data confirms that:

- The model learns meaningful protein representations
- The proposed architecture is stable and effective
- Performance metrics are interpretable and scientifically defensible

This phase establishes a reliable foundation for subsequent, more complex protein function prediction tasks.
