# NeuroSleep Analytics: EEG-Based Sleep Stage Classification using Machine Learning methods

## Overview

This project investigates whether sleep stages can be accurately classified using EEG recordings alone. The study is motivated by the need for practical evidence on whether automated EEG-based sleep staging is reliable enough to support future internal research and product development.

Using the Sleep-EDF Expanded dataset, we compare three modeling approaches:

- **SG Boost** as a baseline model
- **Attention-based model**
- **U-Sleep**

The goal is not only to measure predictive performance, but also to determine whether the added complexity of advanced deep learning approaches provides meaningful practical value over a simpler baseline.

---

## Research Question

**Can sleep stages be accurately classified using EEG recordings, and which of SG Boost, Attention, and U-Sleep performs best?**

This project also explores three related questions:

1. Does EEG alone provide enough information for useful sleep stage classification?
2. Do advanced deep learning models outperform a simpler baseline by a practically meaningful margin?
3. Which sleep stages are most often confused, and what do these errors suggest about the limits of EEG-only classification?

---

## Motivation

Manual sleep staging is time-consuming, costly, and dependent on trained specialists. A reliable automated approach could:

- Reduce scoring time
- Improve workflow efficiency
- Support future sleep analysis tools
- Guide internal decision-making on model development

Even if results are not strong enough for deployment, the comparison can still reveal which modeling strategy is most promising and which sleep stages remain difficult to classify.

---

## Dataset

This project uses the **Sleep-EDF Expanded** dataset accessed through **MNE-Python**.

### Data used
- Overnight polysomnography recordings
- Expert-labeled hypnograms
- Two EEG channels:
  - `EEG Fpz-Cz`
  - `EEG Pz-Oz`
- Sampling rate: **100 Hz**

These data allow us to examine whether EEG-only signals contain enough information for accurate sleep stage classification.

---

## Methodology

### 1. Preprocessing

Each recording will be segmented into **30-second epochs** aligned with the sleep stage annotations.

Planned preprocessing steps include:

- Signal filtering
- Normalization
- Epoch segmentation
- Label preparation and mapping

Because some original sleep stages are closely related, a practical label mapping will be defined based on common sleep staging practice.

---

### 2. Models Compared

#### SG Boost
A baseline approach using hand-engineered features extracted from each epoch, such as:

- Summary statistics
- Frequency-domain features

This model serves as an efficient and more interpretable baseline.

#### Attention Model
A deep learning approach that uses EEG signals more directly and learns temporal patterns automatically.

#### U-Sleep
A more advanced architecture designed for automated sleep staging from EEG data, also learning patterns directly from the signal.

---

## Evaluation

The models will be trained and evaluated using:

- Training set
- Validation set
- Test set

Where possible, **subject-level splitting** will be used to better reflect generalization to unseen individuals.

### Metrics
Because sleep stage classification is a multiclass problem with class imbalance, we will evaluate models using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Accuracy alone is not sufficient, since it may hide poor performance on more difficult or less frequent classes.

---

## Expected Outcomes

This study aims to provide decision-relevant evidence for whether EEG-only sleep staging is valuable enough to justify further development.

Possible outcomes include:

- If one of the advanced models clearly outperforms the baseline, further investment in deep learning methods may be justified.
- If the gains are small, a simpler approach may be preferable because it is faster to train, easier to explain, and easier to maintain.
- In either case, the study will help identify which sleep stages are most difficult to classify and where EEG-only approaches may have limitations.

---

## Project Timeline

Estimated timeline:

- **2 days**: preprocessing and label preparation
- **2 days**: feature extraction and SG Boost baseline
- **3–4 days**: training Attention and U-Sleep models
- **2 days**: evaluation, interpretation, and report writing

Because the dataset is public and already labeled, the main challenge is analysis rather than data collection, making the project feasible within the available timeline.

---

## Significance

This project addresses a practical question with direct value for internal decision-making. By the end of the study, we aim to determine:

- Whether EEG can support accurate sleep stage classification
- Whether advanced models improve meaningfully over a simpler baseline
- Whether the results justify further internal development

---

## Repository Structure

```bash
.
├── data/               # Raw or processed dataset files
├── notebooks/          # Exploratory analysis and experiments
├── src/                # Model training, preprocessing, and evaluation code
├── results/            # Metrics, figures, and confusion matrices
├── reports/            # Final write-up or presentation materials
└── README.md
