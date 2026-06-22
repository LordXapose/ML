# Sub-task 5: Gradient Boosting & the Learning Rate vs Number of Trees Trade-off

## Overview

This project investigates the trade-off between **learning rate** and **number of boosting trees** in a Gradient Boosting model using Scikit-Learn's `HistGradientBoostingClassifier`.

The objective was to tune three key hyperparameters and demonstrate how a smaller learning rate generally requires more boosting iterations to reach optimal performance.

The model was evaluated using **Average Precision (PR-AUC)**, the metric used for the Sub-task 6 leaderboard.

---

## Hyperparameters Tuned

The following parameters were explored:

| Parameter      | Values Tested    |
| -------------- | ---------------- |
| learning_rate  | 0.05, 0.10, 0.20 |
| max_iter       | 100, 300, 600    |
| max_leaf_nodes | 15, 31, 63       |

The full experiment evaluated:

```text
3 × 3 × 3 = 27 configurations
```

using 5-fold cross-validation.

---

## Dataset

Instead, experiments were performed using a realistic UNSW-NB15 dataset derived from the Kaggle archive:

* Based on UNSW-NB15 training and testing sets
* Resampled to 20,000 rows
* Maintained approximately 10% attack rate
* Matches the shape and class imbalance described in the S3 specification

This dataset was saved as:

```text
adt_s3_unsw_clean_realistic.csv
```

---

## Evaluation Metric

The scoring metric used throughout the experiment was:

```python
average_precision
```

Average Precision (PR-AUC) was selected because:

* It is the official leaderboard metric.
* It is more informative than ROC-AUC under class imbalance.
* It better reflects attack-detection performance.

---

# Experiment 1: Main Hyperparameter Sweep

The primary sweep followed the assignment requirements exactly:

```python
HistGradientBoostingClassifier(
    early_stopping=True
)
```

### Top Results

| Learning Rate | Max Iter | Max Leaf Nodes | Average Precision |
| ------------- | -------- | -------------- | ----------------- |
| 0.05          | 300      | 15             | 0.9413            |
| 0.05          | 600      | 15             | 0.9413            |
| 0.10          | 100      | 15             | 0.9406            |
| 0.05          | 100      | 15             | 0.9404            |
| 0.10          | 300      | 15             | 0.9399            |

### Best Configuration

```text
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 15

Cross-Validated Average Precision = 0.9413
Held-Out Test Average Precision = 0.9437
```

### Observation

Scores remained almost unchanged across different `max_iter` values.

This occurred because early stopping converged before 100 trees for nearly every configuration.

As a result:
It stopped very early. It got a 94% score, but you couldn't see the Tortoise catching up because the computer quit too soon.

```text
max_iter became a ceiling rather than the actual tree count.
```

The model stopped early and never used most of the additional trees.

---

# Experiment 2: Early Stopping Disabled

To investigate the effect of tree count directly:

```python
early_stopping=False
```

was used while keeping:

```text
max_iter = [100, 300, 600]
max_leaf_nodes = 15
```

### Results

| Learning Rate | 100 Trees | 300 Trees | 600 Trees |
| ------------- | --------- | --------- | --------- |
| 0.05          | 0.9418    | 0.9416    | 0.9394    |
| 0.10          | 0.9422    | 0.9405    | 0.9387    |
| 0.20          | 0.9402    | 0.9379    | 0.9370    |

### Observation

Performance decreased as more trees were added.
Scores got worse because it was overthinking and memorizing the practice data. Too many steps hurt the fast learner

This indicates that:

```text
100 trees was already near the optimum.
```

Adding additional trees pushed the model into overfitting.

---

# Experiment 3: Making the Trade-off Visible

The assignment specifically required showing that:

> A lower learning rate only becomes competitive when the number of trees is increased.

To reveal this behavior, the experiment moved into an under-trained regime:

```text
learning_rate = [0.02, 0.05]
max_iter = [20, 50, 150]
max_leaf_nodes = 31
early_stopping = False
```

### Results

| Learning Rate | 20 Trees | 50 Trees | 150 Trees |
| ------------- | -------- | -------- | --------- |
| 0.02          | 0.9195   | 0.9301   | 0.9401    |
| 0.05          | 0.9301   | 0.9371   | 0.9414    |

### Trade-off Analysis

Difference between learning rates:

| Trees | Gap    |
| ----- | ------ |
| 20    | 0.0106 |
| 50    | 0.0070 |
| 150   | 0.0013 |

The smaller learning rate starts behind but steadily catches up as additional trees are added.
 Finally! The Tortoise starts way behind, but by 150 steps, it has almost caught up to the Hare.

This is the exact learning-rate versus number-of-trees trade-off predicted by Gradient Boosting theory.

---

# Key Findings

### 1. Best Model

```text
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 15

CV Average Precision = 0.9413
Test Average Precision = 0.9437
```

### 2. Early Stopping Dominates

With early stopping enabled:

* Most models converged before 100 trees.
* Increasing `max_iter` beyond 100 produced almost no benefit.

### 3. Lower Learning Rates Need More Trees

The dedicated trade-off experiment showed:

```text
Small learning rates learn more slowly.
```

However:

```text
Given enough trees, they catch up to larger learning rates.
```

### 4. Shallower Trees Performed Best

The best configuration used:

```text
max_leaf_nodes = 15
```

Smaller trees worked better because boosting benefits from combining many weak learners rather than a few highly complex trees.

---

# Conclusion

Prove the trade-off: **small learning rate (tortoise)** needs **more trees (steps)** to catch up to the **fast learner (hare)**.

**The Setup**  
- 20k rows, 10% attack rate.  
- Scoring = Average Precision (handles imbalance).  
- Trees kept simple (`max_leaf_nodes=15`) to avoid overcomplicating each step.

**The 3 Runs in 3 Bullets**  
1. **Main sweep (early stopping ON)**: Best score = **0.941**. Trade-off *hidden* because early stopping cut trees too soon.  
2. **Forced 100–600 trees (early stopping OFF)**: Scores *dropped* → too many trees caused overfitting.  
3. **Lowered range 20–150 trees**: Tortoise (lr=0.02) starts behind Hare (lr=0.05), but **catches up** at 150 trees (gap shrinks from 0.010 to 0.001). **Trade-off proven.**

**The Final Takeaway**  
The trade-off exists, just at a **smaller tree scale** than the brief's example suggested. Slow-and-steady wins when given enough steps.
This project successfully demonstrated the classic Gradient Boosting trade-off between learning rate and number of trees.

The best-performing model achieved an Average Precision score of **0.9413** using:

```text
learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 15
```

A dedicated companion experiment further confirmed that smaller learning rates require more boosting iterations to reach comparable performance, providing direct evidence of the learning-rate versus number-of-trees relationship.
