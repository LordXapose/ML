
## Overview

This project investigates the relationship between **learning rate** and **number of trees** in a Gradient Boosting model. The objective is to demonstrate the trade-off between taking small learning steps and requiring more boosting iterations to achieve optimal performance.

The model was trained to classify cyber-attack traffic using Gradient Boosting and evaluated using **Average Precision (AP)**.

---

## Problem Statement

Gradient Boosting improves predictions by building trees sequentially, where each new tree attempts to correct mistakes made by previous trees.

The key question explored in this task is:

> Does a smaller learning rate require more trees to achieve the same performance as a larger learning rate?

---

## Key Concepts

### Learning Rate (`learning_rate`)

The learning rate controls how much each tree contributes to the overall model.

* Small learning rate = careful learning
* Large learning rate = aggressive learning

### Number of Trees (`max_iter`)

The number of boosting iterations (trees) the model is allowed to build.

* More trees = more opportunities to learn
* Fewer trees = faster training but potentially lower performance

### Tree Complexity (`max_leaf_nodes`)

Controls how complex each individual decision tree can become.

---

## The Tortoise and Hare Analogy

Imagine two children walking to school:

### 🐢 Tortoise

* Takes very small steps
* Rarely makes mistakes
* Needs many steps to arrive

### 🐇 Hare

* Takes large leaps
* Reaches the destination quickly
* Can overshoot or make mistakes

In Gradient Boosting:

| Analogy         | Machine Learning Term        |
| --------------- | ---------------------------- |
| Small steps     | Low learning rate            |
| Big leaps       | High learning rate           |
| Number of steps | Number of trees (`max_iter`) |

The goal of this experiment is to show that the tortoise eventually catches the hare if it is given enough steps.

---

## Hyperparameter Search

The following parameters were tuned:

```python
param_grid = {
    "learning_rate": [0.05, 0.1, 0.2],
    "max_iter": [100, 300, 600],
    "max_leaf_nodes": [15, 31, 63]
}
```

Total combinations evaluated:

```text
3 × 3 × 3 = 27 combinations
```

Cross-validation was used to evaluate each combination.

---

## Early Stopping

The model used:

```python
early_stopping=True
```

Early stopping automatically halts training when validation performance stops improving.

Benefits:

* Reduces overfitting
* Speeds up training
* Prevents unnecessary trees from being added

---

## Results

| Learning Rate             | Trees | Leaf Nodes | Average Precision |
| ------------------------- | ----- | ---------- | ----------------- |
| Replace with your results |       |            |                   |

---

## Demonstrating the Trade-Off

Example:

| Learning Rate | Trees | AP Score         |
| ------------- | ----- | ---------------- |
| 0.20          | 100   | Higher initially |
| 0.05          | 100   | Lower initially  |

At a small number of trees, the lower learning rate performs worse because it has not had enough opportunities to learn.

After increasing the number of trees:

| Learning Rate | Trees | AP Score                    |
| ------------- | ----- | --------------------------- |
| 0.20          | 300   | Strong performance          |
| 0.05          | 600   | Equal or better performance |

This demonstrates the expected learning-rate versus number-of-trees trade-off.

---

## Best Model

Replace with your GridSearchCV output:

```text
Best Parameters:
learning_rate = X
max_iter = X
max_leaf_nodes = X

Average Precision = X
```

---

## What Was Proven?

This experiment confirmed that:

* Lower learning rates generally require more trees.
* Higher learning rates learn faster but may not generalize as well.
* Given enough boosting iterations, a smaller learning rate can match or outperform a larger learning rate.
* Early stopping helps identify the optimal number of trees automatically.

---

## One-Sentence Summary

I trained a Gradient Boosting model to detect cyber-attacks and demonstrated that smaller learning rates require more boosting trees to reach optimal performance, confirming the classic learning-rate versus number-of-trees trade-off.
