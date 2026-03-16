# Mini AI Model Trainer Framework

A Python project that simulates how AI models are configured, trained, and evaluated — built to demonstrate all core Object-Oriented Programming (OOP) concepts.

---

## What This Project Does

This framework mimics a simplified version of how real ML libraries like PyTorch or scikit-learn organize their training pipelines. It includes:

- A **configuration object** that stores model hyper-parameters
- An **abstract base class** that enforces a common interface for all models
- Two **concrete model classes** (Linear Regression and Neural Network) with different training and evaluation behaviour
- A **DataLoader** class that holds a dataset independently
- A **Trainer** class that orchestrates the full pipeline: load data → train → evaluate

---

## Project Structure

```
AI_Model_Trainer/
│
├── ai_trainer_framework.py    ← all code lives here
└── README.md
```

---

## How to Run

No installations needed — this project only uses Python's built-in `abc` module.

**Step 1 — Open your terminal and navigate to the project folder:**
```bash
cd mini_ai_trainer
```

**Step 2 — Run the file:**
```bash
python ai_trainer_framework.py
```

If that doesn't work, try:
```bash
python3 ai_trainer_framework.py
```

---

## Expected Output

```
[Config] LinearRegression | lr=0.01 | epochs=10
[Config] NeuralNetwork    | lr=0.001 | epochs=20
Models created: 2

--- Training LinearRegression ---
LinearRegression: Training on 5 samples for 10 epochs (lr=0.01)
LinearRegression: Evaluation MSE = 0.042

--- Training NeuralNetwork ---
NeuralNetwork [64, 32, 1]: Training on 5 samples for 20 epochs (lr=0.001)
NeuralNetwork: Evaluation Accuracy = 91.5%
```

---

## OOP Concepts Demonstrated

| Concept | Where Applied |
|---|---|
| Class Attribute | `BaseModel.total_models` — counts all model instances |
| Instance Attributes | `ModelConfig.learning_rate`, `model.config`, etc. |
| Abstraction (ABC) | `BaseModel` with `@abstractmethod` on `train()` and `evaluate()` |
| Single Inheritance | `LinearRegressionModel` and `NeuralNetworkModel` both inherit `BaseModel` |
| Method Overriding | Both subclasses override `train()` and `evaluate()` differently |
| `super()` | Both child `__init__` methods call `super().__init__()` |
| Polymorphism | `Trainer.run()` works with any `BaseModel` subclass unchanged |
| Composition | `BaseModel` owns a `ModelConfig` instance |
| Aggregation | `DataLoader` is created externally and passed into `Trainer` |
| Magic Method | `__repr__` on `ModelConfig` for clean print output |
| Instance Methods | `train()`, `evaluate()`, `run()`, `fetch()` |

---

## Class Overview

### `ModelConfig`
Stores hyper-parameters for a model: `model_name`, `learning_rate`, and `epochs`. Implements `__repr__` for readable output. Used as a composition object inside `BaseModel`.

### `BaseModel` (Abstract)
Abstract base class that defines the interface all models must follow. Contains the class attribute `total_models` to track how many model instances have been created. Forces subclasses to implement `train()` and `evaluate()`.

### `LinearRegressionModel`
Concrete subclass of `BaseModel`. Simulates training a linear regression model and reports a fixed MSE score on evaluation.

### `NeuralNetworkModel`
Concrete subclass of `BaseModel`. Has an extra `layers` attribute (list of integers representing layer sizes). Reports accuracy on evaluation instead of MSE — showing polymorphic behaviour.

### `DataLoader`
Standalone class that holds a dataset. Passed into `Trainer` from outside — demonstrating aggregation (Trainer uses it but does not own it).

### `Trainer`
Orchestrates the full training pipeline. Accepts any `BaseModel` and a `DataLoader`. Its `run()` method calls `train()` then `evaluate()` — working correctly for any model type (polymorphism).

---

## Requirements

- Python 3.10+
- No external libraries required
