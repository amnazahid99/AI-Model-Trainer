"""
Mini AI Model Trainer Framework
OOP concepts covered: class attributes, instance attributes, abstraction (ABC),
single inheritance, method overriding, super(), polymorphism,
composition, aggregation, __repr__, and instance methods.
"""

from abc import ABC, abstractmethod


# ──────────────────────────────────────────────
# ModelConfig  →  composition object
# ──────────────────────────────────────────────
class ModelConfig:
    """Holds hyper-parameter settings for a model. Embedded inside BaseModel via composition."""

    def __init__(self, name: str, lr: float = 0.01, epochs: int = 10):
        self.model_name    = name      # instance attribute
        self.learning_rate = lr        # instance attribute
        self.epochs        = epochs    # instance attribute

    def __repr__(self) -> str:         # magic method
        return (
            f"[Config] {self.model_name:<15} | "
            f"lr={self.learning_rate} | "
            f"epochs={self.epochs}"
        )


# ──────────────────────────────────────────────
# BaseModel  →  abstract base class
# ──────────────────────────────────────────────
class BaseModel(ABC):
    """
    Defines the shared interface every model must follow.
    Tracks how many model objects have been created via a class attribute.
    """

    total_models: int = 0    # class attribute — shared by all subclasses

    def __init__(self, cfg: ModelConfig):
        self.config = cfg                  # composition — BaseModel owns a ModelConfig
        BaseModel.total_models += 1        # increment class attribute on each creation

    @abstractmethod
    def train(self, data) -> None:
        """Subclasses must implement their own training logic."""
        ...

    @abstractmethod
    def evaluate(self, data) -> None:
        """Subclasses must implement their own evaluation logic."""
        ...


# ──────────────────────────────────────────────
# LinearRegressionModel  →  concrete subclass
# ──────────────────────────────────────────────
class LinearRegressionModel(BaseModel):
    """
    Inherits BaseModel (single inheritance).
    Overrides train() and evaluate() with regression-specific behaviour.
    Calls parent __init__ through super().
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)     # super() — passes cfg up to BaseModel.__init__

    def train(self, data) -> None:          # method overriding
        print(
            f"LinearRegression: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs "
            f"(lr={self.config.learning_rate})"
        )

    def evaluate(self, data) -> None:       # method overriding
        print("LinearRegression: Evaluation MSE = 0.042")


# ──────────────────────────────────────────────
# NeuralNetworkModel  →  concrete subclass
# ──────────────────────────────────────────────
class NeuralNetworkModel(BaseModel):
    """
    Inherits BaseModel (single inheritance).
    Adds an extra instance attribute `layers`.
    Overrides train() and evaluate() differently — this is polymorphism in action.
    """

    def __init__(self, cfg: ModelConfig, layers: list[int] = None):
        super().__init__(cfg)     # super() — passes cfg up to BaseModel.__init__
        self.layers = layers if layers is not None else [64, 32, 1]  # extra instance attribute

    def train(self, data) -> None:          # method overriding
        print(
            f"NeuralNetwork {self.layers}: Training on {len(data)} samples "
            f"for {self.config.epochs} epochs "
            f"(lr={self.config.learning_rate})"
        )

    def evaluate(self, data) -> None:       # method overriding
        print("NeuralNetwork: Evaluation Accuracy = 91.5%")


# ──────────────────────────────────────────────
# DataLoader  →  independent class (aggregation)
# ──────────────────────────────────────────────
class DataLoader:
    """
    Stores a dataset on its own — not tied to any model.
    Passed into Trainer from outside, so Trainer doesn't own it (aggregation).
    """

    def __init__(self, records: list):
        self.records = records             # instance attribute

    def fetch(self) -> list:               # instance method
        return self.records


# ──────────────────────────────────────────────
# Trainer  →  orchestrator (uses aggregation)
# ──────────────────────────────────────────────
class Trainer:
    """
    Receives any BaseModel and a DataLoader from outside (aggregation — does not own them).
    run() is polymorphic: the same method works correctly for any BaseModel subclass.
    """

    def __init__(self, model: BaseModel, loader: DataLoader):
        self.model  = model    # aggregation
        self.loader = loader   # aggregation

    def run(self) -> None:     # instance method
        """Full pipeline: fetch data → train → evaluate."""
        dataset    = self.loader.fetch()
        model_name = self.model.config.model_name

        print(f"\n--- Training {model_name} ---")
        self.model.train(dataset)       # polymorphic dispatch
        self.model.evaluate(dataset)    # polymorphic dispatch


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
def main():
    # Create config objects  (these will be embedded via composition)
    cfg_lr = ModelConfig("LinearRegression", lr=0.01,  epochs=10)
    cfg_nn = ModelConfig("NeuralNetwork",    lr=0.001, epochs=20)

    print(cfg_lr)   # __repr__ magic method
    print(cfg_nn)   # __repr__ magic method

    # Instantiate models  (class attribute total_models increments each time)
    lr_model = LinearRegressionModel(cfg=cfg_lr)
    nn_model = NeuralNetworkModel(cfg=cfg_nn, layers=[64, 32, 1])

    print(f"Models created: {BaseModel.total_models}")

    # Single shared DataLoader  (aggregation — lives outside of Trainer)
    loader = DataLoader(records=[1, 2, 3, 4, 5])

    # Trainers — same Trainer.run() works for both models (polymorphism)
    Trainer(model=lr_model, loader=loader).run()
    Trainer(model=nn_model, loader=loader).run()


if __name__ == "__main__":
    main()