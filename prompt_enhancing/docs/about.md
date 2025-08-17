# About the project

## Missing features

- The development of two abstract class inheriting the `ClassifierMixin` (for
the `predict` method) and the `BaseEstimator` classes:
  - `FieldOracle` an Estimator pr
- The integration of MLFlow inside the project should have been strengthened
with both these following features:
  - the support of SKLearn models for the logging, the tracing and
  the scoring
  - the logging of the dspy models' signatures (efforts have been
  made inside the code to explicitly declare those signatures, but the
  integration of dspy in mlflow is still experimental, so the logging of the
  signatures is not that easy)
