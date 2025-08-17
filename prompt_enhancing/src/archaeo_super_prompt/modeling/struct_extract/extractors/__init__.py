"""Predictors for the extraction of 1 field.

They are either a subclass of the FieldExtractor abstract class or a another
special class, e.g. in the case of an "Oracle predictor", using directly the
training value to simulate a correct input inserted by a human contributor, if
a good extraction model is not implemented yet.
"""
