# Experimental protocol

We describe here how the evaluations have been carried out.

1. 150 documents have been inspected and classified among these classes:
   - digitally-born
   - to be scanned but clean
   - to be scanned and dirty
2. If there is no optimization to be done for the prompt models, then we evaluate over
every processable document (we name "processable" a document for which the scan
by the vision LLM does not raise a timeout).
3. If there is an optimization to do, we split randomly the set between a
training and an evaluation set (with 20% of the records kept for the
training).
4. In the analysis, we do not pay attention for now to the class of the PDF
documents mentioned above.

The evaluation is defined for each field among a perfect match or a close
match. On the last specialized models, the evaluation method is more accurately
defined. In fact, even the implementation of a tolerance (e.g. for the guessed
date) is handled thanks to the data which is cleaner than at the epocha of the
first model. The metric setting is also better for these models, which has
enabled them to be optimized by DSPy.
