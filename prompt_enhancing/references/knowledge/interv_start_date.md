# Start date of an intervention

This field is important to be predicted as it enables other fields to be
inferred. It is specified by the contributors in the `data_intervento` field in
the Magoh database, but not in a strict numeric format as this date is not
always known with the same precision.

The format of this date is therefore arbitrary and set differently according to
each contributor (even sometimes with human misunderstandings of the definition
of the field). However, with combining regularly seen patterns and information
from some contributors, it has been possible to normalized most of those dates
into a window of two possible dates with an explicit precision.

The detail about this normalization is written in the
`NormalizeInterventionDate.ipynb` notebook.
