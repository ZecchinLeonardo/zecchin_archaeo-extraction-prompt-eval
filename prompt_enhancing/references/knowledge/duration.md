# Duration of an intervention

The duration of an intervention is an optional field when a contributor insert
it into Magoh. This field is indeed supposed to equal None when the value is
impossible to be extracted from the archive documents.

However, in documents recorded from the Mappa project, an older project before
Magoh, this field did not exist during their registering, and it has been kept
as None during the migration into Magoh. Therefore, for an evaluation or a
training of a llm-extractor for the duration of an intervention, please only
pass records for which the "Motivazione" was a string similar to "Magoh
Project".

The records for which the [start date of the intervention have been
normalized](./interv_start_date.md) have also normalized duration value, with a
level of precision which varies between the day, the month or the year.
