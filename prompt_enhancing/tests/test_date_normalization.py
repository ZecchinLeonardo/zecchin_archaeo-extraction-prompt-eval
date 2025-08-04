"""Test the normalization of the intervention dates."""

from archaeo_super_prompt.dataset.normalization.intervention_date import (
    transforms,
)
from archaeo_super_prompt.dataset.normalization.intervention_date.utils import (
    InterventionDataForDateNormalizationRowSchema,
    Date,
)


def _is_equal(output: Date | None, expected: Date):
    return output is not None and output == expected


def test_day_period_transform():
    """Test if a day period is extracted."""
    inpt = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="27 febbraio - 29 settembre 1981",
        anno=1981,
        processed_date=None,
    )
    inpt_without_year = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="27 febbraio - 29 settembre",
        anno=1981,
        processed_date=None,
    )
    expected = Date("27/febbraio/1981", "29/settembre/1981", "day")

    assert _is_equal(transforms.get_day_period(inpt), expected)
    assert _is_equal(transforms.get_day_period(inpt_without_year), expected)

def test_single_day_period():
    """Test the extraction of a single day period."""
    single_day_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="27 febbraio 1981",
        anno=1981,
        processed_date=None,
    )
    single_day_input_without_year = (
        InterventionDataForDateNormalizationRowSchema(
            idscheda=8,
            data_protocollo="",
            data_intervento="27 febbraio",
            anno=1981,
            processed_date=None,
        )
    )
    expected_single = Date("27/febbraio/1981", "27/febbraio/1981", "day")
    assert _is_equal(
        transforms.get_single_day_period(single_day_input), expected_single
    )
    assert _is_equal(
        transforms.get_single_day_period(single_day_input_without_year),
        expected_single,
    )
