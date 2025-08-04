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
        norm_date=None,
    )
    inpt_without_year = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="27 febbraio - 29 settembre",
        anno=1981,
        norm_date=None,
    )
    expected = Date("27/febbraio/1981", "29/settembre/1981", "day")

    assert _is_equal(transforms.get_day_period(inpt), expected)
    assert _is_equal(transforms.get_day_period(inpt_without_year), expected)
    assert _is_equal(transforms.generic_period(inpt), expected)
    assert _is_equal(transforms.generic_period(inpt_without_year), expected)


def test_single_day_period():
    """Test the extraction of a single day period."""
    single_day_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="27 febbraio 1981",
        anno=1981,
        norm_date=None,
    )
    single_day_input_without_year = (
        InterventionDataForDateNormalizationRowSchema(
            idscheda=8,
            data_protocollo="",
            data_intervento="27 febbraio",
            anno=1981,
            norm_date=None,
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
    assert _is_equal(
        transforms.generic_single_period(single_day_input), expected_single
    )
    assert _is_equal(
        transforms.generic_single_period(single_day_input_without_year),
        expected_single,
    )


def test_month_period():
    month_and_year_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="febbraio 1979 -  settembre 1981",
        anno=1981,
        norm_date=None,
    )
    assert _is_equal(
        transforms.get_month_period(month_and_year_input),
        Date("1/febbraio/1979", "28/settembre/1981", "month"),
    )
    assert _is_equal(
        transforms.generic_period(month_and_year_input),
        Date("1/febbraio/1979", "28/settembre/1981", "month"),
    )
    month_and_single_year_input = (
        InterventionDataForDateNormalizationRowSchema(
            idscheda=8,
            data_protocollo="",
            data_intervento="febbraio -  settembre 1981",
            anno=1981,
            norm_date=None,
        )
    )
    expected_months_year = Date(
        "1/febbraio/1981", "28/settembre/1981", "month"
    )
    assert _is_equal(
        transforms.get_month_period(month_and_single_year_input),
        expected_months_year,
    )
    assert _is_equal(
        transforms.generic_period(month_and_single_year_input),
        expected_months_year,
    )
    month_without_year_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="febbraio -  settembre",
        anno=1981,
        norm_date=None,
    )
    assert _is_equal(
        transforms.get_month_period(month_without_year_input),
        expected_months_year,
    )
    assert _is_equal(
        transforms.generic_period(month_without_year_input),
        expected_months_year,
    )


def test_single_month_period():
    expected_month_year = Date("1/febbraio/1981", "28/febbraio/1981", "month")
    month_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="febbraio 1981",
        anno=1981,
        norm_date=None,
    )
    month_without_year_input = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="febbraio",
        anno=1981,
        norm_date=None,
    )
    assert _is_equal(
        transforms.get_single_month_period(month_input), expected_month_year
    )
    assert _is_equal(
        transforms.get_single_month_period(month_without_year_input),
        expected_month_year,
    )
    assert _is_equal(
        transforms.generic_single_period(month_input), expected_month_year
    )
    assert _is_equal(
        transforms.generic_single_period(month_without_year_input),
        expected_month_year,
    )


def test_year_period():
    assert _is_equal(
        transforms.generic_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="2005 -",
                anno=2006,
                norm_date=None,
            )
        ),
        Date("1/1/2005", "31/12/2006", "year"),
    )
    assert _is_equal(
        transforms.generic_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="2006 -",
                anno=2006,
                norm_date=None,
            )
        ),
        Date("1/1/2006", "31/12/2006", "year"),
    )


def test_year_single_period():
    assert _is_equal(
        transforms.generic_single_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="2006 ",
                anno=2006,
                norm_date=None,
            )
        ),
        Date("1/1/2006", "31/12/2006", "year"),
    )


def test_generic_period():
    assert _is_equal(
        transforms.generic_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="6-10 novembre 2006",
                anno=2006,
                norm_date=None,
            )
        ),
        Date("6/novembre/2006", "10/novembre/2006", "day"),
    )
    assert _is_equal(
        transforms.generic_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="25 settembre 2019 - 09 ottobre",
                anno=2019,
                norm_date=None,
            )
        ),
        Date("25/settembre/2019", "09/ottobre/2019", "day"),
    )
    assert _is_equal(
        transforms.generic_period(
            InterventionDataForDateNormalizationRowSchema(
                idscheda=8,
                data_protocollo="",
                data_intervento="7 - 30 giugno",
                anno=2019,
                norm_date=None,
            )
        ),
        Date("7/giugno/2019", "30/giugno/2019", "day"),
    )


def test_generic_period_reject():
    single_period = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="febbraio",
        anno=1981,
        norm_date=None,
    )
    period = InterventionDataForDateNormalizationRowSchema(
        idscheda=8,
        data_protocollo="",
        data_intervento="7 - 30 giugno",
        anno=2019,
        norm_date=None,
    )
    assert transforms.generic_period(single_period) is None
    assert transforms.generic_single_period(period) is None
