"""Test generic features of the FieldExtractor."""

from archaeo_super_prompt.modeling.struct_extract.field_extractor import to_prediction, prediction_to_output
from pydantic import BaseModel

def test_type_bijection():
    """Test the bijection relationship between two methods of TypedDspyModule."""
    class OutputModel(BaseModel):
        multiplied_foo: str

    my_output = OutputModel(multiplied_foo="hellohellohello")
    assert(my_output == prediction_to_output(OutputModel, to_prediction(my_output)))
