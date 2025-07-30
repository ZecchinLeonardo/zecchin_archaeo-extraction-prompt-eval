"""Test generic features of the FieldExtractor."""

from archaeo_super_prompt.modeling.struct_extract.field_extractor import TypedDspyModule
from pydantic import BaseModel

def test_type_bijection():
    """Test the bijection relationship between two methods of TypedDspyModule."""
    class InputModel(BaseModel):
        foo: str
        bar: int

    class OutputModel(BaseModel):
        multiplied_foo: str

    class MyTypedDspyModule(TypedDspyModule[InputModel, OutputModel]):
        def __init__(self):
            super().__init__(OutputModel)

        def forward(self, foo: str, bar: int):
            return self._to_prediction(OutputModel(multiplied_foo=foo*bar))

        def test(self):
            my_output = OutputModel(multiplied_foo="hellohellohello")
            assert(my_output == self._prediction_to_output(self._to_prediction(my_output)))


    test_typed_module = MyTypedDspyModule()
    test_typed_module.test()
