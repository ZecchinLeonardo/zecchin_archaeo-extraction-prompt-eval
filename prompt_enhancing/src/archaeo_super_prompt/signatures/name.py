import pydantic

class Name(pydantic.BaseModel):
    """The first name and the surname of a person
    """
    first_name: str
    surname: str

def toMappaNaming(name: Name) -> str:
    return f"{name.first_name[0].upper()}. {name.surname}"
