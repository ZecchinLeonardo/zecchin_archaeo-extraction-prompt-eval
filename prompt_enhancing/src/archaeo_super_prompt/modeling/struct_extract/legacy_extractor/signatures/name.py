from typing import override
import pydantic


class Name(pydantic.BaseModel):
    """The first name and the surname of a person"""

    first_name: str
    surname: str

    @override
    def __str__(self):
        return f"{self.first_name[0].upper() if len(self.first_name) >= 1 else ''}. {self.surname}"
