from typing import List, Tuple
from ..modeling.entity_extractor.types import CompleteEntity
import functools as fnt


def visualize_entities(content: str, entities: List[CompleteEntity]):
    """Render the content with all its extracted entities highlighted and
    labeled with their entities. The rendered string is written with Markdown
    syntax and is ready to be displayed in a notebook
    """

    def add(acc: Tuple[str, int], entity: CompleteEntity) -> Tuple[str, int]:
        """acc contains the accumulated string and the length of the
        already-processed content
        """
        acc_text, processed_source_content_length = acc
        to_be_add = (
            content[processed_source_content_length : entity.start]
            + f"<mark>{entity.word}</mark> `{entity.entity}`"
        )
        return acc_text + to_be_add, entity.end

    text_with_all_marked_entities, processed_source_content_length = (
        fnt.reduce(add, entities, ("", 0))
    )
    return (
        text_with_all_marked_entities
        + content[processed_source_content_length:]
    )
