"""All the instances and properties here can be statically computed (meaning they can be computed once and in runtime be used e.g. from a cache) as well as their embedding, if the Embedder model is fixed."""

from dataclasses import dataclass


@dataclass
class RetrievableFieldOption:
    name: str
    thesaurus: list[str]  # a list of synonyms matching the same field
    text_for_embedding: str
    """the text that will be passed into the embedder
    """
    description_for_llm: str
    """the text that will be be passed into the query
    """
    examples: str


@dataclass
class RetrievableField:
    name: str
    keywords: str  # list of related keywords
    text_for_embedding: str
    """the text that will be passed into the embedder
    """
    description_for_llm: str
    """the text that will be be passed into the query
    """
    choices: list[RetrievableFieldOption]
