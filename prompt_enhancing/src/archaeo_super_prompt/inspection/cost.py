import dspy


def inspect_cost(lm: dspy.LM):
    """cost in USD of the usage of a language model here, as calculated by
    LiteLLM for certain providers
    """
    cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
    return cost
