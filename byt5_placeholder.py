"""Reserved extension point for a ByT5 implementation."""


class ByT5Placeholder:
    """A LM placeholder, kept separate from the conventional receiver."""

    def predict(self, observation: object):
        # Fail explicitly so this placeholder cannot be mistaken for inference.
        raise NotImplementedError("LM prediction is not included.")
