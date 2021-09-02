import tensorflow as tf


class AdvModel:
    def __init__(self, input_size, output_sizes):
        self.input_size = input_size
        self.output_sizes = output_sizes
        parts = self._build_parts()

    def _build_parts(self):
        return None
