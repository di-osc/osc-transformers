



class LogitsProcessor(object):
    """
    Abstract base class for all logit processors that can be applied during generation.
    """

    def __call__(self, input_ids: int, scores: float) -> float:
        """
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                The sequence used as context to build and/or append to the prompt.
            scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`):
                The scores to consider while doing the logit processor's work.

        Returns:
            :obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.vocab_size)`: The modified scores.
        """
        raise NotImplementedError("Abstract method")