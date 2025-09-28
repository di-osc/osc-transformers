import catalogue
import confection


class Registry(confection.registry):
    architecture = catalogue.create("osc", "transformers", "architecture", entry_points=True)
    attention = catalogue.create("osc", "transformers", "attention", entry_points=True)
    feedforward = catalogue.create("osc", "transformers", "feedforward", entry_points=True)
    normalization = catalogue.create("osc", "transformers", "normalization", entry_points=True)
    head = catalogue.create("osc", "transformers", "head", entry_points=True)
    embedding = catalogue.create("osc", "transformers", "embedding", entry_points=True)
    sampler = catalogue.create("osc", "transformers", "sampler", entry_points=True)
    quantizers = catalogue.create("osc", "transformers", "quantizers", entry_points=True)


__all__ = ["Registry"]
