import catalogue
import confection


class Registry(confection.registry):
    models = catalogue.create("osc", "asr", "models", entry_points=True)


__all__ = ["Registry"]
