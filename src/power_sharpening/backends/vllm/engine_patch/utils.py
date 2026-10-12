from contextlib import contextmanager
import sys



@contextmanager
def patch(replacements: list[dict[str, type]]):
    """Temporarily patch classes or functions in a module.

    Args:
      replacements: A mapping from the full path of the class/function to
        the new class/function to use.
    """
    contexts = []
    for replacement in replacements:
        module_path = replacement["module"]
        new_class = replacement["class"]
        class_name = replacement.get("name", new_class.__name__)

        module = sys.modules[module_path]
        original_model_class = getattr(module, class_name)

        contexts.append((module, class_name, original_model_class, new_class))

    try:
        for module, class_name, _, new_class in contexts:
            setattr(module, class_name, new_class)
        yield
    finally:
        for module, class_name, original_model_class, _ in contexts:
            setattr(module, class_name, original_model_class)


class _dummytqdm:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def update(self, n=1):
        pass

    def set_description(self, desc=None):
        pass


