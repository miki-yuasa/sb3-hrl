import importlib


def get_class(cls_name: str) -> type:
    """Get a class by its name from the module."""
    if cls_name and "." in cls_name:
        module_name, class_name = cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    else:
        raise ValueError(f"Class name '{cls_name}' must include the module prefix.")
