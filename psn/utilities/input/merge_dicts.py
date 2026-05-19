"""Dictionary merging utility."""


def merge_dicts(base, override):
    """Merge two dicts, with override taking precedence.

    Combines two dicts, with fields from override replacing any
    matching fields in base.

    Parameters
    ----------
    base : dict
        Dict with default or base field values.
    override : dict
        Dict with fields that should replace those in base.

    Returns
    -------
    merged : dict
        Dict containing all fields from base, with any fields
        present in override replaced by their override values.
    """
    merged = base.copy()
    for key, value in override.items():
        merged[key] = value
    return merged
