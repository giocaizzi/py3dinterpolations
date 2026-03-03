"""Plotting utilities."""

SLICING_AXIS: dict[str, dict[str, str]] = {
    "X": {"X'": "Y", "Y'": "Z"},
    "Y": {"X'": "X", "Y'": "Z"},
    "Z": {"X'": "X", "Y'": "Y"},
}


def number_of_plots(n: int, n_cols: int = 4) -> tuple[int, int]:
    """Determine grid layout rows and columns for n subplots."""
    n_rows = (n + n_cols - 1) // n_cols
    return n_rows, n_cols
