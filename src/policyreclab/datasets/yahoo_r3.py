from __future__ import annotations

from pathlib import Path
import numpy as np

from policyreclab.datasets.mnar_ratings import RatingTriples


def load_yahoo_r3_triples(path: str | Path) -> RatingTriples:
    """Load Yahoo R3 triples from standard or common mirror text formats.

    Supported rows are whitespace-separated or comma-separated:
    ``user item rating`` or ``user,item,rating``. A single header row such as
    ``uid,iid,rating`` is ignored.

    The original Yahoo files are one-based; common mirrors are zero-based.
    Indexing is inferred at file level from the presence of user ID 0 so user
    and item IDs cannot be shifted inconsistently merely because item 0 is
    absent from a particular sample.
    """
    rows: list[tuple[int, int, float]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = (
                [part.strip() for part in line.split(",")]
                if "," in line
                else line.split()
            )
            if len(parts) < 3:
                raise ValueError(
                    f"Yahoo R3 row {line_number} must contain user, item, rating"
                )

            try:
                user = int(parts[0])
                item = int(parts[1])
                rating = float(parts[2])
            except ValueError:
                if not rows and line_number == 1:
                    continue
                raise ValueError(
                    f"Yahoo R3 row {line_number} contains non-numeric fields"
                ) from None

            rows.append((user, item, rating))

    if not rows:
        raise ValueError("Yahoo R3 rating file must not be empty")

    values = np.asarray(rows, dtype=np.float64)
    users = values[:, 0].astype(np.int64)
    items = values[:, 1].astype(np.int64)
    ratings = values[:, 2].astype(np.float64)

    if np.any((ratings < 1.0) | (ratings > 5.0)):
        raise ValueError("Yahoo R3 ratings must lie in [1, 5]")

    zero_based = bool(np.any(users == 0))
    if not zero_based:
        users = users - 1
        items = items - 1

    if np.any(users < 0) or np.any(users >= 15400):
        raise ValueError("Yahoo R3 user IDs exceed documented bounds")
    if np.any(items < 0) or np.any(items >= 1000):
        raise ValueError("Yahoo R3 item IDs exceed documented bounds")

    return RatingTriples(users=users, items=items, ratings=ratings)
