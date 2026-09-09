from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class RatingTriples:
    users: IntArray
    items: IntArray
    ratings: FloatArray
    @property
    def n_ratings(self) -> int: return int(self.ratings.size)

def load_rating_triples(path: str|Path, *, delimiter: str|None=None) -> RatingTriples:
    rows=[]
    with Path(path).open(encoding="utf-8") as h:
        for line in h:
            line=line.strip()
            if not line or line.startswith("#"): continue
            parts=line.split(delimiter) if delimiter else line.split()
            if len(parts)<3: raise ValueError("rating row must contain user, item, rating")
            rows.append((int(parts[0]),int(parts[1]),float(parts[2])))
    if not rows: raise ValueError("rating file must not be empty")
    a=np.asarray(rows,dtype=float)
    if np.any(a[:,2]<1) or np.any(a[:,2]>5): raise ValueError("ratings must lie in [1, 5]")
    return RatingTriples(a[:,0].astype(np.int64),a[:,1].astype(np.int64),a[:,2].astype(float))

def load_coat_matrix(path: str|Path) -> RatingTriples:
    matrix=np.loadtxt(path,dtype=float)
    if matrix.ndim!=2: raise ValueError("Coat matrix must be 2D")
    users,items=np.nonzero(matrix)
    ratings=matrix[users,items]
    if np.any((ratings<1)|(ratings>5)): raise ValueError("nonzero Coat ratings must lie in [1, 5]")
    return RatingTriples(users.astype(np.int64),items.astype(np.int64),ratings.astype(float))
