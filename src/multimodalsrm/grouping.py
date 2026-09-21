"""Deterministic free latent coordinates for each pooling constraint."""


def complete_affinity(subjects):
    """Represent exact sharing with unit edges and an implicit zero diagonal."""
    return {s: {t: 1.0 for t in subjects if t != s} for s in subjects}


def latent_groups(subjects, pooling, affinity=None):
    """Return groups and members in input order; positive edges imply sharing."""
    subjects = list(subjects)
    if pooling == "shared":
        return [subjects] if subjects else []
    if pooling != "components":
        return [[s] for s in subjects]
    if affinity is None:
        raise ValueError("components pooling requires a named affinity graph")
    remaining = set(subjects)
    groups = []
    for first in subjects:
        if first not in remaining:
            continue
        reached, queue = {first}, [first]
        while queue:
            s = queue.pop()
            for t, weight in affinity.get(s, {}).items():
                if weight > 0 and t in remaining and t not in reached:
                    reached.add(t)
                    queue.append(t)
        group = [s for s in subjects if s in reached]
        groups.append(group)
        remaining.difference_update(group)
    return groups
