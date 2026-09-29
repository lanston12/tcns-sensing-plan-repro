"""Independent mathematical checks, not a rerun of the user's experiments.

Checks a proposed N-source permutation extension of the manuscript's
2-source mixed-installation example. The model is synchronous, scalar,
conditionally independent, equal P/S, with exactly one reliable observation
per source. No wireless performance or novelty claim follows.

Usage: python check_installation_extension.py [--out result.json]
"""
from __future__ import annotations
import argparse
import itertools
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def cycle_lengths(perm: tuple[int, ...]) -> list[int]:
    seen: set[int] = set()
    lengths: list[int] = []
    for root in range(len(perm)):
        if root in seen:
            continue
        j, length = root, 0
        while j not in seen:
            seen.add(j)
            length += 1
            j = perm[j]
        lengths.append(length)
    return lengths


def scalar_cost(prior: list[Fraction], variances: list[list[Fraction]],
                assignment: tuple[int, ...]) -> Fraction:
    total = Fraction(0)
    for j, p in enumerate(prior):
        precision = 1 / p
        for i, a in enumerate(assignment):
            if a == j:
                precision += 1 / variances[i][j]
        total += 1 / precision
    return total


def run_checks() -> dict:
    pairs = [(0.01, 0.03), (0.05, 0.02), (0.1, 0.1), (0.2, 0.01)]
    cases = 0
    permutations = 0
    max_error = 0.0
    max_expectation_error = 0.0
    max_worst_error = 0.0
    q = 0.37
    for n in range(2, 7):
        z = np.array(list(itertools.product([0, 1], repeat=n)), dtype=int)
        nodes = np.arange(n)
        mask_probability = q ** z.sum(axis=1) * (1 - q) ** (n - z.sum(axis=1))
        for perm0 in itertools.permutations(range(n)):
            permutations += 1
            perm = np.array(perm0)
            previous = np.argsort(perm)
            targets = np.where(z == 1, perm[None, :], nodes[None, :])
            counts = np.array([(targets == j).sum(axis=1) for j in nodes]).T
            empty = (counts == 0).sum(axis=1)
            cut = np.abs(z - z[:, previous]).sum(axis=1) / 2
            assert np.array_equal(empty, cut)
            max_empty = sum(length // 2 for length in cycle_lengths(perm0) if length > 1)
            assert int(empty.max()) == max_empty
            moved = int((perm != nodes).sum())
            for p, s in pairs:
                delta = 2 * p ** 3 / ((s + p) * (s + 2 * p))
                exact = (1 / (1 / p + counts / s)).sum(axis=1)
                optimal = n * p * s / (p + s)
                formula = optimal + delta * cut
                max_error = max(max_error, float(np.max(np.abs(exact - formula))))
                expected = float(mask_probability @ (exact - optimal))
                predicted = delta * moved * q * (1 - q)
                max_expectation_error = max(max_expectation_error, abs(expected - predicted))
                max_worst_error = max(max_worst_error,
                                      abs(float((exact - optimal).max()) - delta * max_empty))
                cases += len(z)
    assert max_error < 1e-12
    assert max_expectation_error < 1e-12
    assert max_worst_error < 1e-12

    # Deliberately constructed rational example: old/new grants are unique
    # optima in different receiver covariance contexts. Source S is unchanged.
    variances = [[Fraction(1, 25), Fraction(1, 40)],
                 [Fraction(1, 250), Fraction(1, 100)]]
    priors = {"A": [Fraction(1, 50), Fraction(9, 50)],
              "B": [Fraction(6, 25), Fraction(1, 25)]}
    grants = list(itertools.product(range(2), repeat=2))
    rows = []
    exact_costs = {}
    for name, prior in priors.items():
        cs = {g: scalar_cost(prior, variances, g) for g in grants}
        exact_costs[name] = cs
        for grant, cost in cs.items():
            rows.append({"context": name, "grant": [x + 1 for x in grant],
                         "cost_rational": str(cost), "cost": float(cost)})
    assert min(exact_costs['A'], key=exact_costs['A'].get) == (0, 1)
    assert min(exact_costs['B'], key=exact_costs['B'].get) == (1, 0)
    assert len(set(exact_costs['A'].values())) == 4
    assert len(set(exact_costs['B'].values())) == 4
    assert min(exact_costs['B'][(0, 0)], exact_costs['B'][(1, 1)]) > max(
        exact_costs['B'][(0, 1)], exact_costs['B'][(1, 0)])
    return {
        "scope": "New mathematical illustration only; NOT NR-V2X data, NOT a validation sample, NOT a novelty verification",
        "n_range": [2, 6],
        "prior_noise_pairs": pairs,
        "permutations_checked": permutations,
        "mask_parameter_cases": cases,
        "max_formula_absolute_error": max_error,
        "max_independent_expectation_absolute_error": max_expectation_error,
        "max_worst_case_absolute_error": max_worst_error,
        "non_tie_example": {"source_variances": [[str(x) for x in row] for row in variances],
                            "prior_contexts": {k: [str(x) for x in v] for k, v in priors.items()},
                            "costs": rows},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('mathematical_check.json'))
    args = parser.parse_args()
    result = run_checks()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'non_tie_example'}, indent=2))

if __name__ == '__main__':
    main()
