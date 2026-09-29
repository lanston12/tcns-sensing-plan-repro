"""Exhaustive finite-domain checks; reads archived checks without changing evidence."""
import itertools as it
import json
import importlib.util
from fractions import Fraction as F
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def masks(n):
    return np.array(list(it.product((0, 1), repeat=n)), dtype=int)

def check_plan(a, b, p, s, weights, rng):
    n, m = len(a), len(p)
    zz = masks(n)
    actions = np.where(zz, b, a)
    direct = sum(weights[j] / (1 / p[j] + ((actions == j) / s[:, j]).sum(1)) for j in range(m))
    factors, scopes, errors = [], [], []
    law = rng.dirichlet(np.ones(len(zz)))  # Arbitrarily correlated, full-support law.
    expected = 0.0
    for j in range(m):
        scope = np.flatnonzero((a == j) != (b == j)).tolist()
        scopes.append(scope)
        d = len(scope)
        values, coeff = {}, {}
        base = 1 / p[j] + ((a == j) / s[:, j]).sum()
        for mask in range(1 << d):
            subset = tuple(scope[k] for k in range(d) if mask >> k & 1)
            values[subset] = weights[j] / (base + sum(((b[i] == j) * 1 - (a[i] == j) * 1) / s[i, j] for i in subset))
            coeff[subset] = sum((-1) ** (len(subset) - len(t)) * values[t]
                                for k in range(len(subset) + 1) for t in it.combinations(subset, k))
        reconstructed = np.zeros(len(zz))
        for subset, value in coeff.items():
            monomial = zz[:, subset].prod(1) if subset else np.ones(len(zz))
            reconstructed += value * monomial
            expected += value * (law @ monomial)
        true_factor = weights[j] / (1 / p[j] + ((actions == j) / s[:, j]).sum(1))
        errors.append(float(np.max(abs(true_factor - reconstructed))))
        factors.append(true_factor)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for scope in scopes:
        for i in scope[1:]:
            parent[find(i)] = find(scope[0])
    components = [tuple(i for i in range(n) if find(i) == r) for r in sorted({find(i) for i in range(n)})]
    constant = sum(f[0] for h, f in zip(scopes, factors) if not h)
    component_max = constant
    restricted_sets = []
    restricted_max = constant
    for comp in components:
        f = sum((v for h, v in zip(scopes, factors) if h and h[0] in comp), np.zeros(len(zz)))
        local = masks(len(comp))
        representatives = [next(k for k, z in enumerate(zz) if np.array_equal(z[list(comp)], x)) for x in local]
        component_max += f[representatives].max()
        keep = rng.random(len(local)) > .45
        keep[0] = True
        allowed = {tuple(x) for x in local[keep]}
        restricted_sets.append(allowed)
        restricted_max += f[np.array(representatives)[keep]].max()
    eligible = np.array([all(tuple(z[list(c)]) in allowed for c, allowed in zip(components, restricted_sets)) for z in zz])
    errors += [abs(float(law @ direct) - expected), abs(float(direct.max()) - component_max),
               abs(float(direct[eligible].max()) - restricted_max)]
    assert max(errors) < 1e-10, (a, b, errors)
    return max(errors), len(zz), len(components) > 1, max(map(len, scopes), default=0)

def main():
    rng = np.random.default_rng(20260927)
    cases = mask_cases = disconnected = 0
    max_error = 0.0
    orders = set()
    # Exhaust every old/new plan over {abstain, target 0, target 1}, N=1..3.
    for n in range(1, 4):
        plans = list(it.product((-1, 0, 1), repeat=n))
        for a, b in it.product(plans, repeat=2):
            result = check_plan(np.array(a), np.array(b), rng.uniform(.1, 2, 2),
                                rng.uniform(.1, 2, (n, 2)), rng.uniform(.2, 2, 2), rng)
            e, k, dc, d = result
            max_error = max(max_error, e); cases += 1; mask_cases += k; disconnected += dc; orders.add(d)
    exhaustive = cases
    # Include unchanged owners, repeated ownership, abstention, heterogeneous weights.
    for n in range(2, 8):
        for _ in range(100):
            m = int(rng.integers(1, 5))
            e, k, dc, d = check_plan(rng.integers(-1, m, n), rng.integers(-1, m, n),
                                    rng.uniform(.1, 2, m), rng.uniform(.1, 2, (n, m)),
                                    rng.uniform(.2, 2, m), rng)
            max_error = max(max_error, e); cases += 1; mask_cases += k; disconnected += dc; orders.add(d)
    cubic = sum((-1) ** (3-k) * F([1, 3, 3, 1][k], 1+k) for k in range(4))
    even = sum(F(1, 1+sum(z)) for z in it.product((0, 1), repeat=3) if sum(z) % 2 == 0) / 4
    odd = sum(F(1, 1+sum(z)) for z in it.product((0, 1), repeat=3) if sum(z) % 2 == 1) / 4
    assert cubic == F(-1, 4) and even == F(1, 2) and odd == F(7, 16)
    # Two unit-prior/noise owners change to abstention. Coupled support matters.
    constrained=max(F(1,2-z[0])+F(1,2-z[1]) for z in [(1,0),(0,1)])
    assert constrained == F(3,2) < 2
    threshold_cases=1000
    for _ in range(threshold_cases):
        p1,p2,s11,s21,s22=[F(int(rng.integers(1,100)),100) for _ in range(5)]
        u=1/(1/p1+1/s11)
        duplicate=1/(1/u+1/s21)+p2
        split=u+1/(1/p2+1/s22)
        assert (duplicate <= split) == (u*u/(u+s21) >= p2*p2/(p2+s22))
    correlated_cycle_cases=0; correlated_cycle_error=0.0
    for n in range(2,7):
        zz=masks(n)
        for perm in it.permutations(range(n)):
            perm=np.array(perm);inv=np.argsort(perm)
            law=rng.dirichlet(np.ones(len(zz)))
            actions=np.where(zz,perm,np.arange(n))
            counts=np.array([(actions==j).sum(1) for j in range(n)]).T
            cost=(1/(1/.2+counts/.03)).sum(1); optimum=n*.2*.03/(.2+.03)
            q=law@zz; pairs=np.einsum('k,ki,kj->ij',law,zz,zz)
            delta=2*.2**3/((.03+.2)*(.03+2*.2))
            formula=delta/2*sum(q[j]+q[inv[j]]-2*pairs[j,inv[j]] for j in range(n))
            correlated_cycle_error=max(correlated_cycle_error,abs(float(law@(cost-optimum))-formula))
            correlated_cycle_cases+=1
    assert correlated_cycle_error < 1e-12
    source = ROOT / 'publication_track/installation_focus/provided/check_installation_extension.py'
    spec = importlib.util.spec_from_file_location('archived_checker', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original = module.run_checks()
    output = dict(status='passed', seed=20260927,
                  scope='Finite mathematical checks, not new network experiments or a novelty proof.',
                  original_cycle_and_witness=original,
                  hypergraph=dict(exhaustive_plan_pairs=exhaustive, random_plan_pairs=cases-exhaustive,
                                  all_plan_pairs=cases, full_mask_cases=mask_cases,
                                  disconnected_plan_pairs=disconnected, observed_scope_orders=sorted(orders),
                                  max_absolute_error=max_error,
                                  checks=['target-factor expansion at every mask', 'correlated expectation',
                                          'unrestricted component maximum', 'product-restricted component maximum']),
                  exact_cubic_witness=dict(coefficient=str(cubic), even_parity_expectation=str(even),
                                           odd_parity_expectation=str(odd), equal_marginals='1/2', equal_pairs='1/4'),
                  exact_redundancy_threshold_cases=threshold_cases,
                  correlated_cycle_laws=dict(permutations=correlated_cycle_cases,max_error=correlated_cycle_error),
                  nonproduct_constraint_counterexample='max[1/(2-z1)+1/(2-z2)] over {10,01}=3/2; sum of projection maxima=2')
    (ROOT / 'results/tcns_structural_checks.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k != 'original_cycle_and_witness'}, indent=2))

if __name__ == '__main__':
    main()
