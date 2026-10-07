"""Offline, stereochemistry-aware catalogue and formula-isolated data views."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from xtbflow.m0.metrics import be_to_mol
from xtbflow.m0.t1x_data import T1xCache

EV_TO_KCAL_MOL = 23.060547830619
MAX_MAPS = 100000


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def reactant_mol(z, b, x):
    mol = be_to_mol(np.asarray(z), np.asarray(b), sanitize=True)
    conf = Chem.Conformer(len(z))
    conf.Set3D(True)
    for i, point in enumerate(x):
        conf.SetAtomPosition(i, tuple(float(v) for v in point))
    mol.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def graph_identity(z, b, x):
    mol = reactant_mol(z, b, x)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    charge = int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))
    if charge != 0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError("outside neutral closed-shell cache contract")
    return digest([smiles, charge, 1])[:20], smiles, mol


def legal_maps(query_mol, ref_mol, query_z, ref_z, query_b, ref_b, cap=MAX_MAPS):
    """Return query-index -> reference-index maps, with exact BE and stereo checks.

    RDKit sanitization can aromaticize bonds; the explicit BE check below prevents
    mappings from changing a Kekule bond order or a diagonal electron count.
    Never silently replace a failed/capped mapping search with the identity.
    """
    raw = ref_mol.GetSubstructMatches(query_mol, uniquify=False, useChirality=True,
                                     maxMatches=cap + 1)
    if len(raw) > cap:
        raise ValueError("atom_mapping_cap_exceeded")
    maps = []
    for row in raw:
        pi = np.asarray(row, dtype=np.int64)
        if np.array_equal(query_z, np.asarray(ref_z)[pi]) and np.array_equal(
                query_b, np.asarray(ref_b)[np.ix_(pi, pi)]):
            maps.append(pi)
    if not maps:
        raise ValueError("no_exact_reactant_atom_mapping")
    return np.asarray(maps)


def canonical_event(b_r, b_p, perms):
    """Directed complete BE delta, quotiented only by reactant automorphisms."""
    delta = np.asarray(b_p, dtype=np.int64) - np.asarray(b_r, dtype=np.int64)
    # Include diagonal changes; off-diagonal entries retain signed bond orders.
    triangle = np.triu_indices(len(delta))
    values = delta[perms[:, :, None], perms[:, None, :]][:, triangle[0], triangle[1]]
    key = min(tuple(int(v) for v in row) for row in values)
    return digest(key)[:24]


def build_catalogue(cache_path):
    cache = T1xCache(Path(cache_path))
    arrays = cache.arrays
    buckets = defaultdict(list)
    failures = []
    for i in range(len(cache)):
        r = cache.reaction(i)
        try:
            if not np.isfinite(arrays['energies'][i]).all():
                raise ValueError("nonfinite_reference_energy")
            pid, smiles, mol = graph_identity(r['z'], r['b_r'], r['x_r'])
            r.update(parent_id=pid, smiles=smiles, mol=mol,
                     reference_id=str(arrays['rxn_key'][i]),
                     formula=str(arrays['formula'][i]),
                     official_split=int(arrays['split'][i]),
                     energies=arrays['energies'][i])
            buckets[pid].append(r)
        except (ValueError, RuntimeError) as exc:
            failures.append({'cache_index': i, 'reason': str(exc)})
    parents, references = [], []
    for pid, records in sorted(buckets.items()):
        try:
            formulas = {r['formula'] for r in records}
            splits = {r['official_split'] for r in records}
            if len(formulas) != 1 or len(splits) != 1:
                raise ValueError("parent_crosses_formula_or_official_split")
            anchor = min(records, key=lambda r: (r['energies'][0], r['reference_id']))
            perms = legal_maps(anchor['mol'], anchor['mol'], anchor['z'], anchor['z'],
                               anchor['b_r'], anchor['b_r'])
            refs = []
            for r in sorted(records, key=lambda r: r['reference_id']):
                pi = legal_maps(anchor['mol'], r['mol'], anchor['z'], r['z'],
                                anchor['b_r'], r['b_r'])[0]
                bp = r['b_p'][np.ix_(pi, pi)]
                er, ets, ep = (float(v) for v in r['energies'])
                refs.append(dict(
                    reference_id=r['reference_id'], parent_id=pid, cache_index=r['index'],
                    original_atom_mapping=pi.tolist(), b_p=bp.astype(int).tolist(),
                    x_ts=r['x_ts'][pi].tolist(), x_r=r['x_r'][pi].tolist(),
                    x_p=r['x_p'][pi].tolist(), energy_r_ev=er, energy_ts_ev=ets,
                    energy_p_ev=ep, native_barrier_kcal=(ets-er)*EV_TO_KCAL_MOL,
                    catalog_barrier_kcal=(ets-float(anchor['energies'][0]))*EV_TO_KCAL_MOL,
                    channel_id=canonical_event(anchor['b_r'], bp, perms),
                    energy_source='T1x wB97X/6-31G(d) electronic energy',
                    coordinate_unit='angstrom', energy_unit='eV'))
            best = min(r['catalog_barrier_kcal'] for r in refs)
            parent = dict(parent_id=pid, smiles=anchor['smiles'], formula=anchor['formula'],
                          split_group=anchor['formula']+'|q0|m1',
                          official_split=anchor['official_split'], charge=0, multiplicity=1,
                          anchor_cache_index=anchor['index'],
                          anchor_reference_id=anchor['reference_id'],
                          query_id=pid+'_anchor_'+digest(anchor['x_r'].tolist())[:12],
                          atomic_numbers=anchor['z'].tolist(), x_r=anchor['x_r'].tolist(),
                          b_r=anchor['b_r'].astype(int).tolist(), permutations=perms.tolist(),
                          reference_ids=[r['reference_id'] for r in refs],
                          cache_indices=[r['cache_index'] for r in refs],
                          known_channels=len({r['channel_id'] for r in refs}),
                          best_reference_ids=[r['reference_id'] for r in refs
                                              if r['catalog_barrier_kcal']<=best+1e-6],
                          best_channel_ids=sorted({r['channel_id'] for r in refs
                                                   if r['catalog_barrier_kcal']<=best+1e-6}),
                          best_barrier_kcal=best)
            parents.append(parent)
            references.extend(refs)
        except (ValueError, RuntimeError) as exc:
            failures.append({'parent_id': pid, 'cache_indices': [r['index'] for r in records],
                             'reason': str(exc)})
    return parents, references, failures


def inventory(parents, references, failures):
    out = {'parents_total': len(parents), 'references_total': len(references),
           'exclusions': failures, 'official_splits': {}}
    sets = {}
    for s, name in enumerate(('train', 'val', 'test')):
        ps = [p for p in parents if p['official_split'] == s]
        eligible = [p for p in ps if p['known_channels'] >= 2]
        sizes = Counter(p['split_group'] for p in eligible)
        sets[name] = set(p['split_group'] for p in ps)
        out['official_splits'][name] = dict(parents=len(ps), eligible_parents=len(eligible),
            formula_groups=len(sizes), group_sizes=dict(sorted(sizes.items())),
            max_group_fraction=max(sizes.values(), default=0)/max(len(eligible), 1),
            top10_groups=sizes.most_common(10),
            capped_parent_totals={str(cap): sum(min(cap, n) for n in sizes.values())
                                  for cap in (4, 6, 8, 10, 12)},
            reactions=sum(len(p['cache_indices']) for p in ps))
    out['official_formula_overlaps'] = {a+'__'+b: sorted(sets[a]&sets[b])
                                        for a in sets for b in sets if a<b}
    return out


def reserve_split(parents, cap, development=60, screen_reserve=350, seed='v1a-20261007',
                  development_cap=4):
    """Reserve unseen formulas before fitting; final screen N is set after power.

    Cap selects parents by hash within a formula, using only catalogue inventory.
    All other parents of a reserved formula are excluded from training too.
    """
    eligible = defaultdict(list)
    for p in parents:
        if p['official_split']==0 and p['known_channels']>=2:
            eligible[p['split_group']].append(p['parent_id'])
    eligible = {g: sorted(ids, key=lambda p: digest([seed,p]))[:cap]
                for g, ids in eligible.items()}
    groups = sorted(eligible, key=lambda g: digest([seed,g]))
    chosen = {}
    cursor = 0
    for name, target in [('development', development), ('screen_reserve', screen_reserve)]:
        selected, gs = [], []
        while len(selected)<target and cursor<len(groups):
            g = groups[cursor]
            if name == 'screen_reserve' and len(selected)+len(eligible[g])>target:
                break
            cursor += 1
            gs.append(g)
            selected.extend(eligible[g][:development_cap] if name=='development' else eligible[g])
        if len(selected)<target-(cap-1 if name=='screen_reserve' else 0):
            raise ValueError(f"insufficient groups for {name}: {len(selected)} < {target}")
        chosen[name] = dict(parent_ids=selected, formula_groups=gs)
    reserved = set(chosen['development']['formula_groups']+chosen['screen_reserve']['formula_groups'])
    train = sorted((p for p in parents if p['official_split']==0 and p['split_group'] not in reserved),
                   key=lambda p: p['parent_id'])
    chosen['train'] = dict(parent_ids=[p['parent_id'] for p in train],
                          formula_groups=sorted({p['split_group'] for p in train}),
                          cache_indices=sorted(i for p in train for i in p['cache_indices']))
    chosen.update(seed=seed, max_parents_per_formula=cap,
                  development_max_parents_per_formula=development_cap, screen_frozen=False,
                  official_val_and_test='excluded; previously used in M0',
                  existing_M0_checkpoints='development_only; retrain on this split')
    chosen['split_hash'] = digest(chosen)
    return chosen
