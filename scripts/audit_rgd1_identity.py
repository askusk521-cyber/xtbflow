#!/usr/bin/env python3
"""Audit RGD1 CSV↔HDF5 identity joins without positional assumptions."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
 d=hashlib.sha256()
 with path.open('rb') as h:
  for b in iter(lambda:h.read(1024*1024),b''): d.update(b)
 return d.hexdigest()

def decode(x: Any) -> str:
 return x.decode() if isinstance(x,(bytes,bytearray)) else str(x)

def side_atoms(smiles: str, Chem: Any):
 atoms=[]
 for part in smiles.split('.'):
  mol=Chem.MolFromSmiles(part, sanitize=False)
  if mol is None: return None
  atoms.extend(mol.GetAtoms())
 return atoms

def mapped_expected(smiles: str, Chem: Any):
 atoms=side_atoms(smiles,Chem)
 if atoms is None: return None,'smiles_parse_fail'
 maps=[int(a.GetAtomMapNum()) for a in atoms]
 if any(x<=0 for x in maps): return None,'map_id_invalid'
 if len(set(maps))!=len(maps): return None,'duplicate_map_id'
 ordered=sorted(maps)
 if ordered != list(range(ordered[0],ordered[-1]+1)): return None,'missing_map_id'
 return [int(next(a for a in atoms if int(a.GetAtomMapNum())==m).GetAtomicNum()) for m in ordered],None

def bond_map(smiles: str, Chem: Any):
 atoms=[]; bonds={}
 for part in smiles.split('.'):
  mol=Chem.MolFromSmiles(part, sanitize=False)
  if mol is None: return None
  atoms.extend(mol.GetAtoms())
  for b in mol.GetBonds():
   i,j=sorted((int(b.GetBeginAtom().GetAtomMapNum()),int(b.GetEndAtom().GetAtomMapNum())))
   bonds[(i,j)]=float(b.GetBondTypeAsDouble())
 return bonds

def main():
 p=argparse.ArgumentParser(); p.add_argument('--cache',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--extras-output',type=Path,required=True); args=p.parse_args()
 try:
  import h5py
  from rdkit import Chem
 except ImportError as e: raise SystemExit(str(e))
 csv_path=args.cache/'RGD1CHNO_AMsmiles.csv'; h5_path=args.cache/'RGD1_CHNO.h5'; info_path=args.cache/'DFT_reaction_info.csv'; randp=args.cache/'RandP_smiles.txt'
 paths=[csv_path,h5_path,info_path,randp]
 if any(not x.is_file() for x in paths): raise SystemExit('missing RGD1 asset')
 assets={x.name:{'size_bytes':x.stat().st_size,'sha256':sha256(x)} for x in paths}
 rows=[]; csv_ids=[]; counts=Counter(); reasons=Counter(); examples={}
 with csv_path.open(newline='',encoding='utf-8-sig') as h:
  reader=csv.DictReader(h); fields=list(reader.fieldnames or [])
  for row in reader:
   rid=row.get('reaction',''); rows.append(row); csv_ids.append(rid)
 counts['csv_rows']=len(rows); counts['csv_unique_ids']=len(set(csv_ids)); counts['csv_duplicate_ids']=len(csv_ids)-len(set(csv_ids))
 h5_ids=set()
 with h5py.File(h5_path,'r') as h5: h5_ids=set(str(k) for k in h5.keys())
 counts['hdf5_groups']=len(h5_ids); counts['csv_ids_missing_from_hdf5']=len(set(csv_ids)-h5_ids); counts['hdf5_only_groups']=len(h5_ids-set(csv_ids))
 extras=sorted(h5_ids-set(csv_ids)); args.extras_output.parent.mkdir(parents=True,exist_ok=True); args.extras_output.write_text(''.join(json.dumps({'record_id':x,'reason':'hdf5_only_no_identity_join'})+'\n' for x in extras),encoding='utf-8')
 def fail(rid,reason,detail=None):
  reasons[reason]+=1; examples.setdefault(reason,{'record_id':rid,'detail':detail or {}})
 with h5py.File(h5_path,'r') as h5:
  for row in rows:
   rid=row.get('reaction','');
   if rid not in h5: fail(rid,'csv_id_missing_from_hdf5'); continue
   g=h5[rid]; counts['identity_joined']+=1
   for key in ('reactant','product'):
    expected,err=mapped_expected(row[key],Chem)
    if err: fail(rid,err,{'side':key}); continue
    obs=[int(x) for x in g['elements'][()].tolist()]
    if expected != obs: fail(rid,'atomic_inventory_or_order_mismatch',{'side':key,'expected':expected,'observed':obs})
    else: counts[f'{key}_map_order_exact']+=1
   rg=[int(x) for x in g['elements'][()].tolist()]
   if tuple(g['RG'].shape)!=(len(rg),3) or tuple(g['PG'].shape)!=(len(rg),3) or tuple(g['TSG'].shape)!=(len(rg),3): fail(rid,'geometry_shape_mismatch')
   if not all(k in g for k in ('Rsmiles','Psmiles','RG','PG','TSG','elements')): fail(rid,'required_hdf5_field_missing')
   rb=bond_map(row['reactant'],Chem); pb=bond_map(row['product'],Chem)
   if rb is not None and pb is not None:
    edits=sum(pb.get(k,0.0)!=rb.get(k,0.0) for k in set(rb)|set(pb)); counts['bond_change']+=int(edits>0); counts['no_bond_change']+=int(edits==0)
   counts['charge_missing']+=1; counts['multiplicity_missing']+=1
 # Aromaticity is deliberately not inferred from SMILES text here.  The
 # identity audit reports mapping/order and explicit source fields only.
 report={'schema':'xtbflow-rgd1-identity-audit/v1','source_dataset':'RGD1','source_revision':'figshare:21066901@v6','assets':assets,'locators':{'mapped_csv':'external-cache:rgd1/figshare-21066901-v6/RGD1CHNO_AMsmiles.csv','geometry_hdf5':'external-cache:rgd1/figshare-21066901-v6/RGD1_CHNO.h5','reaction_info':'external-cache:rgd1/figshare-21066901-v6/DFT_reaction_info.csv','endpoint_mapping':'external-cache:rgd1/figshare-21066901-v6/RandP_smiles.txt'},'counts':dict(sorted(counts.items())),'reason_counts':dict(sorted(reasons.items())),'reason_examples':examples,'hdf5_only_ids_jsonl':args.extras_output.name,'mapping_evidence':'map_id-sorted atomic numbers exactly match HDF5 elements for joined rows; HDF5 has no map IDs, so same-element identity remains unresolved without source correspondence/graph-isomorphism evidence.','admission':{'status':'quarantine','admitted_count':0,'blocking_reasons':['charge_missing','multiplicity_missing','license/protocol gate unresolved until source terms are pinned','same-element atom correspondence requires graph-isomorphism evidence']}}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print(json.dumps(report['counts'],sort_keys=True))
if __name__=='__main__': main()
