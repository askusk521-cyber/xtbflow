# Bounded quantum campaign after the first tautomer pilot

The owner asked whether GQC is accurate and authorized launching the previously
proposed calculations while compute is available. This campaign separates
implementation checks, chemical-model sensitivity and reaction discovery.
It reuses the qualified compiled runtime; CUDA PBE/PBE0/r2SCAN forces remain
outside production use. No training or sealed labels are involved.

Live inspection found four GPUs, with two other GPU jobs active. This request
therefore raises our concurrency from one to **two GPU workers and two CPU
workers**, using finite Slurm arrays. Upper bounds: 48 allocated GPU hours,
512 allocated CPU-core hours, 20,000 quantum operations and 32 GiB retained
data. Each task has its own stricter time/call/memory limits. These are ceilings,
not guaranteed consumption or scientific success. No automatic external retry.

## First wave: existing geometry, new questions

- Six already verified enol/carbonyl representatives are exposed development
  inputs, not new blind test cases. Reuse them rather than repeating their
  completed optimizations or frequencies.
- On six displaced geometries, check GQC B3LYP/def2-SVP forces against two
  finite-difference scales and independent CPU PySCF. On six original minima,
  repeat the comparison at def2-TZVP to check the larger-basis route.
- Check the previously unqualified GQC M06-2X route on the R6 pair; a failed
  force check is retained and does not authorize its use for optimization.
- Calculate six independent all-electron CCSD(T)/cc-pVDZ single points with
  CPU PySCF. This is a finite-basis reference with single-reference assumptions,
  not experimental truth or a complete-basis result.
- Calculate twelve C-PCM electrostatic single points at B3LYP/def2-TZVP:
  six geometries each in declared water/acetonitrile dielectric models.
  They are not solution free energies or rate constants.

First-wave manifest preparation runs on n2 from the completed pilot's raw
identity-checked coordinates and their pinned hashes. The manifest records
source/configuration identities and individual task caps. It must not be
modified after submission. The worker preserves started/completed/failed
operation journals and a structured report; a scheduler kill leaves an
unsettled operation, never a zero-cost `not_run`.

## Subsequent wave: concrete candidates A-D

Prepare a separate immutable input/endpoint manifest before submitting its
jobs. Include the missing conjugated carbonyl connectivities, a bounded set
of extra conformers, and independently constructed DA/Michael/hemiacetal
opening endpoints. Preserve explicit atom identities by graph construction.
No missing source mapping, environment or state is silently filled in.

The intended eight bounded path searches cover: R6 tautomerization without and
with one explicit water; R6 DA capture; R51 DA/reverse-DA and product ring
opening; competing DA and neutral aza-Michael hypotheses for the R3066-style
amino-furan/maleimide pair; and water-assisted R51 product opening. These are
proposed mechanisms, not established reactions. Numerical gates and valid
endpoints are prerequisites. Raw path maxima require TS refinement, projected
frequencies, bidirectional IRC and endpoint identity before a connection claim.

Explicit-water comparisons use the same composition on both ends. Different
water-count total energies are not directly compared. Bulk-water effects,
hydrolysis networks and solution kinetics remain conditional on actual path
and thermodynamic evidence, not on the mere presence of a water cluster.

Final submission evidence will include every task index and job/dependency ID,
the exact execution source, manifest/configuration hashes and actual startup
status. Queued, running, failed, blocked by an upstream numerical gate and
completed are reported separately. A post-array CPU collector will write a
final status snapshot without launching extra quantum calculations.

Base/head: #207 (`codex/gqc-tautomer-20261114`) →
`codex/qc-campaign-20261114`. No PR is merged by this campaign.

## Frozen second wave

The n2 preparation smoke produced 28 opt/frequency pairs, two additional
derivative gates (nitrogen-containing bimolecular system and explicit-water
cluster), eight path chains, six matched gas B3LYP/def2-TZVP single points,
and two R6 all-electron CCSD(T)/cc-pVTZ single points. The combined upper
bounds are 40.8334 GPU hours, 451.334 CPU-core hours and 18,156 operations.
The preparation uses RDKit ETKDGv3/MMFF with fixed seeds, preserves explicit
hydrogen atom maps through bond edits, and records missing/failed preparations.
Extra conformers are fixed new starts, not a complete conformational ensemble.
The extra R51 chiral carbonyl starts include both mirror configurations.

GQC B3LYP/def2-SVP drives endpoint optimization and seven-image NEB. PySCF CPU
B3LYP/def2-SVP supplies all Hessians, TS refinement and both IRC directions.
An initial TS Hessian must also pass a gradient finite-difference directional
check. Every native operation remains within the path's 1,600-operation and
150-minute allocation. All-negative-frequency/gradient/identity gates are
strict; an unresolved endpoint blocks its dependent path. NEB need not have
converged to supply a candidate for independent TS refinement, but its status
is retained and never itself certifies a reaction.

Each finite cluster uses one water placement; the neutral Michael hypothesis
does not test all stepwise/zwitterionic possibilities. Double-ended structures
are deliberate chemical hypotheses, not reaction-blind model inputs. Local
electronic barriers reference the verified finite-complex minimum and cannot
be reported as solution activation free energies or bulk hydrolysis rates.

Failed work is journaled before exceptions propagate. Completed child exit
codes are retained separately from scientific status. A scheduler hard kill
may leave `started`/unsettled work; the collector preserves it. Disk limits
(4 GiB/task and 32 GiB/campaign) are checked between operations, with a
campaign check every ten operations, not enforced as filesystem quotas.

Primary software references:
[PySCF CCSD(T)](https://pyscf.org/user/cc.html),
[PySCF solvent models](https://pyscf.org/user/solvent.html),
[ASE NEB](https://docs.ase-lib.org/ase/neb.html),
[geomeTRIC TS](https://geometric.readthedocs.io/en/latest/transition.html),
[geomeTRIC IRC](https://geometric.readthedocs.io/en/latest/irc.html).
