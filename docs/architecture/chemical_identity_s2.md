# S2 chemical identity repair

For #185, based on #179 at `5ccba76e668047607d24480a3f8392cd296b1874`.
This narrow branch does not merge the trained/reference implementation. S0 is
the separately reviewed #190 contract; its diagnostic parent and seed are fixed.

Both principal-axis bases are now right handed, so all 24 orientation transforms
and all 72 enumerated placements are proper rotations. Labelled atom rows and
intracomponent distances/parity are preserved. Version 2 records the enumeration
size and that pose selection is deterministic and seed-independent. Degenerate
eigenframes have no unique physical orientation; only finite SO(3) transforms
are promised. No pose-diversity or molecular accuracy claim follows.

LiveXTB already builds from current snapshot geometry. It now also verifies the
returned calculator input hash, charge/spin, protocol and force atom count.
Snapshot ID, original reactant input hash and current calculator hash are distinct.
Regression tests send two snapshots through a recording fake backend and reject
a stale result. Coordinates remain angstrom, energy hartree and forces
hartree/angstrom. No vector/Hessian is handed off by this adapter.

The old PVO provider has a different code lineage and receives a separate fix;
its one-step historical output is not rewritten. The four-dimensional event
score fixture remains a software fixture. A score change at fixed geometry
cannot establish chemical feedback; the trained bridge and decoder require
their own frozen diagnostic evidence.

Validation envelope: n2 `xtbflow`, CPU only, one thread, 600 wall seconds per
focused/full pytest and reusable unittest invocation, 64 MiB log cap, zero
external retries, zero training and zero real calculator calls. Tests use fake
backends; all raw historical results and hashes remain unchanged. No evidence
shows that the reflection counterexample caused the historical same_endpoint.
