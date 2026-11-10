# S3 failed provider work

This change targets #186 and depends on the S2 identity branch based on #179.
The real runner writes a durable request-fingerprint journal before invocation.
Started exceptions retain synchronous native-backend partial calls and measured
wall/process CPU time even if no physical feedback exists. The outer runtime
ledger uses the same charged physical operation; these are two views of one
call and must not be summed. Heuristic event-provider calls stay separate.

Unknown generic failures have `measured_cost=null`, a separately labelled
conservative reservation charge, and unknown termination. They halt the shared
budget. A timeout is not proof that an external worker stopped. Recovery needs
external termination/consumption evidence; this runner does not resume it.
Known overruns are charged in full, reservations close, and later launch is
blocked. If outer settlement fails, a separate settlement-failed receipt keeps
the event and adapter ledger and certification cannot start.

The journal is single-writer; recovery requires a fresh budget and restores
charges/seen identities without double counting. An interrupted started entry
restores its conservative charge and halts. Optional in-memory use remains for
fixtures; live execution supplies the durable path. This does not turn arbitrary
unbounded third-party providers into safely cancellable jobs.

Tests cover prelaunch refusal, one/multiple started calls, exception, timeout,
cancellation, malformed output, overrun, duplicate/recovery and two-layer live
ledger reconciliation. n2 software-only envelope: one CPU thread, 600 seconds
per test command, 64 MiB logs, zero real calculator/model/training calls and
no automatic retry. No historical 500/402 ledger or verdict is modified.

## Review follow-up: complete arm recovery and stop propagation

Review found that adapter-level duplicate suppression did not protect the
outer ledger/candidate or certification entry. The prior tests did not qualify
that boundary. The arm now creates an exclusive durable launch receipt, saves
the candidate before certification, and binds the completed result to the
config/snapshot/input/protocol/runner and all arm artifact hashes. Verified
replay returns the saved candidate and cumulative ledger with zero **new** cost;
it does not invoke either backend or rewrite any prior file. A partial/legacy
attempt, changed identity, unknown termination or unsettled work is blocked,
with existing files and any saved candidate retained. No duplicate charge is
turned into a historical charge. The top-level CLI refuses nonempty output.

The certification boundary checks the shared guidance halt state, unknown
termination, unfinished settlement, overrun/refusal and failed provider status.
Any such stop keeps actual/conservative costs and returns a structured blocked
reason before creating a certification calculator/token. A settlement error
retains its event separately, rather than publishing an empty successful ledger.
The run loop also stops before another arm when its guidance budget has halted.

New tests exercise `_run_arm` twice, interruption at certification entry, a
two-second fake measurement against a one-second wall cap, an unfinished journal,
settlement failure and changed context/artifact identities. They use no real
calculator or trained checkpoint. They do not imply that these defects affected
the historical 500/402 outcomes. The same finite software envelope above applies.
