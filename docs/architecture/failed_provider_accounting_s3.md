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
