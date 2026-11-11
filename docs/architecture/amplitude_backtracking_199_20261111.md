# Bounded residual backtracking — #199

The installed `bounded_backtrack` option uses the shared post-Euler Injection
and returns its updated state, or the exact unpushed base. Query, updated BE,
next clocks, reflected G and u stay fixed throughout the trials. Cached #198
values are accepted only under the exact report/cache hashes; no baseline
forward or derivative is repeated. All new model and E/F attempts use #198's
durable meter and process termination boundary.

The single registered change is amplitude. alpha_cap=0.1 Å per geometry clock;
normalize G by per-atom RMS and use the old max-atom common scaling. Try lambda
1, 1/2, 1/4 and select the first valid trial with squared centered-force residual
<=0.99 baseline, positive endpoint displacement dot G, <=10% local linearization
error and endpoint RMS <= RMS(delta0). The JVP per unit L2 is multiplied by the
actual scaled delta L2, not by the RMS dose. The zero-force L2 floor is 1e-10
Hartree/Å. An unqualified/nonpositive G skips without a direction change.

The fixed step is the shared lambda=1 trial. Trial geometry never advances the
base/event between attempts. Rejections and SCF failures remain charged. Missing
report or unknown termination/settlement stops the stage. The no-push outcome
is finite and explicit. Force residual is not a saddle or reaction certificate.

Use the same n2 source/asset setup as #198, a clean exact commit and new output
folder. Run `scripts/optimize_guidance_amplitude.py` with its same five asset
arguments plus `--diagnostic "$XTB_198_RAW" --out "$XTB_NEW_OUT"` inside
`srun --partition=main --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G
--time=00:15:00 timeout 898`, after activating xtbflow and setting all CPU
thread variables to 1 and CUDA_VISIBLE_DEVICES empty. Retain at most 64 MiB.
The committed manifest limits new work to 12 E/F, 9 forwards, zero new JVP,
zero quantum Hessian/HVP/reference/training/GPU, 900 wall/process+child CPU
seconds. Scientific code/report SHA and full per-attempt costs are published
with the result; all private tensors stay on n2.

Only a nonzero accepted update admits the registered #188 pilot. New actual
intervened states in that pilot require freshly charged forces/JVPs. If all
three states fail, retain all trials and report the specific cause without
running a four-arm zero-intervention chemistry comparison.
