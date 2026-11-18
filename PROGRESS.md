# Progress

## Handoff 2026-11-04

Reconstructed on 2026-11-06 by the terminal-cleanup coordinator from local Git state.
The original shutdown prompt hit the weekly limit; its pane was closed on November 4.

State:
- Checkout: fix/kf-endpoint-padding; base f72d85e (2026-10-19).
- That commit changes src/dftpy/functional/kedf/hc.py and lwt.py and adds tests/unit/test_kf_endpoint_padding.py.
- Its subject is "fix: relative kF checkpoint padding in LWT and HC".
- The tracked checkout was clean before this documentation append.
- This reconstruction did not run the tests, revalidate numerical behavior, or check current upstream PR status.

Owned jobs and timers:
- This reconstruction started no job, model worker, timer or cron.
- The closed session's background-job ownership and timer cancellation were not independently verified.
- Do not infer that a scientific job finished from the pane closure.

Next action:
- Read the existing patch and current upstream review state before doing any further development.
- Preserve the patch and its tests; do not broaden the fix during session recovery.
- Contact, PR comments, publication and upstream submission require their existing authorization.
- origin points to Quantum-MultiScale/DFTpy; this handoff is a local documentation commit, not an upstream push.

## Handoff 2026-11-06

Verified read-only with gh (account doubleMcyber) and `git fetch origin`:
- PR #16 https://github.com/Quantum-MultiScale/DFTpy/pull/16 is MERGED into dev on 2026-11-03 06:02 UTC by shaoxc (merge commit e686db1).
- Only feedback: one "Thank you" comment from shaoxc. No reviews, no change requests, no related open issues.
- origin/dev head is still e686db1; hc.py, lwt.py and tests/unit/test_kf_endpoint_padding.py are byte-identical to f72d85e.
- doubleMcyber has no other PRs (open or closed) on the repo.

State: the kF padding patch workflow is complete. Nothing actionable remains.
Owned jobs/timers: none.
Next action: owner picks the next fix (if any) from the vetted backlog; re-verify that item against current origin/dev first.
Gates: posting, new PRs and pushes to public upstream need explicit owner authority. This branch's PROGRESS.md commits are local-only.

## Handoff 2026-11-18

Six fixes are committed on local branches, each cut from origin/dev e686db1, each with a regression test that fails on dev and passes with the fix. Nothing is pushed and no PR is open; pushing and PRs need the owner's go.

| Branch | Commit | Fix |
| --- | --- | --- |
| fix/mpi-nonlocal-energy | c2bcdd7 (amended after Astra review) | Nonlocal/GGA KEDF and rVV10 energies were rank-local under MPI; WT+LDA+Hartree total on 2 ranks was 4.273 Ha vs 0.354 Ha serial |
| fix/lbfgs-two-loop | 716ea15 | L-BFGS two-loop skipped the oldest pair; history held Bound+1 pairs |
| fix/tile | e3272b0 | grid.tile() TypeError; field.tile() dropped the tiled grid |
| fix/gga-params-mutation | 1f22170 | GGAFs overwrote GGA_KEDF_list defaults; GGA() extended the caller's list |
| fix/tn-spin-direction | c608a64 | TN direction of spin channels > 0 replaced by spin 0 |
| fix/td-vector-potential-kick | 0b889e7 | np.zeros[3] crash; sigma1 branch used sigma |

Verification (2026-11-18, .venv Python 3.11.9, NumPy 2.4.6, Open MPI 5.0.11, mpi4py 4.1.2):
- integration/all-fixes (local merge of all six): `pytest tests` 67 passed, 3 skipped; `mpirun -n 2 python -m pytest --with-mpi tests` exit 0.
- MPI branch alone also passed on 3 ranks, including the two existing MPI test files.
- Environment change: installed Homebrew open-mpi and fftw, and mpi4py, mpi4py-fft, pytest-mpi into .venv.

Not done: no independent review of the diffs. Ideation with gpt-6.2-astra failed (model not supported on the ChatGPT account through Codex); Fable 5.2 on the productivemarlon account (~/.claude-c) was over its limit until 13:00 PST. The coordinator session ran on the main account.

Next action: owner decides whether to push and open PRs, and in what order (suggest MPI energy first, then L-BFGS). Remaining backlog items are in the session memory file dftpy-verified-roi-backlog.
Owned jobs/timers: none.

Addendum 2026-11-18 (later): gpt-6.2-astra review ran (read-only). Applied to fix/mpi-nonlocal-energy: LWT line-integral energy reduced, NG/MG energies use mp.asum (safe for ndarray energy densities), Barrier removed from the new test. Confirmed by probe and NOT fixed: KEDF stress under MPI is over-counted on dev (TF diagonal doubled on 2 ranks) and stays wrong on the branch; needs its own PR. integration/all-fixes predates the amend. Astra's other new findings (unverified by the coordinator): spin-polarized KEDF potential averaged across channels (kedf/__init__.py:178), stale stress energy cache (:216), TotalFunctional overwriting same-type components, td/utils.py kick mask mutating cached grid coordinates, TF energy density missing x.
