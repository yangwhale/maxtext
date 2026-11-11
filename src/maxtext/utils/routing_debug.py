# Copyright 2023–2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""debug_routing_digest: record every routed top-k selection on the host, to count how many (token, expert) slots a
rematerialized backward re-selects differently from the forward (remat_policy=full recomputes the router).

Each call of RoutedMoE.get_topk appends its (per-device) top-k indices to LOG through jax.debug.callback; a recompute in
the backward pass appends again. count_reselections pairs every call with the earlier call it repeats (the earlier call
of the same shape whose selections agree best, if they agree on more than half the slots; a different layer agrees on
far fewer) and counts the slots that differ. Diagnostics only."""

import numpy as np

LOG: list = []


def record(indices):
  LOG.append(np.array(indices, copy=True))


def count_reselections(log=None, min_agree=0.5) -> dict:
  """{calls, repeats, slots_compared, slots_differ, fraction}; a call is a repeat when an earlier call agrees on more
  than min_agree of its slots (compared as per-token sets of experts)."""
  log = LOG if log is None else log
  flat = [np.sort(np.asarray(x).reshape(-1, np.asarray(x).shape[-1]), axis=-1) for x in log]
  used = set()
  repeats, compared, differ = 0, 0, 0
  for i, x in enumerate(flat):
    best, best_j = -1.0, None
    for j in range(i):
      if j in used or flat[j].shape != x.shape:
        continue
      agree = float(np.mean(np.all(flat[j] == x, axis=-1)))
      if agree > best:
        best, best_j = agree, j
    if best_j is not None and best > min_agree:
      used.add(best_j)
      used.add(i)
      repeats += 1
      tok_differ = ~np.all(flat[best_j] == x, axis=-1)
      # slots: experts of a token that are not in the other selection
      d = sum(len(set(a) - set(b)) for a, b in zip(flat[best_j][tok_differ], x[tok_differ]))
      compared += x.size
      differ += int(d)
  return {
      "calls": len(flat),
      "repeats": repeats,
      "slots_compared": compared,
      "slots_differ": differ,
      "fraction": differ / compared if compared else None,
  }
