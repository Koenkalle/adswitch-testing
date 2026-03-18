# adswitch-testing

C++ implementation of the **ADSWITCH** algorithm from:

> Auer & Chiang, "Adaptively Tracking the Best Bandit Arm with an Unknown Number of Distribution Changes", COLT 2019.
> https://proceedings.mlr.press/v99/auer19a/auer19a.pdf

---

## Overview

ADSWITCH is an online learning algorithm for the non-stationary multi-armed bandit problem. The algorithm runs in *epochs*: within each epoch it uses a UCB index to select arms, and after every pull it runs a change-detection test. When a statistically significant distribution shift is detected, a new epoch begins and all arm statistics are reset.

Two implementations are provided:

| File | Description | Time per pull | Total time |
|------|-------------|--------------|------------|
| `include/adswitch_basic.hpp` | Basic ADSWITCH | O(n) | O(T²) |
| `include/adswitch_fast.hpp` | ADSWITCH + Remark 3 speedup | O(log n) | O(T log T) |

### Remark 3 speedup

The basic algorithm checks all *n − 1* split points of an arm's reward history, giving O(n) work per pull. Remark 3 from the paper observes that it suffices to check only O(log n) split points (powers of 2) because any change at position *c* is covered by the split at `s = 2^⌊log₂(c)⌋ ≤ c`. Combined with O(1) range-sum queries from a running prefix array, this reduces the per-pull work to O(log n) and the total work from O(T²) to O(T log T), while preserving the same regret guarantees up to logarithmic factors.

---

## Repository structure

```
adswitch-testing/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── adswitch_basic.hpp    # Basic ADSWITCH (O(n) CDTest per pull)
│   ├── adswitch_fast.hpp     # Fast ADSWITCH with Remark 3 (O(log n) per pull)
│   └── environments.hpp      # Five test environment types
├── src/
│   └── main.cpp              # Experiment runner comparing both algorithms
└── tests/
    └── test_adswitch.cpp     # Unit tests (no external framework required)
```

---

## Test environments

| Class | Description |
|-------|-------------|
| `StochasticEnv` | Fixed Bernoulli distributions (stationary baseline) |
| `DriftingEnv` | Means perform a Gaussian random walk (slow drift) |
| `SharpSwitchEnv` | User-defined phases with abrupt distribution changes |
| `BigSwitchEnv` | Randomly generated phases with large gaps (default Δ = 0.4) |
| `SmallSwitchEnv` | Randomly generated phases with small gaps (default Δ = 0.1) |

---

## Build & run

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Run unit tests
./adswitch_tests

# Run experiment comparison
./adswitch_runner
```

### Example output

```
ADSWITCH Experiment Runner
Arms K=5  Horizon T=10000

=== Stochastic ===
Algorithm       Regret    Resets         Time
------------------------------------------------
Basic            400.3         0      312.07 ms
Fast (R3)        400.3         0        2.14 ms

=== Sharp-switch ===
Algorithm       Regret    Resets         Time
------------------------------------------------
Basic            632.4         3      113.73 ms
Fast (R3)        730.8         3        1.89 ms
```

The Fast variant is consistently **10–150× faster** than Basic while achieving comparable cumulative regret across all environment types.

---

## Algorithm parameters

Both `ADSwitchBasic` and `ADSwitchFast` share the same interface:

```cpp
// K: number of arms, alpha: UCB exploration coefficient, delta: confidence
ADSwitchBasic algo(K, /*alpha=*/4.0, /*delta=*/0.05);

int arm = algo.select_arm(t);   // call before each pull
bool reset = algo.update(arm, reward);  // returns true if epoch reset
```

**`alpha`** – UCB exploration coefficient (recommended: 2–4).  
**`delta`** – Confidence parameter for the change-detection threshold. Smaller values reduce false alarms at the cost of slower detection.
