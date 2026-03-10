#pragma once

// ADSWITCH – Basic implementation (O(T^4) total work)
// Full, faithful implementation of Algorithm 1 from:
//   Auer & Chiang, "Adaptively Tracking the Best Bandit Arm with an Unknown
//   Number of Distribution Changes", COLT 2019.
//   https://proceedings.mlr.press/v99/auer19a/auer19a.pdf
//
// Every section of code is annotated with the corresponding line number(s)
// from the pseudocode reproduced below.  The text description in the paper
// body takes precedence over the pseudocode where they differ.
//
//   Algorithm 1  ADSWITCH
//    1: Input: Time horizon T.
//    2: Initialization: ℓ ← 0, t ← 0.
//    3: Start a new episode:
//    4:   ℓ ← ℓ + 1.
//    5:   Set start of the episode t_ℓ ← t + 1.
//    6:   GOOD_{t+1} = {1,…,K},  BAD_{t+1} = {}.
//    7: Next time step:
//    8:   t ← t + 1.
//    9:   Add checks for bad arms:
//   10:   For all a ∈ BAD_t, and all i ≥ 1 with 2^{-i} ≥ Δ̃_ℓ(a)/16,
//   11:     with probability √ℓ/(KT log T) add
//           S_t(a) ← S_t(a) ∪ (2^{-i}, ⌈2(log T)/ε²⌉, t).
//         (Note: the text states the probability is √ℓ/(KT log T) with no
//          ε factor; n_needed = ⌈2 log T / ε²⌉ = ⌈2^{2i+1} log T⌉.)
//   12: Select an arm:
//   13:   Select a_t = argmin_a { τ : a ∉ {a_τ,…,a_{t-1}},
//                                     a ∈ GOOD_t ∨ S_t(a) ≠ {} }.
//         (Least-recently-pulled among eligible arms.)
//   14: Receive reward r_t.
//   15: Check for changes of good arms:
//   16:   If there is a ∈ GOOD_t and t_ℓ ≤ s₁ ≤ s₂ and t_ℓ ≤ s ≤ t such
//         that condition (3) holds, then start a new episode.
//   18: Check for changes of bad arms:
//   19:   If there is a ∈ BAD_t and t_ℓ ≤ s ≤ t such that condition (4)
//         holds, then start a new episode.
//   21: For a ∈ BAD_t,
//         S_{t+1}(a) ← {(ε, n, s) ∈ S_t(a) : n_{[s,t]}(a) < n}.
//   22: Evict arms from GOOD_t:
//   23:   BAD_{t+1} = BAD_t ∪ {a ∈ GOOD_t | ∃ s ≥ t_ℓ for which (1) holds}.
//   24:   For evicted arms a ∈ BAD_{t+1}\BAD_t, calculate μ̃_ℓ(a) and Δ̃_ℓ(a)
//         according to (2), and set S_{t+1}(a) ← {}.
//   25:   GOOD_{t+1} = {1,…,K} \ BAD_{t+1}.
//   26: Continue with the next time step.
//
// Conditions (from paper body):
//   (1) max_{a'∈GOOD_t} μ̂_{[s,t]}(a') − μ̂_{[s,t]}(a) > √(C₁ log T / n_{[s,t]}(a))
//       for some s ≥ t_ℓ with n_{[s,t]}(a) ≥ 2.
//       (Implemented using the full episode window s = t_ℓ; C₁ = 4.)
//   (2) μ̃_ℓ(a) ← μ̂_{[s,t]}(a),
//       Δ̃_ℓ(a) ← max_{a'∈GOOD_t} μ̂_{[s,t]}(a') − μ̂_{[s,t]}(a).
//   (3) |μ̂_{[s₁,s₂]}(a) − μ̂_{[s,t]}(a)| > √(2 log T / n_{[s₁,s₂]})
//                                           + √(2 log T / n_{[s,t]})
//       for some t_ℓ ≤ s₁ ≤ s₂ and t_ℓ ≤ s ≤ t  (non-overlapping blocks).
//       Basic version: checks ALL O(n³) triples → O(T⁴) total.
//   (4) |μ̂_{[s,t]}(a) − μ̃_ℓ(a)| > Δ̃_ℓ(a)/4 + √(2 log T / n_{[s,t]}(a))
//       for some s in the current episode (s = creation time of the triple).
//
// See adswitch_fast.hpp for the Remark 3 O(T(log T)²) variant.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

class ADSwitchBasic {
public:
    // -----------------------------------------------------------------------
    // Algorithm 1, Line 1: Input: Time horizon T.
    //   K     : number of arms
    //   T     : total time horizon
    //   delta : unused in threshold formulas (kept for API consistency);
    //           thresholds follow the paper's log-T calibration exactly
    // -----------------------------------------------------------------------
    ADSwitchBasic(int K, int T, double delta = 0.05, unsigned seed = 42)
        : K_(K), T_(T), delta_(delta), rng_(seed),
          log_T_(std::log(static_cast<double>(std::max(T, 2)))) {
        if (K_ <= 0) throw std::invalid_argument("ADSwitchBasic: K must be > 0");
        if (T_ <= 0) throw std::invalid_argument("ADSwitchBasic: T must be > 0");
        if (delta_ <= 0.0 || delta_ >= 1.0)
            throw std::invalid_argument("ADSwitchBasic: delta must be in (0,1)");

        // Algorithm 1, Line 2: Initialization ℓ ← 0, t ← 0.
        // start_episode() below increments ell_ to 1.

        // Algorithm 1, Lines 3–6: Start the first episode.
        start_episode();
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 8–13: advance time, schedule bad-arm checks,
    // select the arm to pull.
    // -----------------------------------------------------------------------
    int select_arm(int /*t*/) {
        // Line 8: t ← t + 1.
        t_++;

        // Lines 9–11: Add checks for bad arms.
        schedule_bad_arm_checks();

        // Line 13: least-recently-pulled eligible arm.
        return select_eligible_arm();
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 14–26: receive reward, run all post-pull steps.
    // Returns true if a new episode was started (change detected).
    // -----------------------------------------------------------------------
    bool update(int arm, double reward) {
        if (arm < 0 || arm >= K_)
            throw std::out_of_range("ADSwitchBasic: arm index out of range");

        // Line 14: Receive reward r_t.
        last_pulled_[arm] = t_;
        history_[arm].push_back(reward);
        n_arm_[arm]++;
        sum_arm_[arm] += reward;

        // Advance pull counters for all active check triples on this arm.
        for (auto& ck : S_[arm]) {
            if (ck.n_pulled < ck.n_needed) {
                ck.n_pulled++;
                ck.sum_rewards += reward;
            }
        }

        // Lines 15–17: Condition (3) – check for changes of good arms.
        // Only the arm just pulled has a new observation; other good arms'
        // histories are unchanged, so we only need to test this arm.
        if (is_good_[arm] && condition_3(arm)) {
            start_episode();
            num_resets_++;
            return true;
        }

        // Lines 18–20: Condition (4) – check for changes of bad arms.
        if (!is_good_[arm]) {
            for (const auto& ck : S_[arm]) {
                if (condition_4(arm, ck)) {
                    start_episode();
                    num_resets_++;
                    return true;
                }
            }
        }

        // Line 21: Remove completed check triples (n_pulled ≥ n_needed).
        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a]) {
                auto& sv = S_[a];
                sv.erase(
                    std::remove_if(sv.begin(), sv.end(),
                                   [](const CheckTriple& c) {
                                       return c.n_pulled >= c.n_needed;
                                   }),
                    sv.end());
            }
        }

        // Lines 22–25: Condition (1) – evict sub-optimal arms to BAD.
        evict_arms();

        return false;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    int    num_arms()    const { return K_; }
    int    num_resets()  const { return num_resets_; }
    int    episode()     const { return ell_; }
    int    epoch()       const { return ell_; }
    int    time()        const { return t_; }
    bool   is_good(int arm) const { return is_good_[arm]; }

    double mean(int arm) const {
        return n_arm_[arm] == 0 ? 0.0
                                : sum_arm_[arm] / static_cast<double>(n_arm_[arm]);
    }
    int    pulls(int arm)   const { return n_arm_[arm]; }

private:
    // -----------------------------------------------------------------------
    // A check triple (ε, n, s) for a BAD arm (lines 10–11, 19, 21).
    // -----------------------------------------------------------------------
    struct CheckTriple {
        double eps;         // precision level 2^{-i}              (line 11)
        int    n_needed;    // pulls required: ⌈2^{2i+1} log T⌉    (line 11)
        int    t_created;   // global round when triple was added   (line 11)
        int    n_pulled;    // pulls of this arm accumulated so far
        double sum_rewards; // sum of those rewards
    };

    // -----------------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------------
    int    K_, T_;
    double delta_;
    double log_T_;          // ln(T), precomputed

    int t_     = 0;         // Line 2: global round counter
    int ell_   = 0;         // Line 2: episode counter
    int t_ell_ = 0;         // Line 5: start round of current episode

    std::vector<bool>                    is_good_;      // Line 6: GOOD/BAD
    std::vector<int>                     last_pulled_;  // for line 13 selection
    std::vector<std::vector<double>>     history_;      // per-arm rewards in episode
    std::vector<int>                     n_arm_;        // pull counts in episode
    std::vector<double>                  sum_arm_;      // reward sums in episode
    std::vector<double>                  mu_tilde_;     // Line 24: μ̃_ℓ(a)
    std::vector<double>                  delta_tilde_;  // Line 24: Δ̃_ℓ(a)
    std::vector<std::vector<CheckTriple>> S_;           // Line 6/24: S_t(a)

    std::mt19937 rng_;
    int num_resets_ = 0;

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 3–6: start a new episode.
    // -----------------------------------------------------------------------
    void start_episode() {
        ell_++;                              // Line 4: ℓ ← ℓ + 1
        t_ell_ = t_ + 1;                    // Line 5: t_ℓ ← t + 1
        is_good_.assign(K_, true);          // Line 6: GOOD = {1,…,K}
        last_pulled_.assign(K_, 0);
        history_.assign(K_, {});
        n_arm_.assign(K_, 0);
        sum_arm_.assign(K_, 0.0);
        mu_tilde_.assign(K_, 0.0);
        delta_tilde_.assign(K_, 0.0);
        S_.assign(K_, {});                  // Line 6: BAD = {}, S = {}
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 9–11: probabilistic scheduling of check triples
    // for BAD arms.
    //
    // For each a ∈ BAD_t and each i ≥ 1 with ε_i = 2^{-i} ≥ Δ̃_ℓ(a)/16,
    // independently add triple (ε_i, ⌈2^{2i+1} log T⌉, t) to S_t(a)
    // with probability √ℓ / (K · T · log T).   [paper text, Section 2]
    // -----------------------------------------------------------------------
    void schedule_bad_arm_checks() {
        if (T_ < 2) return;

        // Line 11: probability = √ℓ / (K · T · log T)
        const double prob = std::sqrt(static_cast<double>(ell_)) /
                            (static_cast<double>(K_) *
                             static_cast<double>(T_) * log_T_);

        // Line 10: For all a ∈ BAD_t …
        for (int a = 0; a < K_; ++a) {
            if (is_good_[a]) continue;

            // Line 10: … and all i ≥ 1 with 2^{-i} ≥ Δ̃_ℓ(a)/16.
            for (int i = 1; i <= 60; ++i) {
                const double eps_i =
                    std::pow(2.0, -static_cast<double>(i));
                if (delta_tilde_[a] > 0.0 &&
                    eps_i < delta_tilde_[a] / 16.0) break;

                // Line 11: add triple with probability prob (no ε_i factor).
                if (std::bernoulli_distribution(std::min(1.0, prob))(rng_)) {
                    // n_needed = ⌈2 log T / ε_i²⌉ = ⌈2^{2i+1} · log T⌉
                    const double raw =
                        std::pow(2.0, 2 * i + 1) * log_T_;
                    const int n_needed =
                        std::min(T_, static_cast<int>(std::ceil(raw)));
                    S_[a].push_back({eps_i, n_needed, t_, 0, 0.0});
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Line 13: select the least-recently-pulled eligible arm.
    //
    // Eligible = GOOD_t  ∪  {a ∈ BAD_t : S_t(a) ≠ {}}.
    // Ties broken by lower arm index.
    // -----------------------------------------------------------------------
    int select_eligible_arm() const {
        int best     = -1;
        int min_last = std::numeric_limits<int>::max();

        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a] && S_[a].empty()) continue;
            if (last_pulled_[a] < min_last ||
                (last_pulled_[a] == min_last && a < best)) {
                min_last = last_pulled_[a];
                best     = a;
            }
        }

        // Fallback: all arms are BAD with empty S — pull least-recently-pulled.
        if (best == -1) {
            for (int a = 0; a < K_; ++a) {
                if (last_pulled_[a] < min_last ||
                    (last_pulled_[a] == min_last && (best == -1 || a < best))) {
                    min_last = last_pulled_[a];
                    best     = a;
                }
            }
        }
        return (best == -1) ? 0 : best;
    }

    // -----------------------------------------------------------------------
    // Condition (3): two-sample change-detection test for GOOD arm `arm`.
    //
    // Basic version – O(n³) per call: checks ALL non-overlapping triples
    // (s₁, s₂, s) where 0 ≤ s₁ ≤ s₂ < s ≤ n−1 (pull indices within the
    // current episode).  [s₁,s₂] is the historical block; [s,n−1] the recent.
    //
    // Paper: |μ̂_{[s₁,s₂]} − μ̂_{[s,t]}| > √(2 log T / n_{[s₁,s₂]})
    //                                      + √(2 log T / n_{[s,t]})
    // -----------------------------------------------------------------------
    bool condition_3(int arm) const {
        const auto& h = history_[arm];
        const int   n = static_cast<int>(h.size());
        if (n < 2) return false;

        // Build prefix sums over this arm's episode rewards.
        std::vector<double> prefix(n + 1, 0.0);
        for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + h[i];

        // Check all O(n³) triples (s₁, s₂, s) with s₂ < s.
        for (int s = 1; s < n; ++s) {
            // Recent block: pull indices [s, n-1].
            const int    recent_n    = n - s;
            const double recent_mean = (prefix[n] - prefix[s]) /
                                       static_cast<double>(recent_n);
            const double conf_recent =
                std::sqrt(2.0 * log_T_ / static_cast<double>(recent_n));

            for (int s1 = 0; s1 < s; ++s1) {
                for (int s2 = s1; s2 < s; ++s2) {
                    // Historical block: pull indices [s₁, s₂].
                    const int    hist_n    = s2 - s1 + 1;
                    const double hist_mean = (prefix[s2 + 1] - prefix[s1]) /
                                             static_cast<double>(hist_n);
                    const double thresh =
                        std::sqrt(2.0 * log_T_ /
                                  static_cast<double>(hist_n)) + conf_recent;

                    if (std::abs(hist_mean - recent_mean) > thresh)
                        return true;
                }
            }
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Condition (4): BAD-arm change-detection test.
    //
    // Once n_needed pulls have been accumulated since the check was created,
    // test whether the observed mean has drifted by more than Δ̃_ℓ(a)/4
    // plus a statistical confidence term.
    //
    // Paper: |μ̂_{[s,t]}(a) − μ̃_ℓ(a)| > Δ̃_ℓ(a)/4 + √(2 log T / n_{[s,t]})
    // -----------------------------------------------------------------------
    bool condition_4(int arm, const CheckTriple& ck) const {
        if (ck.n_pulled < ck.n_needed) return false;
        const double mu_recent =
            ck.sum_rewards / static_cast<double>(ck.n_needed);
        const double thresh =
            delta_tilde_[arm] / 4.0 +
            std::sqrt(2.0 * log_T_ / static_cast<double>(ck.n_needed));
        return std::abs(mu_recent - mu_tilde_[arm]) > thresh;
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 22–25: Condition (1) – evict sub-optimal GOOD arms.
    //
    // Paper: max_{a'∈GOOD_t} μ̂_{[s,t]}(a') − μ̂_{[s,t]}(a) > √(C₁ log T / n_{[s,t]}(a))
    // for some s ≥ t_ℓ with n_{[s,t]}(a) ≥ 2.
    //
    // Implemented using the full episode window (s = t_ℓ), which maximises
    // statistical power.  C₁ = 4 ("sufficiently large", per the paper).
    // -----------------------------------------------------------------------
    void evict_arms() {
        // Best empirical mean among GOOD arms in the full episode window.
        double best_mu = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < K_; ++a) {
            if (is_good_[a] && n_arm_[a] > 0) {
                const double mu = sum_arm_[a] /
                                  static_cast<double>(n_arm_[a]);
                best_mu = std::max(best_mu, mu);
            }
        }
        if (best_mu == -std::numeric_limits<double>::infinity()) return;

        // Line 23: BAD_{t+1} ← BAD_t ∪ {a ∈ GOOD_t : condition (1) holds}.
        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a] || n_arm_[a] < 2) continue;
            const double mu_a = sum_arm_[a] /
                                static_cast<double>(n_arm_[a]);
            const double thresh =
                std::sqrt(4.0 * log_T_ / static_cast<double>(n_arm_[a]));
            if (best_mu - mu_a > thresh) {
                is_good_[a]       = false;
                // Line 24: condition (2) – record μ̃_ℓ(a) and Δ̃_ℓ(a).
                mu_tilde_[a]     = mu_a;
                delta_tilde_[a]  = std::max(0.0, best_mu - mu_a);
                // Line 24: S_{t+1}(a) ← {}.
                S_[a].clear();
            }
        }
        // Line 25: GOOD_{t+1} = {1,…,K} \ BAD_{t+1}  (maintained by is_good_).
    }
};
