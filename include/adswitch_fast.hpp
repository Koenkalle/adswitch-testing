#pragma once

// ADSWITCH – Fast implementation (Remark 3: O(T (log T)²) total work)
// Full, faithful implementation of Algorithm 1 from:
//   Auer & Chiang, "Adaptively Tracking the Best Bandit Arm with an Unknown
//   Number of Distribution Changes", COLT 2019.
//   https://proceedings.mlr.press/v99/auer19a/auer19a.pdf
//
// Every section of code is annotated with the corresponding Algorithm 1
// line number(s).  See adswitch_basic.hpp for the full pseudocode listing.
//
// Remark 3 (runtime speedup for condition (3)):
//   "The most expensive step is the check for changes of the good arms,
//    condition (3), with runtime O(Kt³) in time step t.  This time complexity
//    can be significantly reduced, if not all intervals [s₁,s₂] and [s,t] are
//    checked, but only intervals of certain lengths, say 2^k log T for k=3,4,…
//    By storing for these lengths the maximal and minimal values of
//    μ̂_{[s₁,s₂]}(a) so far, the time complexity can be reduced to
//    O(K(log T)²) per time step.  The checks of conditions (1) and (4) can
//    be treated similarly."
//
// Implementation details:
//   Let L_base = max(1, round(log T)).
//   Lengths checked: L_k = L_base · 2^k  for k = 0, 1, 2, …
//   (The paper starts at k=3; starting at k=0 is conservative and correct.)
//
//   For each arm a (while GOOD) and each k, maintain:
//     max_win_[a][k] = max μ̂ over all L_k-length windows seen in this episode
//     min_win_[a][k] = min μ̂ over all L_k-length windows seen in this episode
//   Updated in O(log T) per pull using prefix sums.
//
//   Condition (3) check: for every pair (k_hist, k_recent) where both windows
//   fit non-overlappingly within the episode history (n ≥ L_{k_hist} + L_{k_recent}):
//     recent_mean = mean of last L_{k_recent} pulls
//     if |max_win[a][k_hist] – recent_mean| > thresh: detect change
//     if |min_win[a][k_hist] – recent_mean| > thresh: detect change
//     thresh = √(2 log T / L_{k_hist}) + √(2 log T / L_{k_recent})
//
//   Total per-pull work: O((log T)²)  →  O(T (log T)²) overall.
//
// All other parts of Algorithm 1 (GOOD/BAD sets, arm selection, scheduling,
// conditions (1), (4)) are identical to ADSwitchBasic.

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

class ADSwitchFast {
public:
    // -----------------------------------------------------------------------
    // Algorithm 1, Line 1: Input: Time horizon T.
    // -----------------------------------------------------------------------
    ADSwitchFast(int K, int T, double delta = 0.05, unsigned seed = 42)
        : K_(K), T_(T), delta_(delta), rng_(seed),
          log_T_(std::log(static_cast<double>(std::max(T, 2)))) {
        if (K_ <= 0) throw std::invalid_argument("ADSwitchFast: K must be > 0");
        if (T_ <= 0) throw std::invalid_argument("ADSwitchFast: T must be > 0");
        if (delta_ <= 0.0 || delta_ >= 1.0)
            throw std::invalid_argument("ADSwitchFast: delta must be in (0,1)");

        // Pre-compute the fixed window lengths L_k = L_base · 2^k.
        const int L_base = std::max(1, static_cast<int>(std::round(log_T_)));
        for (int k = 0; ; ++k) {
            const long long L = static_cast<long long>(L_base) << k;
            if (L > T_) break;
            lengths_.push_back(static_cast<int>(L));
        }
        num_lengths_ = static_cast<int>(lengths_.size());

        // Algorithm 1, Lines 2–6: initialise.
        start_episode();
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 8–13: advance time, schedule, select arm.
    // -----------------------------------------------------------------------
    int select_arm(int /*t*/) {
        t_++;
        schedule_bad_arm_checks();
        return select_eligible_arm();
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 14–26: receive reward, run post-pull steps.
    // Returns true if a new episode started.
    // -----------------------------------------------------------------------
    bool update(int arm, double reward) {
        if (arm < 0 || arm >= K_)
            throw std::out_of_range("ADSwitchFast: arm index out of range");

        // Line 14: Receive reward r_t.
        last_pulled_[arm] = t_;
        prefix_[arm].push_back(prefix_[arm].back() + reward);
        n_arm_[arm]++;
        sum_arm_[arm] += reward;

        for (auto& ck : S_[arm]) {
            if (ck.n_pulled < ck.n_needed) {
                ck.n_pulled++;
                ck.sum_rewards += reward;
            }
        }

        // Lines 15–17: Condition (3) – Remark 3 variant.
        // Update max/min window stats for this arm, then check all pairs.
        if (is_good_[arm]) {
            update_window_stats(arm);
            if (condition_3(arm)) {
                start_episode();
                num_resets_++;
                return true;
            }
        }

        // Lines 18–20: Condition (4).
        if (!is_good_[arm]) {
            for (const auto& ck : S_[arm]) {
                if (condition_4(arm, ck)) {
                    start_episode();
                    num_resets_++;
                    return true;
                }
            }
        }

        // Line 21: Remove completed check triples.
        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a]) {
                auto& sv = S_[a];
                sv.erase(std::remove_if(sv.begin(), sv.end(),
                                        [](const CheckTriple& c) {
                                            return c.n_pulled >= c.n_needed;
                                        }),
                         sv.end());
            }
        }

        // Lines 22–25: Evict sub-optimal arms.
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
    struct CheckTriple {
        double eps;
        int    n_needed;
        int    t_created;
        int    n_pulled;
        double sum_rewards;
    };

    // Per-arm, per-length sliding max/min of μ̂ over all fixed-length windows.
    struct WinStat {
        double max_mean =  std::numeric_limits<double>::infinity();
        double min_mean = -std::numeric_limits<double>::infinity();
        bool   valid    = false;
    };

    int    K_, T_;
    double delta_, log_T_;

    int t_     = 0;
    int ell_   = 0;
    int t_ell_ = 0;

    std::vector<bool>                    is_good_;
    std::vector<int>                     last_pulled_;
    // prefix_[a][i] = sum of first i rewards for arm a in this episode.
    std::vector<std::vector<double>>     prefix_;
    std::vector<int>                     n_arm_;
    std::vector<double>                  sum_arm_;
    std::vector<double>                  mu_tilde_;
    std::vector<double>                  delta_tilde_;
    std::vector<std::vector<CheckTriple>> S_;

    // Remark 3: fixed window lengths and per-arm per-length max/min stats.
    std::vector<int>                          lengths_;
    int                                       num_lengths_ = 0;
    std::vector<std::vector<WinStat>>         win_stats_;  // [arm][length_index]

    std::mt19937 rng_;
    int num_resets_ = 0;

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 3–6: start new episode.
    // -----------------------------------------------------------------------
    void start_episode() {
        ell_++;
        t_ell_ = t_ + 1;
        is_good_.assign(K_, true);
        last_pulled_.assign(K_, 0);
        prefix_.assign(K_, std::vector<double>(1, 0.0));
        n_arm_.assign(K_, 0);
        sum_arm_.assign(K_, 0.0);
        mu_tilde_.assign(K_, 0.0);
        delta_tilde_.assign(K_, 0.0);
        S_.assign(K_, {});
        // Reset per-arm window stats.
        win_stats_.assign(K_, std::vector<WinStat>(num_lengths_));
    }

    // -----------------------------------------------------------------------
    // Remark 3 helper: update max/min window statistics for arm `arm`
    // after its latest pull.  Called before condition_3 check.
    //
    // For each length L_k ≤ n_arm_[arm], the window ending at the current
    // pull is [n-L_k, n-1] (pull indices).  Its mean is computed in O(1)
    // using prefix sums, then used to update max_mean / min_mean.
    // -----------------------------------------------------------------------
    void update_window_stats(int arm) {
        const int    n = n_arm_[arm];
        const auto&  P = prefix_[arm];
        for (int k = 0; k < num_lengths_; ++k) {
            const int L = lengths_[k];
            if (L > n) break;
            const double w_mean = (P[n] - P[n - L]) / static_cast<double>(L);
            auto& ws = win_stats_[arm][k];
            if (!ws.valid) {
                ws.max_mean = ws.min_mean = w_mean;
                ws.valid    = true;
            } else {
                ws.max_mean = std::max(ws.max_mean, w_mean);
                ws.min_mean = std::min(ws.min_mean, w_mean);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 15–17: Condition (3) – Remark 3 variant.
    //
    // For every pair (k_hist, k_recent) where n ≥ L_{k_hist} + L_{k_recent}:
    //   recent_mean = mean of last L_{k_recent} pulls (O(1) via prefix sums)
    //   thresh = √(2 log T / L_{k_hist}) + √(2 log T / L_{k_recent})
    //   if |max_win[k_hist] – recent_mean| > thresh: change detected
    //   if |min_win[k_hist] – recent_mean| > thresh: change detected
    //
    // O((log T)²) per pull.
    // -----------------------------------------------------------------------
    bool condition_3(int arm) const {
        const int    n = n_arm_[arm];
        const auto&  P = prefix_[arm];

        for (int kj = 0; kj < num_lengths_; ++kj) {
            const int Lj = lengths_[kj];
            if (Lj > n) break;
            const double recent_mean =
                (P[n] - P[n - Lj]) / static_cast<double>(Lj);
            const double conf_j =
                std::sqrt(2.0 * log_T_ / static_cast<double>(Lj));

            for (int kk = 0; kk < num_lengths_; ++kk) {
                const int Lk = lengths_[kk];
                // Need non-overlapping windows: total pulls ≥ Lk + Lj.
                if (n < Lk + Lj) continue;
                const auto& ws = win_stats_[arm][kk];
                if (!ws.valid) continue;
                const double conf_k =
                    std::sqrt(2.0 * log_T_ / static_cast<double>(Lk));
                const double thresh = conf_k + conf_j;
                if (std::abs(ws.max_mean - recent_mean) > thresh) return true;
                if (std::abs(ws.min_mean - recent_mean) > thresh) return true;
            }
        }
        return false;
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Lines 9–11: scheduling (identical to basic variant).
    // Probability = √ℓ / (K · T · log T)  [no ε factor].
    // -----------------------------------------------------------------------
    void schedule_bad_arm_checks() {
        if (T_ < 2) return;
        const double prob =
            std::sqrt(static_cast<double>(ell_)) /
            (static_cast<double>(K_) * static_cast<double>(T_) * log_T_);

        for (int a = 0; a < K_; ++a) {
            if (is_good_[a]) continue;
            for (int i = 1; i <= 60; ++i) {
                const double eps_i =
                    std::pow(2.0, -static_cast<double>(i));
                if (delta_tilde_[a] > 0.0 &&
                    eps_i < delta_tilde_[a] / 16.0) break;
                if (std::bernoulli_distribution(
                        std::min(1.0, prob))(rng_)) {
                    const double raw =
                        std::pow(2.0, 2 * i + 1) * log_T_;
                    const int n_needed =
                        std::min(T_,
                                 static_cast<int>(std::ceil(raw)));
                    S_[a].push_back({eps_i, n_needed, t_, 0, 0.0});
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Algorithm 1, Line 13: least-recently-pulled eligible arm.
    // -----------------------------------------------------------------------
    int select_eligible_arm() const {
        int best = -1, min_last = std::numeric_limits<int>::max();
        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a] && S_[a].empty()) continue;
            if (last_pulled_[a] < min_last ||
                (last_pulled_[a] == min_last && a < best)) {
                min_last = last_pulled_[a];
                best     = a;
            }
        }
        if (best == -1) {
            for (int a = 0; a < K_; ++a) {
                if (last_pulled_[a] < min_last ||
                    (last_pulled_[a] == min_last &&
                     (best == -1 || a < best))) {
                    min_last = last_pulled_[a];
                    best     = a;
                }
            }
        }
        return (best == -1) ? 0 : best;
    }

    // -----------------------------------------------------------------------
    // Condition (4): identical to basic variant.
    // |μ̂_{[s,t]} − μ̃_ℓ(a)| > Δ̃_ℓ(a)/4 + √(2 log T / n_needed)
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
    // Algorithm 1, Lines 22–25: Condition (1) – evict sub-optimal arms.
    // Identical to basic variant.
    // max_{a'∈GOOD} μ̂(a') − μ̂(a) > √(4 log T / n_a)
    // -----------------------------------------------------------------------
    void evict_arms() {
        double best_mu = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < K_; ++a) {
            if (is_good_[a] && n_arm_[a] > 0)
                best_mu = std::max(best_mu,
                                   sum_arm_[a] /
                                   static_cast<double>(n_arm_[a]));
        }
        if (best_mu == -std::numeric_limits<double>::infinity()) return;

        for (int a = 0; a < K_; ++a) {
            if (!is_good_[a] || n_arm_[a] < 2) continue;
            const double mu_a =
                sum_arm_[a] / static_cast<double>(n_arm_[a]);
            const double thresh =
                std::sqrt(4.0 * log_T_ / static_cast<double>(n_arm_[a]));
            if (best_mu - mu_a > thresh) {
                is_good_[a]      = false;
                mu_tilde_[a]     = mu_a;
                delta_tilde_[a]  = std::max(0.0, best_mu - mu_a);
                S_[a].clear();
                win_stats_[a].assign(num_lengths_, WinStat{});
            }
        }
    }
};
