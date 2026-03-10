#pragma once

// ADSWITCH – Fast implementation with the Remark 3 runtime speedup
// Reference: Auer & Chiang, "Adaptively Tracking the Best Bandit Arm with an
// Unknown Number of Distribution Changes", COLT 2019.
// https://proceedings.mlr.press/v99/auer19a/auer19a.pdf
//
// Remark 3 (runtime speedup):
//   The basic algorithm checks all O(n) split points of the observation
//   history, leading to O(T^2) total work.  Remark 3 observes that it
//   suffices to check only O(log n) strategically chosen split points while
//   retaining the same regret guarantees up to logarithmic factors.
//
//   The key insight is that any change at position c in the history can be
//   detected by comparing the block [c, 2c] against [1, c), because the
//   split point s = 2^floor(log2(c)) satisfies s <= c < 2s.  Therefore
//   checking splits only at the set of powers of two
//
//       S(n) = { 2^k : 0 <= k, 2^k < n }
//
//   covers every possible change location within a factor of 2, which is
//   sufficient for detection guarantees.
//
//   Maintaining a prefix-sum array allows each of the O(log n) comparisons
//   to execute in O(1) time, so the total work per pull is O(log n) and the
//   total work over T rounds is O(T log T).
//
//   Threshold adjustment:
//   Because only O(log n) split points are checked (instead of O(n)), the
//   union-bound argument inside the log uses log2(n)+1 comparisons instead
//   of n, yielding a tighter (smaller) threshold that is still valid.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

class ADSwitchFast {
public:
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    // K       : number of arms
    // alpha   : UCB exploration coefficient
    // delta   : confidence parameter for the change-detection threshold
    explicit ADSwitchFast(int K, double alpha = 4.0, double delta = 0.01)
        : K_(K), alpha_(alpha), delta_(delta) {
        if (K_ <= 0) throw std::invalid_argument("ADSwitchFast: K must be > 0");
        if (alpha_ <= 0) throw std::invalid_argument("ADSwitchFast: alpha must be > 0");
        if (delta_ <= 0 || delta_ >= 1.0)
            throw std::invalid_argument("ADSwitchFast: delta must be in (0,1)");
        reset_epoch();
    }

    // -----------------------------------------------------------------
    // Public interface
    // -----------------------------------------------------------------

    // Choose an arm to pull at total round t (1-indexed).
    int select_arm(int t) const {
        // Prioritise arms not yet pulled in this epoch (initialisation).
        for (int a = 0; a < K_; ++a) {
            if (n_[a] == 0) return a;
        }
        // UCB rule.
        int best = 0;
        double best_score = -std::numeric_limits<double>::infinity();
        for (int a = 0; a < K_; ++a) {
            double score = mean(a) + std::sqrt(alpha_ * std::log(static_cast<double>(t)) /
                                               static_cast<double>(n_[a]));
            if (score > best_score) {
                best_score = score;
                best = a;
            }
        }
        return best;
    }

    // Record the reward for the arm that was just pulled.
    // Returns true if a change was detected (epoch has been reset).
    bool update(int arm, double reward) {
        if (arm < 0 || arm >= K_) throw std::out_of_range("ADSwitchFast: arm out of range");

        // Update running statistics.
        prefix_[arm].push_back(prefix_[arm].back() + reward);
        ++n_[arm];
        sum_[arm] += reward;

        // Run the fast change-detection test.
        if (cd_test_fast(arm)) {
            reset_epoch();
            ++num_resets_;
            return true;
        }
        return false;
    }

    // -----------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------

    int num_arms()   const { return K_; }
    int num_resets() const { return num_resets_; }
    int epoch()      const { return epoch_; }

    double mean(int arm) const {
        if (n_[arm] == 0) return 0.0;
        return sum_[arm] / static_cast<double>(n_[arm]);
    }

    int pulls(int arm) const { return n_[arm]; }

    int best_arm() const {
        int best = 0;
        for (int a = 1; a < K_; ++a) {
            if (mean(a) > mean(best)) best = a;
        }
        return best;
    }

private:
    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    void reset_epoch() {
        ++epoch_;
        n_.assign(K_, 0);
        sum_.assign(K_, 0.0);
        // Each prefix vector starts with a single sentinel 0.
        prefix_.assign(K_, std::vector<double>(1, 0.0));
    }

    // Fast change-detection test: check only O(log n) split points.
    //
    // For arm `arm` with n current-epoch pulls, the split points checked are:
    //   s = 2^k  for k = 0, 1, ..., floor(log2(n-1))
    //
    // This covers every possible change location because any c satisfies
    //   2^floor(log2(c)) <= c < 2^(floor(log2(c))+1).
    bool cd_test_fast(int arm) const {
        const int n = n_[arm];
        if (n < 2) return false;

        const auto& P = prefix_[arm];  // P[i] = sum of first i rewards (P[0]=0)
        const double total = P[n];

        // Number of split points we will check: floor(log2(n-1)) + 1.
        // We use (log2(n)+1) in the union-bound instead of n to get a tighter
        // threshold that is valid for this reduced test set.
        const double log2n = std::log2(static_cast<double>(n));
        const double cnt = log2n + 1.0;  // number of checks (upper bound)
        const double log_factor =
            std::log(2.0 * static_cast<double>(K_) * cnt / delta_);

        // Iterate over split points s = 2^k while s < n.
        for (int k = 0; (1 << k) < n; ++k) {
            const int s = 1 << k;
            const double m1 = static_cast<double>(s);
            const double m2 = static_cast<double>(n - s);

            const double mean1 = P[s] / m1;
            const double mean2 = (total - P[s]) / m2;

            const double threshold = std::sqrt(log_factor * (1.0 / m1 + 1.0 / m2) / 2.0);

            if (std::abs(mean1 - mean2) > threshold) {
                return true;
            }
        }
        return false;
    }

    // -----------------------------------------------------------------
    // Member variables
    // -----------------------------------------------------------------
    int    K_;
    double alpha_;
    double delta_;

    std::vector<int>                  n_;       // per-arm pull counts (current epoch)
    std::vector<double>               sum_;     // per-arm reward sums (current epoch)
    // prefix_[a][i] = sum of first i rewards for arm a in this epoch (prefix_[a][0]=0).
    std::vector<std::vector<double>>  prefix_;

    int epoch_      = 0;
    int num_resets_ = 0;
};
