#pragma once

// ADSWITCH – Basic implementation
// Reference: Auer & Chiang, "Adaptively Tracking the Best Bandit Arm with an
// Unknown Number of Distribution Changes", COLT 2019.
// https://proceedings.mlr.press/v99/auer19a/auer19a.pdf
//
// Summary of the algorithm:
//   The algorithm runs in epochs.  At the start of each epoch every arm's
//   statistics are reset and each arm is pulled once for initialisation.
//   Subsequent pulls within the epoch use the UCB index.  After every pull the
//   change-detection test (CDTest) is run on the history of the selected arm.
//   If any split of that history yields a mean difference that exceeds the
//   confidence threshold, the epoch ends and a new one begins.
//
// Change-detection threshold (Hoeffding two-sample bound):
//   For a window split into a left block of size m1 and a right block of size
//   m2, the threshold is
//
//       B(m1, m2) = sqrt( log(2 * K * n / delta) * (1/m1 + 1/m2) / 2 )
//
//   where n = m1 + m2 is the total number of pulls of that arm in the current
//   epoch, K is the number of arms, and delta is the confidence parameter.
//   This follows from Hoeffding's inequality with a union bound over at most
//   K * n split-point comparisons across the epoch.
//
// This basic implementation checks ALL n-1 possible split points, giving
//   O(n) work per pull and O(T^2) worst-case total work.
//
// See adswitch_fast.hpp for the O(T log T) variant described in Remark 3.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

class ADSwitchBasic {
public:
    // -----------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------

    // K       : number of arms
    // alpha   : UCB exploration coefficient  (paper uses alpha >= 1)
    // delta   : confidence parameter for the change-detection threshold
    //           (smaller delta → wider threshold → fewer false alarms)
    // The threshold constant can be adjusted here; the value 1.0 matches the
    // standard Hoeffding / two-sample form.
    explicit ADSwitchBasic(int K, double alpha = 4.0, double delta = 0.01)
        : K_(K), alpha_(alpha), delta_(delta) {
        if (K_ <= 0) throw std::invalid_argument("ADSwitchBasic: K must be > 0");
        if (alpha_ <= 0) throw std::invalid_argument("ADSwitchBasic: alpha must be > 0");
        if (delta_ <= 0 || delta_ >= 1.0)
            throw std::invalid_argument("ADSwitchBasic: delta must be in (0,1)");
        reset_epoch();
    }

    // -----------------------------------------------------------------
    // Public interface
    // -----------------------------------------------------------------

    // Choose an arm to pull at total round t (1-indexed is fine).
    // Must be called before update() each round.
    int select_arm(int t) const {
        // If any arm has not been pulled in this epoch, prioritise it
        // (initialisation pulls).
        for (int a = 0; a < K_; ++a) {
            if (n_[a] == 0) return a;
        }
        // UCB rule: argmax { empirical mean + sqrt(alpha * log(t) / n_a) }
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
        if (arm < 0 || arm >= K_) throw std::out_of_range("ADSwitchBasic: arm out of range");
        // Update statistics for this arm.
        history_[arm].push_back(reward);
        ++n_[arm];
        sum_[arm] += reward;

        // Run change-detection test on the updated arm.
        if (cd_test(arm)) {
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

    // Return the arm currently estimated to be best (highest mean).
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

    // Reset all per-arm statistics to start a fresh epoch.
    void reset_epoch() {
        ++epoch_;
        n_.assign(K_, 0);
        sum_.assign(K_, 0.0);
        history_.assign(K_, {});
    }

    // Change-detection test on arm `arm`.
    // Returns true if a significant distributional shift is detected.
    //
    // For each split point s (1 ≤ s ≤ n-1) the test checks whether the
    // empirical means of the two blocks differ by more than B(s, n-s).
    bool cd_test(int arm) const {
        const auto& h = history_[arm];
        int n = static_cast<int>(h.size());
        if (n < 2) return false;

        // Build prefix sums once.
        std::vector<double> prefix(n + 1, 0.0);
        for (int i = 0; i < n; ++i) prefix[i + 1] = prefix[i] + h[i];
        const double total = prefix[n];

        // Pre-compute the log factor shared across all splits.
        // Standard two-sample Hoeffding bound with a union bound over K * n
        // comparisons (K arms, up to n-1 split points per arm per pull event).
        // P(any false positive in this epoch) ≤ δ.
        const double log_factor =
            std::log(2.0 * static_cast<double>(K_) * static_cast<double>(n) / delta_);

        for (int s = 1; s < n; ++s) {
            const double m1 = static_cast<double>(s);
            const double m2 = static_cast<double>(n - s);

            const double mean1 = prefix[s] / m1;
            const double mean2 = (total - prefix[s]) / m2;

            // Hoeffding-based threshold for Bernoulli rewards in [0,1].
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

    std::vector<int>                  n_;        // per-arm pull counts (current epoch)
    std::vector<double>               sum_;      // per-arm reward sums (current epoch)
    std::vector<std::vector<double>>  history_;  // per-arm reward histories (current epoch)

    int epoch_      = 0;  // epoch counter (incremented at each reset)
    int num_resets_ = 0;  // total number of epoch resets triggered
};
