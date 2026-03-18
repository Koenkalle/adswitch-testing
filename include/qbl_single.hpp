#pragma once

// QBL – single-choice variant (m = 1)
// Based on Algorithm 6 (QBL.M) specialized to selecting a single arm.
//
// API matches the experiment framework used in this workspace:
//   - select_arm(t) -> chosen arm index
//   - update(arm, reward) -> true iff the current leader is demoted
//   - num_resets() -> number of demotions so far

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "../updatable_queue.h"

class QBLSingle {
public:
    QBLSingle(int K, double gamma = 0.1, unsigned seed = 42)
        : K_(K), gamma_(gamma), rng_(seed),
          noise_dist_(1.0 - gamma, 1.0 + gamma),
          priorities_(K), rewards_(K, 1.0), counts_(K, 1), active_(K, false),
          total_reward_(static_cast<double>(K)), total_count_(K) {
        if (K_ <= 0) {
            throw std::invalid_argument("QBLSingle: K must be > 0");
        }
        if (gamma_ < 0.0 || gamma_ > 1.0) {
            throw std::invalid_argument("QBLSingle: gamma must be in [0,1]");
        }

        std::iota(priorities_.begin(), priorities_.end(), 0);
        std::shuffle(priorities_.begin(), priorities_.end(), rng_);
        for (int arm = 0; arm < K_; ++arm) {
            queue_.push(arm, priorities_[arm]);
        }
    }

    int select_arm(int /*t*/) {
        if (queue_.empty()) {
            throw std::runtime_error("QBLSingle: internal queue is empty");
        }
        return queue_.top().key;
    }

    bool update(int arm, double reward) {
        if (arm < 0 || arm >= K_) {
            throw std::out_of_range("QBLSingle: arm index out of range");
        }
        if (queue_.empty()) {
            throw std::runtime_error("QBLSingle: internal queue is empty");
        }

        const int leader = queue_.top().key;
        if (leader != arm) {
            throw std::invalid_argument("QBLSingle: update arm must match current leader");
        }

        if (!active_[arm]) {
            const double local_mean = rewards_[arm] / static_cast<double>(counts_[arm]);
            total_reward_ = total_reward_ - rewards_[arm] + local_mean;
            total_count_ = total_count_ - counts_[arm] + 1;
            rewards_[arm] = local_mean;
            counts_[arm] = 1;
            active_[arm] = true;
        }

        total_reward_ += reward;
        total_count_ += 1;
        rewards_[arm] += reward;
        counts_[arm] += 1;

        const double global_mean = total_reward_ / static_cast<double>(total_count_);
        const double local_mean = rewards_[arm] / static_cast<double>(counts_[arm]);
        const double multiplier = (gamma_ == 0.0) ? 1.0 : noise_dist_(rng_);

        if (global_mean >= local_mean * multiplier) {
            const int top_priority = queue_.top().priority;
            const int new_priority = std::min(priorities_[arm] - 1,
                                              top_priority - 1 + counts_[arm] - K_);
            priorities_[arm] = new_priority;
            queue_.update(arm, new_priority);
            active_[arm] = false;
            num_demotions_++;
            return true;
        }
        return false;
    }

    int num_arms() const { return K_; }
    int num_resets() const { return num_demotions_; }
    int num_demotions() const { return num_demotions_; }
    int priority(int arm) const { return priorities_.at(arm); }
    int count(int arm) const { return counts_.at(arm); }
    double reward_sum(int arm) const { return rewards_.at(arm); }

private:
    int K_;
    double gamma_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> noise_dist_;
    better_priority_queue::updatable_priority_queue<int, int> queue_;

    std::vector<int> priorities_;
    std::vector<double> rewards_;
    std::vector<int> counts_;
    std::vector<bool> active_;

    double total_reward_;
    int total_count_;
    int num_demotions_ = 0;
};
