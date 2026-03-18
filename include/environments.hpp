#pragma once

// Test environments for the ADSWITCH bandit algorithm.
// Implements stochastic, drifting, sharp-switch, big-switch, small-switch,
// and Mod2 environments.

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Abstract base class
// ---------------------------------------------------------------------------

class BanditEnvironment {
public:
    virtual ~BanditEnvironment() = default;

    // Draw a reward in [0,1] for the chosen arm at the current time step.
    virtual double get_reward(int arm) = 0;

    // Advance internal time by one step (update drifting/switching state).
    virtual void advance() = 0;

    // Return the arm with the highest current expected reward.
    virtual int best_arm() const = 0;

    // Number of arms.
    virtual int num_arms() const = 0;

    // Current expected reward for every arm.
    virtual std::vector<double> means() const = 0;

    // Current time step (number of advance() calls).
    virtual int time() const = 0;

    // Rounds at which an abrupt environment shift occurs.
    virtual std::vector<int> shift_rounds() const { return {}; }

    // Human-readable description of this environment.
    virtual std::string description() const = 0;
};

// ---------------------------------------------------------------------------
// Helper: pick the arm with the highest mean
// ---------------------------------------------------------------------------
static inline int argmax_means(const std::vector<double>& m) {
    return static_cast<int>(std::max_element(m.begin(), m.end()) - m.begin());
}

// ---------------------------------------------------------------------------
// 1. Stochastic environment
//    Reward distributions are Bernoulli with fixed means throughout the run.
// ---------------------------------------------------------------------------
class StochasticEnv : public BanditEnvironment {
public:
    explicit StochasticEnv(std::vector<double> means, unsigned seed = 42)
        : means_(std::move(means)), t_(0), rng_(seed) {
        if (means_.empty()) {
            throw std::invalid_argument("StochasticEnv: must have at least one arm");
        }
    }

    double get_reward(int arm) override {
        std::bernoulli_distribution dist(means_.at(arm));
        return dist(rng_) ? 1.0 : 0.0;
    }

    void advance() override { ++t_; }

    int best_arm() const override { return argmax_means(means_); }

    int num_arms() const override { return static_cast<int>(means_.size()); }

    std::vector<double> means() const override { return means_; }

    int time() const override { return t_; }

    std::string description() const override { return "Stochastic (fixed distributions)"; }

private:
    std::vector<double> means_;
    int t_;
    std::mt19937 rng_;
};

// ---------------------------------------------------------------------------
// 2. Drifting stochastic environment
//    Each arm's Bernoulli mean performs an independent Gaussian random walk,
//    clipped to [eps, 1-eps] to keep rewards well-defined.
// ---------------------------------------------------------------------------
class DriftingEnv : public BanditEnvironment {
public:
    DriftingEnv(std::vector<double> initial_means, double drift_rate = 0.002,
                unsigned seed = 42)
        : means_(std::move(initial_means)), t_(0), drift_rate_(drift_rate), rng_(seed) {
        if (means_.empty()) {
            throw std::invalid_argument("DriftingEnv: must have at least one arm");
        }
    }

    double get_reward(int arm) override {
        double p = std::max(0.01, std::min(0.99, means_.at(arm)));
        return std::bernoulli_distribution(p)(rng_) ? 1.0 : 0.0;
    }

    // Each step the means drift by a small Gaussian perturbation.
    void advance() override {
        ++t_;
        std::normal_distribution<double> noise(0.0, drift_rate_);
        for (auto& m : means_) {
            m += noise(rng_);
            m = std::max(0.01, std::min(0.99, m));
        }
    }

    int best_arm() const override { return argmax_means(means_); }

    int num_arms() const override { return static_cast<int>(means_.size()); }

    std::vector<double> means() const override { return means_; }

    int time() const override { return t_; }

    std::string description() const override {
        return "Drifting stochastic (Gaussian random walk, rate=" +
               std::to_string(drift_rate_) + ")";
    }

private:
    std::vector<double> means_;
    int t_;
    double drift_rate_;
    std::mt19937 rng_;
};

// ---------------------------------------------------------------------------
// Internal helper: a piecewise-stationary environment defined by phases.
// Each phase specifies a start time and the arm means for that phase.
// ---------------------------------------------------------------------------
class PiecewiseStationaryEnv : public BanditEnvironment {
public:
    // phases[i] = {start_round, arm_means}
    // phases must be sorted by start_round; first phase must start at 0.
    PiecewiseStationaryEnv(std::vector<std::pair<int, std::vector<double>>> phases,
                           unsigned seed = 42)
        : phases_(std::move(phases)), t_(0), phase_idx_(0), rng_(seed) {
        if (phases_.empty()) {
            throw std::invalid_argument("PiecewiseStationaryEnv: need at least one phase");
        }
        if (phases_[0].first != 0) {
            throw std::invalid_argument("PiecewiseStationaryEnv: first phase must start at t=0");
        }
        means_ = phases_[0].second;
    }

    double get_reward(int arm) override {
        double p = std::max(0.0, std::min(1.0, means_.at(arm)));
        return std::bernoulli_distribution(p)(rng_) ? 1.0 : 0.0;
    }

    void advance() override {
        ++t_;
        while (phase_idx_ + 1 < static_cast<int>(phases_.size()) &&
               t_ >= phases_[phase_idx_ + 1].first) {
            ++phase_idx_;
            means_ = phases_[phase_idx_].second;
        }
    }

    int best_arm() const override { return argmax_means(means_); }

    int num_arms() const override {
        return static_cast<int>(phases_[0].second.size());
    }

    std::vector<double> means() const override { return means_; }

    int time() const override { return t_; }

    int current_phase() const { return phase_idx_; }

    int num_phases() const { return static_cast<int>(phases_.size()); }

    std::vector<int> shift_rounds() const override {
        std::vector<int> shifts;
        shifts.reserve(phases_.size() > 0 ? phases_.size() - 1 : 0);
        for (std::size_t i = 1; i < phases_.size(); ++i) {
            shifts.push_back(phases_[i].first);
        }
        return shifts;
    }

    std::string description() const override { return "Piecewise stationary"; }

protected:
    std::vector<std::pair<int, std::vector<double>>> phases_;
    std::vector<double> means_;
    int t_;
    int phase_idx_;
    std::mt19937 rng_;
};

// ---------------------------------------------------------------------------
// 3. Sharp-switch environment
//    User-specified phases with abrupt distribution changes.
// ---------------------------------------------------------------------------
class SharpSwitchEnv : public PiecewiseStationaryEnv {
public:
    SharpSwitchEnv(std::vector<std::pair<int, std::vector<double>>> phases,
                   unsigned seed = 42)
        : PiecewiseStationaryEnv(std::move(phases), seed) {}

    std::string description() const override {
        return "Sharp-switch (" + std::to_string(phases_.size() - 1) + " switches)";
    }
};

// ---------------------------------------------------------------------------
// 4. Big-switch environment
//    Randomly generated phases where the best arm changes at each switch by a
//    large margin (default gap = 0.4).
// ---------------------------------------------------------------------------
// 4. & 5. Randomly generated switch environments (shared base class)
//    BigSwitchEnv  : large gap between the best arm and others (default Δ = 0.4)
//    SmallSwitchEnv: small gap, switches harder to detect   (default Δ = 0.1)
// ---------------------------------------------------------------------------
class RandomSwitchEnv : public BanditEnvironment {
public:
    RandomSwitchEnv(int K, int T, int num_switches, double gap,
                    unsigned seed)
        : K_(K), gap_(gap), t_(0), phase_idx_(0), rng_(seed) {
        build_phases(T, num_switches);
        means_ = phases_[0].second;
    }

    double get_reward(int arm) override {
        double p = std::max(0.0, std::min(1.0, means_.at(arm)));
        return std::bernoulli_distribution(p)(rng_) ? 1.0 : 0.0;
    }

    void advance() override {
        ++t_;
        while (phase_idx_ + 1 < static_cast<int>(phases_.size()) &&
               t_ >= phases_[phase_idx_ + 1].first) {
            ++phase_idx_;
            means_ = phases_[phase_idx_].second;
        }
    }

    int best_arm() const override { return argmax_means(means_); }

    int num_arms() const override { return K_; }

    std::vector<double> means() const override { return means_; }

    int time() const override { return t_; }

    std::string description() const override {
        return "RandomSwitch (" + std::to_string(phases_.size() - 1) +
               " switches, gap=" + std::to_string(gap_) + ")";
    }

    const std::vector<std::pair<int, std::vector<double>>>& phases() const {
        return phases_;
    }

    std::vector<int> shift_rounds() const override {
        std::vector<int> shifts;
        shifts.reserve(phases_.size() > 0 ? phases_.size() - 1 : 0);
        for (std::size_t i = 1; i < phases_.size(); ++i) {
            shifts.push_back(phases_[i].first);
        }
        return shifts;
    }

protected:
    int    K_;
    double gap_;

private:
    // Build means for a given best arm: best arm gets 0.5 + gap/2, rest get 0.5 - gap/2.
    std::vector<double> make_means(int best) const {
        std::vector<double> m(K_, 0.5 - gap_ / 2.0);
        m[best] = 0.5 + gap_ / 2.0;
        return m;
    }

    void build_phases(int T, int num_switches) {
        std::uniform_int_distribution<int> arm_dist(0, K_ - 1);

        // Evenly space switch points across [1, T).
        std::vector<int> switch_points;
        if (num_switches > 0) {
            int interval = T / (num_switches + 1);
            for (int i = 1; i <= num_switches; ++i) {
                switch_points.push_back(i * interval);
            }
        }

        int best = arm_dist(rng_);
        phases_.push_back({0, make_means(best)});
        for (int sp : switch_points) {
            // Ensure the best arm actually changes at each switch.
            int new_best;
            do {
                new_best = arm_dist(rng_);
            } while (new_best == best && K_ > 1);
            best = new_best;
            phases_.push_back({sp, make_means(best)});
        }
        means_ = phases_[0].second;
    }

    std::vector<std::pair<int, std::vector<double>>> phases_;
    std::vector<double> means_;
    int t_;
    int phase_idx_;
    std::mt19937 rng_;
};

// ---------------------------------------------------------------------------
// 4. Big-switch environment
//    Randomly generated phases where the best arm changes at each switch by a
//    large margin (default gap = 0.4).
// ---------------------------------------------------------------------------
class BigSwitchEnv : public RandomSwitchEnv {
public:
    BigSwitchEnv(int K, int T, int num_switches, double gap = 0.4,
                 unsigned seed = 42)
        : RandomSwitchEnv(K, T, num_switches, gap, seed) {}

    std::string description() const override {
        return "Big-switch (" + std::to_string(phases().size() - 1) +
               " switches, gap=" + std::to_string(gap_) + ")";
    }
};

// ---------------------------------------------------------------------------
// 5. Small-switch environment
//    Same as BigSwitchEnv but with a small gap (default 0.1) so switches are
//    harder to detect.
// ---------------------------------------------------------------------------
class SmallSwitchEnv : public RandomSwitchEnv {
public:
    SmallSwitchEnv(int K, int T, int num_switches, double gap = 0.1,
                   unsigned seed = 42)
        : RandomSwitchEnv(K, T, num_switches, gap, seed) {}

    std::string description() const override {
        return "Small-switch (" + std::to_string(phases().size() - 1) +
               " switches, gap=" + std::to_string(gap_) + ")";
    }
};

// ---------------------------------------------------------------------------
// 6. Mod2 environment
//    Binary rewards with progressively longer stationary streaks. The initial
//    phase rewards even-indexed arms with mean 1 and odd-indexed arms with
//    mean 0; each shift swaps the parity. Shift intervals grow geometrically
//    as 3, 3^2, 3^3, ... rounds, creating increasingly stable phases.
// ---------------------------------------------------------------------------
class Mod2Env : public PiecewiseStationaryEnv {
public:
    Mod2Env(int K, int T, unsigned seed = 42)
        : PiecewiseStationaryEnv(build_phases(K, T), seed), K_(K), T_(T) {
        if (K_ <= 0) {
            throw std::invalid_argument("Mod2Env: K must be > 0");
        }
        if (T_ <= 0) {
            throw std::invalid_argument("Mod2Env: T must be > 0");
        }
    }

    std::string description() const override {
        return "Mod2 (parity swaps after 3, 3^2, 3^3, ... rounds)";
    }

private:
    static std::vector<double> parity_means(int K, bool even_is_best) {
        std::vector<double> means(K, 0.0);
        for (int arm = 0; arm < K; ++arm) {
            const bool is_even = (arm % 2) == 0;
            means[arm] = (is_even == even_is_best) ? 1.0 : 0.0;
        }
        return means;
    }

    static std::vector<std::pair<int, std::vector<double>>> build_phases(int K, int T) {
        if (K <= 0) {
            throw std::invalid_argument("Mod2Env: K must be > 0");
        }
        if (T <= 0) {
            throw std::invalid_argument("Mod2Env: T must be > 0");
        }

        std::vector<std::pair<int, std::vector<double>>> phases;
        phases.push_back({0, parity_means(K, /*even_is_best=*/true)});

        long long elapsed = 0;
        long long interval = 3;
        bool even_is_best = true;
        while (true) {
            elapsed += interval;
            if (elapsed >= T) break;
            even_is_best = !even_is_best;
            phases.push_back({static_cast<int>(elapsed),
                              parity_means(K, even_is_best)});
            interval *= 3;
        }
        return phases;
    }

    int K_;
    int T_;
};
