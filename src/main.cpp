// ADSWITCH experiment runner
// Exercises ADSwitchBasic and ADSwitchFast on five environment types and
// reports cumulative regret, number of epoch resets, and runtime.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "adswitch_basic.hpp"
#include "adswitch_fast.hpp"
#include "environments.hpp"

// ---------------------------------------------------------------------------
// Run a single algorithm on a single environment for T rounds.
// Returns {cumulative_regret, num_resets, elapsed_ms}.
// ---------------------------------------------------------------------------
template <typename Algorithm>
struct RunResult {
    double cumulative_regret;
    int    num_resets;
    double elapsed_ms;
};

template <typename Algorithm>
RunResult<Algorithm> run_experiment(Algorithm& algo, BanditEnvironment& env, int T) {
    double cum_regret = 0.0;

    auto t_start = std::chrono::steady_clock::now();

    for (int t = 1; t <= T; ++t) {
        // 1. Algorithm selects an arm.
        int arm = algo.select_arm(t);

        // 2. Environment reveals reward.
        double reward = env.get_reward(arm);

        // 3. Compute instantaneous regret using the *current* best arm's mean.
        std::vector<double> cur_means = env.means();
        double best_mean = *std::max_element(cur_means.begin(), cur_means.end());
        cum_regret += best_mean - cur_means[arm];

        // 4. Update algorithm with observed reward.
        algo.update(arm, reward);

        // 5. Advance the environment (non-stationary update).
        env.advance();
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return {cum_regret, algo.num_resets(), elapsed_ms};
}

// ---------------------------------------------------------------------------
// Run both algorithms on the given environment.
// ---------------------------------------------------------------------------
static void compare(const std::string& env_name,
                    const std::function<std::unique_ptr<BanditEnvironment>()>& make_env,
                    int T, int K) {
    std::cout << "\n=== " << env_name << " ===\n";
    std::cout << std::left  << std::setw(12) << "Algorithm"
              << std::right
              << std::setw(10) << "Regret"
              << std::setw(10) << "Resets"
              << std::setw(14) << "Time\n";
    std::cout << std::string(48, '-') << "\n";

    auto print_row = [](const std::string& algo_name,
                        double regret, int resets, double elapsed_ms) {
        std::cout << std::left  << std::setw(12) << algo_name
                  << std::right
                  << std::setw(10) << std::fixed << std::setprecision(1) << regret
                  << std::setw(10) << resets
                  << std::setw(12) << std::fixed << std::setprecision(2) << elapsed_ms
                  << " ms\n";
    };

    {
        auto env  = make_env();
        ADSwitchBasic algo(K, /*alpha=*/4.0, /*delta=*/0.05);
        auto res = run_experiment(algo, *env, T);
        print_row("Basic", res.cumulative_regret, res.num_resets, res.elapsed_ms);
    }
    {
        auto env  = make_env();
        ADSwitchFast  algo(K, /*alpha=*/4.0, /*delta=*/0.05);
        auto res = run_experiment(algo, *env, T);
        print_row("Fast (R3)", res.cumulative_regret, res.num_resets, res.elapsed_ms);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    constexpr int T = 10000;
    constexpr int K = 5;

    std::cout << "ADSWITCH Experiment Runner\n";
    std::cout << "Arms K=" << K << "  Horizon T=" << T << "\n";

    // ------------------------------------------------------------------
    // 1. Stochastic (stationary) environment
    // ------------------------------------------------------------------
    compare("Stochastic",
            []() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<StochasticEnv>(
                    std::vector<double>{0.5, 0.6, 0.7, 0.4, 0.55}, /*seed=*/1);
            },
            T, K);

    // ------------------------------------------------------------------
    // 2. Drifting stochastic environment
    // ------------------------------------------------------------------
    compare("Drifting",
            []() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<DriftingEnv>(
                    std::vector<double>{0.5, 0.6, 0.7, 0.4, 0.55},
                    /*drift_rate=*/0.002, /*seed=*/2);
            },
            T, K);

    // ------------------------------------------------------------------
    // 3. Sharp-switch environment (manually specified phases)
    // ------------------------------------------------------------------
    compare("Sharp-switch",
            []() -> std::unique_ptr<BanditEnvironment> {
                using Phase = std::pair<int, std::vector<double>>;
                return std::make_unique<SharpSwitchEnv>(
                    std::vector<Phase>{
                        {0,    {0.3, 0.9, 0.3, 0.3, 0.3}},
                        {2500, {0.3, 0.3, 0.9, 0.3, 0.3}},
                        {5000, {0.9, 0.3, 0.3, 0.3, 0.3}},
                        {7500, {0.3, 0.3, 0.3, 0.3, 0.9}}},
                    /*seed=*/3);
            },
            T, K);

    // ------------------------------------------------------------------
    // 4. Big-switch environment (large gap, several switches)
    // ------------------------------------------------------------------
    compare("Big-switch",
            [T, K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<BigSwitchEnv>(K, T, /*num_switches=*/5,
                                                      /*gap=*/0.4, /*seed=*/4);
            },
            T, K);

    // ------------------------------------------------------------------
    // 5. Small-switch environment (small gap, several switches)
    // ------------------------------------------------------------------
    compare("Small-switch",
            [T, K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<SmallSwitchEnv>(K, T, /*num_switches=*/5,
                                                        /*gap=*/0.1, /*seed=*/5);
            },
            T, K);

    std::cout << "\nDone.\n";
    return 0;
}
