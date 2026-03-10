// Unit tests for ADSwitchBasic, ADSwitchFast, and the test environments.
// Build and run with CMake (target: adswitch_tests).
// No external test framework is required; failures are reported via assertions
// that terminate with a non-zero exit code.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "adswitch_basic.hpp"
#include "adswitch_fast.hpp"
#include "environments.hpp"

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond)                                               \
    do {                                                                \
        ++g_tests_run;                                                  \
        if (!(cond)) {                                                  \
            ++g_tests_failed;                                           \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__      \
                      << "  " << #cond << "\n";                         \
        }                                                               \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_NEAR(a, b, eps)                                          \
    do {                                                                \
        ++g_tests_run;                                                  \
        if (std::abs((a) - (b)) > (eps)) {                              \
            ++g_tests_failed;                                           \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__      \
                      << "  |" << #a << " - " << #b << "| = "          \
                      << std::abs((a) - (b)) << " > " << (eps) << "\n";\
        }                                                               \
    } while (0)

#define ASSERT_THROWS(expr, exc_type)                                   \
    do {                                                                \
        ++g_tests_run;                                                  \
        bool threw = false;                                             \
        try { (expr); } catch (const exc_type&) { threw = true; }      \
        catch (...) {}                                                  \
        if (!threw) {                                                   \
            ++g_tests_failed;                                           \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__      \
                      << "  expected " #exc_type " from: " #expr "\n"; \
        }                                                               \
    } while (0)

static void test_section(const std::string& name) {
    std::cout << "\n--- " << name << " ---\n";
}

// Maximum number of spurious epoch resets allowed in a stationary environment.
// With delta=1e-6 and 500 rounds, occasional false alarms are extremely rare;
// this bound accommodates probabilistic effects while still being meaningful.
static constexpr int kMaxSpuriousResets = 3;

// ===========================================================================
// Environment tests
// ===========================================================================

static void test_stochastic_env() {
    test_section("StochasticEnv");

    std::vector<double> means = {0.3, 0.7, 0.5};
    StochasticEnv env(means, /*seed=*/42);

    ASSERT_TRUE(env.num_arms() == 3);
    ASSERT_TRUE(env.best_arm() == 1);  // arm 1 has mean 0.7
    ASSERT_NEAR(env.means()[1], 0.7, 1e-9);

    // Rewards should be in {0, 1}.
    for (int i = 0; i < 100; ++i) {
        double r = env.get_reward(0);
        ASSERT_TRUE(r == 0.0 || r == 1.0);
        env.advance();
    }

    // Advancing should not change means in stationary env.
    ASSERT_NEAR(env.means()[0], 0.3, 1e-9);
    ASSERT_TRUE(env.time() == 100);

    // Bad arm index.
    ASSERT_THROWS(env.get_reward(3), std::out_of_range);

    // Empty means.
    ASSERT_THROWS(StochasticEnv({}), std::invalid_argument);
}

static void test_drifting_env() {
    test_section("DriftingEnv");

    std::vector<double> init = {0.5, 0.5, 0.5};
    DriftingEnv env(init, /*drift_rate=*/0.01, /*seed=*/7);

    ASSERT_TRUE(env.num_arms() == 3);

    // After many advances the means should have changed.
    for (int i = 0; i < 500; ++i) env.advance();
    // With drift_rate=0.01 and 500 steps, the mean is unlikely to stay exactly
    // the same, but remains clipped to [0.01, 0.99].
    auto m = env.means();
    for (double v : m) {
        ASSERT_TRUE(v >= 0.01 && v <= 0.99);
    }
    ASSERT_TRUE(env.time() == 500);
}

static void test_sharp_switch_env() {
    test_section("SharpSwitchEnv");

    using Phase = std::pair<int, std::vector<double>>;
    SharpSwitchEnv env(
        {Phase{0, {0.9, 0.1}}, Phase{100, {0.1, 0.9}}}, /*seed=*/42);

    ASSERT_TRUE(env.num_arms() == 2);
    ASSERT_TRUE(env.best_arm() == 0);

    // Advance to just before the switch.
    for (int t = 0; t < 99; ++t) env.advance();
    ASSERT_TRUE(env.best_arm() == 0);

    // Advance past the switch point.
    env.advance();
    ASSERT_TRUE(env.best_arm() == 1);

    // First phase must start at 0.
    ASSERT_THROWS(SharpSwitchEnv({Phase{1, {0.5}}}), std::invalid_argument);
}

static void test_big_switch_env() {
    test_section("BigSwitchEnv");

    BigSwitchEnv env(4, 1000, 3, /*gap=*/0.4, /*seed=*/99);
    ASSERT_TRUE(env.num_arms() == 4);
    ASSERT_TRUE(static_cast<int>(env.phases().size()) == 4);  // initial + 3 switches

    // All arm means must remain in [0, 1].
    for (int t = 0; t < 1000; ++t) {
        auto m = env.means();
        for (double v : m) ASSERT_TRUE(v >= 0.0 && v <= 1.0);
        env.advance();
    }
}

static void test_small_switch_env() {
    test_section("SmallSwitchEnv");

    SmallSwitchEnv env(3, 600, 2, /*gap=*/0.1, /*seed=*/11);
    ASSERT_TRUE(env.num_arms() == 3);
    ASSERT_TRUE(static_cast<int>(env.phases().size()) == 3);  // initial + 2 switches
}

// ===========================================================================
// ADSwitchBasic tests
// ===========================================================================

static void test_basic_construction() {
    test_section("ADSwitchBasic construction");

    ADSwitchBasic algo(5);
    ASSERT_TRUE(algo.num_arms()   == 5);
    ASSERT_TRUE(algo.num_resets() == 0);
    ASSERT_TRUE(algo.epoch()      == 1);

    // Invalid arguments.
    ASSERT_THROWS(ADSwitchBasic(0), std::invalid_argument);
    ASSERT_THROWS(ADSwitchBasic(3, -1.0), std::invalid_argument);
    ASSERT_THROWS(ADSwitchBasic(3, 1.0, 0.0), std::invalid_argument);
}

static void test_basic_initialisation_pulls() {
    test_section("ADSwitchBasic initialisation pulls");

    // The first K pulls should cycle through all arms (initialisation).
    constexpr int K = 4;
    ADSwitchBasic algo(K, 4.0, 0.01);

    std::vector<int> init_arms;
    for (int t = 1; t <= K; ++t) {
        int arm = algo.select_arm(t);
        init_arms.push_back(arm);
        algo.update(arm, 0.5);
    }
    // Each arm appears exactly once.
    std::sort(init_arms.begin(), init_arms.end());
    for (int a = 0; a < K; ++a) ASSERT_TRUE(init_arms[a] == a);
}

static void test_basic_invalid_update() {
    test_section("ADSwitchBasic invalid update");

    ADSwitchBasic algo(3);
    ASSERT_THROWS(algo.update(-1, 0.5), std::out_of_range);
    ASSERT_THROWS(algo.update(3,  0.5), std::out_of_range);
}

static void test_basic_no_change_stationary() {
    test_section("ADSwitchBasic stationary environment – no spurious resets");

    // Feed a stationary Bernoulli arm many times; false-detection rate should
    // be low with a tight delta.
    constexpr int K = 2;
    constexpr int T = 500;
    ADSwitchBasic algo(K, 4.0, /*delta=*/1e-6);

    StochasticEnv env({0.4, 0.8}, /*seed=*/3);
    int resets = 0;
    for (int t = 1; t <= T; ++t) {
        int arm = algo.select_arm(t);
        double r = env.get_reward(arm);
        bool reset = algo.update(arm, r);
        if (reset) ++resets;
        env.advance();
    }
    // With very tight delta and stationary rewards, resets should be rare.
    ASSERT_TRUE(resets <= kMaxSpuriousResets);
}

static void test_basic_detects_large_change() {
    test_section("ADSwitchBasic detects large distribution change");

    // Feed a sequence that is clearly non-stationary: first half all zeros,
    // second half all ones.  The algorithm must detect the change.
    constexpr int K = 1;
    ADSwitchBasic algo(K, 4.0, 0.05);

    bool detected = false;
    // Pull arm 0 repeatedly with a strong shift.
    for (int t = 1; t <= 30; ++t) {
        int arm = algo.select_arm(t);
        double r = (t <= 15) ? 0.0 : 1.0;
        if (algo.update(arm, r)) {
            detected = true;
            break;
        }
    }
    ASSERT_TRUE(detected);
    ASSERT_TRUE(algo.num_resets() >= 1);
}

static void test_basic_resets_on_sharp_switch() {
    test_section("ADSwitchBasic resets on sharp-switch environment");

    constexpr int K = 3;
    using Phase = std::pair<int, std::vector<double>>;
    SharpSwitchEnv env(
        {Phase{0, {0.9, 0.1, 0.1}}, Phase{300, {0.1, 0.9, 0.1}},
         Phase{600, {0.1, 0.1, 0.9}}},
        /*seed=*/20);

    ADSwitchBasic algo(K, 4.0, 0.05);
    int resets = 0;
    for (int t = 1; t <= 900; ++t) {
        int arm = algo.select_arm(t);
        bool reset = algo.update(arm, env.get_reward(arm));
        if (reset) ++resets;
        env.advance();
    }
    // Should detect at least the two sharp switches.
    ASSERT_TRUE(resets >= 2);
}

static void test_basic_ucb_favours_best_arm() {
    test_section("ADSwitchBasic UCB favours best arm in stationary env");

    constexpr int K = 3;
    // Arm 1 is clearly best.
    StochasticEnv env({0.3, 0.9, 0.2}, /*seed=*/55);
    ADSwitchBasic algo(K, 4.0, 1e-6);

    int counts[K] = {};
    for (int t = 1; t <= 1000; ++t) {
        int arm = algo.select_arm(t);
        ++counts[arm];
        algo.update(arm, env.get_reward(arm));
        env.advance();
    }
    // Arm 1 should be pulled significantly more than the others.
    ASSERT_TRUE(counts[1] > counts[0]);
    ASSERT_TRUE(counts[1] > counts[2]);
}

// ===========================================================================
// ADSwitchFast tests
// ===========================================================================

static void test_fast_construction() {
    test_section("ADSwitchFast construction");

    ADSwitchFast algo(5);
    ASSERT_TRUE(algo.num_arms()   == 5);
    ASSERT_TRUE(algo.num_resets() == 0);
    ASSERT_TRUE(algo.epoch()      == 1);

    ASSERT_THROWS(ADSwitchFast(0), std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, -1.0), std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, 1.0, 0.0), std::invalid_argument);
}

static void test_fast_initialisation_pulls() {
    test_section("ADSwitchFast initialisation pulls");

    constexpr int K = 4;
    ADSwitchFast algo(K);

    std::vector<int> init_arms;
    for (int t = 1; t <= K; ++t) {
        int arm = algo.select_arm(t);
        init_arms.push_back(arm);
        algo.update(arm, 0.5);
    }
    std::sort(init_arms.begin(), init_arms.end());
    for (int a = 0; a < K; ++a) ASSERT_TRUE(init_arms[a] == a);
}

static void test_fast_no_change_stationary() {
    test_section("ADSwitchFast stationary environment – no spurious resets");

    constexpr int K = 2;
    constexpr int T = 500;
    ADSwitchFast algo(K, 4.0, /*delta=*/1e-6);

    StochasticEnv env({0.4, 0.8}, /*seed=*/3);
    int resets = 0;
    for (int t = 1; t <= T; ++t) {
        int arm = algo.select_arm(t);
        double r = env.get_reward(arm);
        if (algo.update(arm, r)) ++resets;
        env.advance();
    }
    ASSERT_TRUE(resets <= kMaxSpuriousResets);
}

static void test_fast_detects_large_change() {
    test_section("ADSwitchFast detects large distribution change");

    constexpr int K = 1;
    ADSwitchFast algo(K, 4.0, 0.05);

    bool detected = false;
    for (int t = 1; t <= 30; ++t) {
        int arm = algo.select_arm(t);
        double r = (t <= 15) ? 0.0 : 1.0;
        if (algo.update(arm, r)) {
            detected = true;
            break;
        }
    }
    ASSERT_TRUE(detected);
    ASSERT_TRUE(algo.num_resets() >= 1);
}

static void test_fast_resets_on_sharp_switch() {
    test_section("ADSwitchFast resets on sharp-switch environment");

    constexpr int K = 3;
    using Phase = std::pair<int, std::vector<double>>;
    SharpSwitchEnv env(
        {Phase{0, {0.9, 0.1, 0.1}}, Phase{300, {0.1, 0.9, 0.1}},
         Phase{600, {0.1, 0.1, 0.9}}},
        /*seed=*/20);

    ADSwitchFast algo(K, 4.0, 0.05);
    int resets = 0;
    for (int t = 1; t <= 900; ++t) {
        int arm = algo.select_arm(t);
        if (algo.update(arm, env.get_reward(arm))) ++resets;
        env.advance();
    }
    ASSERT_TRUE(resets >= 2);
}

static void test_fast_ucb_favours_best_arm() {
    test_section("ADSwitchFast UCB favours best arm in stationary env");

    constexpr int K = 3;
    StochasticEnv env({0.3, 0.9, 0.2}, /*seed=*/55);
    ADSwitchFast algo(K, 4.0, 1e-6);

    int counts[K] = {};
    for (int t = 1; t <= 1000; ++t) {
        int arm = algo.select_arm(t);
        ++counts[arm];
        algo.update(arm, env.get_reward(arm));
        env.advance();
    }
    ASSERT_TRUE(counts[1] > counts[0]);
    ASSERT_TRUE(counts[1] > counts[2]);
}

// ===========================================================================
// Parity tests – Basic vs Fast should behave similarly
// ===========================================================================

static void test_both_regret_comparable() {
    test_section("Basic vs Fast: comparable regret on big-switch env");

    constexpr int K = 4;
    constexpr int T = 2000;

    double regret_basic = 0.0;
    double regret_fast  = 0.0;

    for (int seed_offset = 0; seed_offset < 5; ++seed_offset) {
        // Use the same environment seed for fair comparison.
        BigSwitchEnv env_b(K, T, 3, 0.4, 100 + seed_offset);
        BigSwitchEnv env_f(K, T, 3, 0.4, 100 + seed_offset);

        ADSwitchBasic algo_b(K, 4.0, 0.05);
        ADSwitchFast  algo_f(K, 4.0, 0.05);

        for (int t = 1; t <= T; ++t) {
            int arm_b = algo_b.select_arm(t);
            int arm_f = algo_f.select_arm(t);

            auto m_b = env_b.means();
            auto m_f = env_f.means();
            double best_b = *std::max_element(m_b.begin(), m_b.end());
            double best_f = *std::max_element(m_f.begin(), m_f.end());

            regret_basic += best_b - m_b[arm_b];
            regret_fast  += best_f - m_f[arm_f];

            algo_b.update(arm_b, env_b.get_reward(arm_b));
            algo_f.update(arm_f, env_f.get_reward(arm_f));
            env_b.advance();
            env_f.advance();
        }
    }

    // Neither algorithm should have regret more than 5× the other.
    ASSERT_TRUE(regret_basic < 5.0 * regret_fast + 1.0);
    ASSERT_TRUE(regret_fast  < 5.0 * regret_basic + 1.0);
}

// ===========================================================================
// main
// ===========================================================================
int main() {
    std::cout << "Running ADSWITCH tests\n";

    // Environment tests
    test_stochastic_env();
    test_drifting_env();
    test_sharp_switch_env();
    test_big_switch_env();
    test_small_switch_env();

    // ADSwitchBasic tests
    test_basic_construction();
    test_basic_initialisation_pulls();
    test_basic_invalid_update();
    test_basic_no_change_stationary();
    test_basic_detects_large_change();
    test_basic_resets_on_sharp_switch();
    test_basic_ucb_favours_best_arm();

    // ADSwitchFast tests
    test_fast_construction();
    test_fast_initialisation_pulls();
    test_fast_no_change_stationary();
    test_fast_detects_large_change();
    test_fast_resets_on_sharp_switch();
    test_fast_ucb_favours_best_arm();

    // Cross-algorithm tests
    test_both_regret_comparable();

    // Report results
    std::cout << "\n========================================\n";
    std::cout << "Tests run:    " << g_tests_run    << "\n";
    std::cout << "Tests failed: " << g_tests_failed << "\n";
    std::cout << "========================================\n";

    return g_tests_failed == 0 ? 0 : 1;
}
