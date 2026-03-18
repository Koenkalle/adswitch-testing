// Unit tests for ADSwitchBasic, ADSwitchFast, and the test environments.
// Build and run with CMake (target: adswitch_tests).
//
// Constructor API: ADSwitchBasic(K, T, delta=0.05, seed=42)
//                  ADSwitchFast (K, T, delta=0.05, seed=42)
//
// Test parameters are chosen so that the paper's exact threshold formulas
// produce detections well within the round budgets, and so that O(n³) work
// per pull remains tractable (basic) or negligible (fast).

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "adswitch_basic.hpp"
#include "adswitch_fast.hpp"
#include "environments.hpp"
#include "qbl_single.hpp"

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

// Upper bound on spurious episode resets accepted in a stationary environment.
static constexpr int kMaxSpuriousResets = 3;

// ===========================================================================
// Environment tests
// ===========================================================================

static void test_stochastic_env() {
    test_section("StochasticEnv");

    std::vector<double> means = {0.3, 0.7, 0.5};
    StochasticEnv env(means, /*seed=*/42);

    ASSERT_TRUE(env.num_arms() == 3);
    ASSERT_TRUE(env.best_arm() == 1);
    ASSERT_NEAR(env.means()[1], 0.7, 1e-9);

    for (int i = 0; i < 100; ++i) {
        double r = env.get_reward(0);
        ASSERT_TRUE(r == 0.0 || r == 1.0);
        env.advance();
    }

    ASSERT_NEAR(env.means()[0], 0.3, 1e-9);
    ASSERT_TRUE(env.time() == 100);
    ASSERT_THROWS(env.get_reward(3), std::out_of_range);
    ASSERT_THROWS(StochasticEnv({}), std::invalid_argument);
}

static void test_drifting_env() {
    test_section("DriftingEnv");

    std::vector<double> init = {0.5, 0.5, 0.5};
    DriftingEnv env(init, /*drift_rate=*/0.01, /*seed=*/7);

    ASSERT_TRUE(env.num_arms() == 3);
    for (int i = 0; i < 500; ++i) env.advance();
    auto m = env.means();
    for (double v : m) ASSERT_TRUE(v >= 0.01 && v <= 0.99);
    ASSERT_TRUE(env.time() == 500);
}

static void test_sharp_switch_env() {
    test_section("SharpSwitchEnv");

    using Phase = std::pair<int, std::vector<double>>;
    SharpSwitchEnv env(
        {Phase{0, {0.9, 0.1}}, Phase{100, {0.1, 0.9}}}, /*seed=*/42);

    ASSERT_TRUE(env.num_arms() == 2);
    ASSERT_TRUE(env.best_arm() == 0);
    for (int t = 0; t < 99; ++t) env.advance();
    ASSERT_TRUE(env.best_arm() == 0);
    env.advance();
    ASSERT_TRUE(env.best_arm() == 1);
    ASSERT_THROWS(SharpSwitchEnv({Phase{1, {0.5}}}), std::invalid_argument);
}

static void test_big_switch_env() {
    test_section("BigSwitchEnv");

    BigSwitchEnv env(4, 1000, 3, /*gap=*/0.4, /*seed=*/99);
    ASSERT_TRUE(env.num_arms() == 4);
    ASSERT_TRUE(static_cast<int>(env.phases().size()) == 4);

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
    ASSERT_TRUE(static_cast<int>(env.phases().size()) == 3);
}

static void test_mod2_env() {
    test_section("Mod2Env");

    Mod2Env env(4, 100, /*seed=*/13);
    ASSERT_TRUE(env.num_arms() == 4);

    auto m0 = env.means();
    ASSERT_TRUE(m0[0] == 1.0);
    ASSERT_TRUE(m0[1] == 0.0);
    ASSERT_TRUE(m0[2] == 1.0);
    ASSERT_TRUE(m0[3] == 0.0);

    for (int t = 0; t < 3; ++t) env.advance();
    auto m1 = env.means();
    ASSERT_TRUE(m1[0] == 0.0);
    ASSERT_TRUE(m1[1] == 1.0);
    ASSERT_TRUE(m1[2] == 0.0);
    ASSERT_TRUE(m1[3] == 1.0);

    for (int t = 0; t < 9; ++t) env.advance();
    auto m2 = env.means();
    ASSERT_TRUE(m2[0] == 1.0);
    ASSERT_TRUE(m2[1] == 0.0);

    ASSERT_THROWS(Mod2Env(0, 100), std::invalid_argument);
    ASSERT_THROWS(Mod2Env(3, 0), std::invalid_argument);
}

// ===========================================================================
// ADSwitchBasic tests
// ===========================================================================

static void test_qbl_construction() {
    test_section("QBLSingle construction");

    QBLSingle algo(5, 0.1, /*seed=*/9);
    ASSERT_TRUE(algo.num_arms() == 5);
    ASSERT_TRUE(algo.num_resets() == 0);

    ASSERT_THROWS(QBLSingle(0, 0.1), std::invalid_argument);
    ASSERT_THROWS(QBLSingle(3, -0.1), std::invalid_argument);
    ASSERT_THROWS(QBLSingle(3, 1.1), std::invalid_argument);
}

static void test_qbl_selects_valid_arm() {
    test_section("QBLSingle selects valid arm");

    constexpr int K = 4;
    QBLSingle algo(K, 0.1, /*seed=*/11);
    const int arm = algo.select_arm(1);
    ASSERT_TRUE(arm >= 0 && arm < K);
}

static void test_qbl_demotes_poor_leader() {
    test_section("QBLSingle demotes poor leader");

    QBLSingle algo(2, /*gamma=*/0.0, /*seed=*/13);
    const int arm = algo.select_arm(1);
    const bool demoted = algo.update(arm, /*reward=*/0.0);

    ASSERT_TRUE(demoted);
    ASSERT_TRUE(algo.num_resets() == 1);
}

static void test_basic_construction() {
    test_section("ADSwitchBasic construction");

    // Valid construction: K=5, T=100.
    ADSwitchBasic algo(5, 100);
    ASSERT_TRUE(algo.num_arms()   == 5);
    ASSERT_TRUE(algo.num_resets() == 0);
    ASSERT_TRUE(algo.episode()    == 1);  // episode starts at 1 after init

    // Invalid arguments.
    ASSERT_THROWS(ADSwitchBasic(0, 100),       std::invalid_argument);  // K=0
    ASSERT_THROWS(ADSwitchBasic(3, 0),         std::invalid_argument);  // T=0
    ASSERT_THROWS(ADSwitchBasic(3, -1),        std::invalid_argument);  // T<0
    ASSERT_THROWS(ADSwitchBasic(3, 100, 0.0),  std::invalid_argument);  // delta=0
    ASSERT_THROWS(ADSwitchBasic(3, 100, 1.0),  std::invalid_argument);  // delta=1
}

static void test_basic_initialisation_pulls() {
    test_section("ADSwitchBasic initialisation pulls");

    // Line 13: least-recently-pulled.  The first K pulls must visit each arm.
    constexpr int K = 4;
    ADSwitchBasic algo(K, /*T=*/100);

    std::vector<int> init_arms;
    for (int t = 1; t <= K; ++t) {
        int arm = algo.select_arm(t);
        init_arms.push_back(arm);
        algo.update(arm, 0.5);
    }
    std::sort(init_arms.begin(), init_arms.end());
    for (int a = 0; a < K; ++a) ASSERT_TRUE(init_arms[a] == a);
}

static void test_basic_invalid_update() {
    test_section("ADSwitchBasic invalid update");

    ADSwitchBasic algo(3, 100);
    ASSERT_THROWS(algo.update(-1, 0.5), std::out_of_range);
    ASSERT_THROWS(algo.update(3,  0.5), std::out_of_range);
}

static void test_basic_no_change_stationary() {
    test_section("ADSwitchBasic stationary environment – no spurious resets");

    // K=2, T=100: thresholds from log(100).  In 100 rounds with stationary
    // Bernoulli arms the probability of any spurious condition-(3) firing
    // is negligible (< 1e-20 per split checked).
    constexpr int K = 2;
    constexpr int T = 100;
    ADSwitchBasic algo(K, T, /*delta=*/0.05);

    StochasticEnv env({0.4, 0.8}, /*seed=*/3);
    int resets = 0;
    for (int t = 1; t <= T; ++t) {
        int  arm   = algo.select_arm(t);
        bool reset = algo.update(arm, env.get_reward(arm));
        if (reset) ++resets;
        env.advance();
    }
    ASSERT_TRUE(resets <= kMaxSpuriousResets);
}

static void test_basic_detects_large_change() {
    test_section("ADSwitchBasic detects large distribution change");

    // K=1, T=100.  Deterministic step: reward 0 for pulls 1-32, reward 1 after.
    // At pull 63 (32 zeros + 31 ones), the triple (s1=0, s2=31, s=32) gives:
    //   thresh = sqrt(2·ln100/32) + sqrt(2·ln100/31) ≈ 0.537 + 0.545 = 1.082 > 1.0.
    // At pull 69 (32+37), same triple:
    //   thresh = sqrt(2·ln100/32) + sqrt(2·ln100/37) ≈ 0.537 + 0.499 = 1.036 > 1.0.
    // At pull 100 (32+68), same triple:
    //   thresh = 0.537 + sqrt(2·ln100/68) ≈ 0.537 + 0.368 = 0.905 < 1.0  → DETECTED.
    constexpr int K = 1;
    constexpr int T = 100;
    ADSwitchBasic algo(K, T);

    bool detected = false;
    for (int t = 1; t <= T; ++t) {
        int    arm = algo.select_arm(t);
        double r   = (t <= 32) ? 0.0 : 1.0;
        if (algo.update(arm, r)) { detected = true; break; }
    }
    ASSERT_TRUE(detected);
    ASSERT_TRUE(algo.num_resets() >= 1);
}

static void test_basic_resets_on_two_switches() {
    test_section("ADSwitchBasic resets on two deterministic switches");

    // K=1, T=100.  Three phases: [1-64]=0, [65-192]=1, [193+]=0.
    //
    // Episode 1 ends with a reset (change 0→1):
    //   At pull 100 (64 zeros + 36 ones), triple (s1=0,s2=63,s=64):
    //     thresh ≈ sqrt(2·ln100/64) + sqrt(2·ln100/36) ≈ 0.379+0.506 = 0.885 < 1.0 → detect.
    //   So reset_1 ≈ round 100.
    //
    // Episode 2 starts; arm 0 receives all-ones until round 192, then zeros.
    //   Let n_ep2 be pull count in episode 2.  At n_ep2=63+37=100 pulls, split (0,62,63):
    //     thresh ≈ 0.382 + sqrt(2·ln100/37) ≈ 0.382+0.499 = 0.881 < 1.0 → detect.
    //   Switch 1→0 is at round 192.  n_ep2 at switch ≈ 192-100+1 = 93 pre-switch pulls.
    //   Detection: 93+37=130 pulls → round ≈ 192+37 = 229 → reset_2 ≈ round 229.
    //
    // Run 250 rounds.  Expect resets >= 2.
    constexpr int K = 1;
    constexpr int T = 100;
    constexpr int rounds = 250;
    ADSwitchBasic algo(K, T);

    int resets = 0;
    for (int t = 1; t <= rounds; ++t) {
        int    arm = algo.select_arm(t);
        double r;
        if (t <= 64)        r = 0.0;
        else if (t <= 192)  r = 1.0;
        else                r = 0.0;
        if (algo.update(arm, r)) ++resets;
    }
    ASSERT_TRUE(resets >= 2);
}

static void test_basic_identifies_bad_arms() {
    test_section("ADSwitchBasic identifies sub-optimal arms as BAD");

    // K=3, T=100.  Arm 1 is best (mean 0.8); arms 0 and 2 are bad (mean 0.2).
    // Condition (1): evict arm a when best_mu − mu_a > sqrt(4·ln(100)/n_a).
    // Gap = 0.6, threshold < 0.6 once n_a > 4·ln100/0.36 ≈ 51.
    // With K=3 and round-robin, each arm has ~133 pulls after 400 rounds.
    // Both sub-optimal arms should be evicted well before round 400.
    constexpr int K = 3;
    constexpr int T = 100;
    ADSwitchBasic algo(K, T, 0.05, /*seed=*/7);

    StochasticEnv env({0.2, 0.8, 0.2}, /*seed=*/7);
    for (int t = 1; t <= 400; ++t) {
        int arm = algo.select_arm(t);
        algo.update(arm, env.get_reward(arm));
        env.advance();
    }
    // Arm 1 (best) must still be GOOD; arms 0 and 2 must be BAD.
    ASSERT_TRUE( algo.is_good(1));
    ASSERT_FALSE(algo.is_good(0));
    ASSERT_FALSE(algo.is_good(2));
}

// ===========================================================================
// ADSwitchFast tests
// ===========================================================================

static void test_fast_construction() {
    test_section("ADSwitchFast construction");

    ADSwitchFast algo(5, 100);
    ASSERT_TRUE(algo.num_arms()   == 5);
    ASSERT_TRUE(algo.num_resets() == 0);
    ASSERT_TRUE(algo.episode()    == 1);

    ASSERT_THROWS(ADSwitchFast(0, 100),      std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, 0),        std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, -1),       std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, 100, 0.0), std::invalid_argument);
    ASSERT_THROWS(ADSwitchFast(3, 100, 1.0), std::invalid_argument);
}

static void test_fast_initialisation_pulls() {
    test_section("ADSwitchFast initialisation pulls");

    constexpr int K = 4;
    ADSwitchFast algo(K, 100);

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

    // T=500: L_base = round(ln500) = 6.  Smallest window length = 6.
    // Threshold for length-6 pair: 2·sqrt(2·ln500/6) ≈ 2·1.44 = 2.88.
    // Exceeding 1.0 with Bernoulli rewards is virtually impossible.
    constexpr int K = 2;
    constexpr int T = 500;
    ADSwitchFast algo(K, T, 0.05);

    StochasticEnv env({0.4, 0.8}, /*seed=*/3);
    int resets = 0;
    for (int t = 1; t <= T; ++t) {
        int  arm = algo.select_arm(t);
        if (algo.update(arm, env.get_reward(arm))) ++resets;
        env.advance();
    }
    ASSERT_TRUE(resets <= kMaxSpuriousResets);
}

static void test_fast_detects_large_change() {
    test_section("ADSwitchFast detects large distribution change");

    // K=1, T=2000.  L_base = round(ln2000) = 8.  Lengths: 8,16,32,64,128,…
    // Deterministic step: reward 0 for pulls 1-64, reward 1 after.
    //
    // At pull 128 (64 zeros + 64 ones), L=64 window:
    //   update_window_stats adds window [64..127] (mean=1.0), so max_win=1.0.
    //   min_win = 0.0  (from window [0..63] added at pull 64).
    //   recent_mean (last 64) = 1.0.
    //   pair (k_hist=64, k_recent=64): n=128 >= 64+64.
    //   thresh = 2·sqrt(2·ln2000/64) ≈ 2·0.487 = 0.974.
    //   |min_win − recent| = |0.0 − 1.0| = 1.0 > 0.974  → DETECTED.
    constexpr int K = 1;
    constexpr int T = 2000;
    ADSwitchFast algo(K, T);

    bool detected = false;
    for (int t = 1; t <= 200; ++t) {
        int    arm = algo.select_arm(t);
        double r   = (t <= 64) ? 0.0 : 1.0;
        if (algo.update(arm, r)) { detected = true; break; }
    }
    ASSERT_TRUE(detected);
    ASSERT_TRUE(algo.num_resets() >= 1);
}

static void test_fast_resets_on_two_switches() {
    test_section("ADSwitchFast resets on two deterministic switches");

    // K=1, T=2000.  L_base=8, lengths 8,16,32,64,128,…
    // Three phases: [1-64]=0, [65-192]=1, [193+]=0.
    //
    // Episode 1 (0→1 at pull 64):
    //   At pull 128: min_win[L=64]=0 vs recent=1.0, diff=1.0>0.974 → reset_1 ≈ pull 128.
    //
    // Episode 2 starts; receives ones (rounds 129-192) then zeros (193+).
    //   In ep2: n_ep2=64 ones, then zeros.
    //   At n_ep2=128 (64+64): window [64..127]=64 zeros, mean=0.
    //   max_win[L=64]=1.0 (ep2's first 64 ones), min_win=0.
    //   recent (last 64)=0.  |max_win−recent|=1.0>0.974 → reset_2 ≈ round 256.
    //
    // Run 300 rounds.  Expect resets >= 2.
    constexpr int K = 1;
    constexpr int T = 2000;
    constexpr int rounds = 300;
    ADSwitchFast algo(K, T);

    int resets = 0;
    for (int t = 1; t <= rounds; ++t) {
        int    arm = algo.select_arm(t);
        double r;
        if (t <= 64)        r = 0.0;
        else if (t <= 192)  r = 1.0;
        else                r = 0.0;
        if (algo.update(arm, r)) ++resets;
    }
    ASSERT_TRUE(resets >= 2);
}

static void test_fast_identifies_bad_arms() {
    test_section("ADSwitchFast identifies sub-optimal arms as BAD");

    // K=3, T=2000.  Arms 0.2 / 0.8 / 0.2.
    // Condition (1): evict when best_mu−mu_a > sqrt(4·ln2000/n_a).
    // Gap=0.6, threshold < 0.6 once n_a > 4·ln2000/0.36 ≈ 84.
    // With K=3, each arm gets ~200 pulls in 600 rounds.
    constexpr int K = 3;
    constexpr int T = 2000;
    ADSwitchFast algo(K, T, 0.05, /*seed=*/7);

    StochasticEnv env({0.2, 0.8, 0.2}, /*seed=*/7);
    for (int t = 1; t <= 600; ++t) {
        int arm = algo.select_arm(t);
        algo.update(arm, env.get_reward(arm));
        env.advance();
    }
    ASSERT_TRUE( algo.is_good(1));
    ASSERT_FALSE(algo.is_good(0));
    ASSERT_FALSE(algo.is_good(2));
}

// ===========================================================================
// Parity test – Basic vs Fast comparable regret
// ===========================================================================

static void test_both_regret_comparable() {
    test_section("Basic vs Fast: comparable regret on big-switch env");

    // T=1000, K=4, 3 switches.  Basic is O(n³) per pull; with episode resets
    // the peak episode pull count stays small (~100-150 per arm), keeping
    // each run fast.  Neither algorithm should accumulate >5× the other's regret.
    constexpr int K = 4;
    constexpr int T = 1000;
    constexpr int rounds = T;

    double regret_basic = 0.0;
    double regret_fast  = 0.0;

    for (int seed_offset = 0; seed_offset < 5; ++seed_offset) {
        BigSwitchEnv env_b(K, T, 3, 0.4, 100 + seed_offset);
        BigSwitchEnv env_f(K, T, 3, 0.4, 100 + seed_offset);

        ADSwitchBasic algo_b(K, T, 0.05);
        ADSwitchFast  algo_f(K, T, 0.05);

        for (int t = 1; t <= rounds; ++t) {
            int arm_b = algo_b.select_arm(t);
            int arm_f = algo_f.select_arm(t);

            auto m_b   = env_b.means();
            auto m_f   = env_f.means();
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

    // Neither algorithm should accumulate more than 5× the other's regret.
    ASSERT_TRUE(regret_basic < 5.0 * regret_fast  + 1.0);
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
    test_mod2_env();

    // QBL tests
    test_qbl_construction();
    test_qbl_selects_valid_arm();
    test_qbl_demotes_poor_leader();

    // ADSwitchBasic tests
    test_basic_construction();
    test_basic_initialisation_pulls();
    test_basic_invalid_update();
    test_basic_no_change_stationary();
    test_basic_detects_large_change();
    test_basic_resets_on_two_switches();
    test_basic_identifies_bad_arms();

    // ADSwitchFast tests
    test_fast_construction();
    test_fast_initialisation_pulls();
    test_fast_no_change_stationary();
    test_fast_detects_large_change();
    test_fast_resets_on_two_switches();
    test_fast_identifies_bad_arms();

    // Cross-algorithm parity
    test_both_regret_comparable();

    std::cout << "\n========================================\n";
    std::cout << "Tests run:    " << g_tests_run    << "\n";
    std::cout << "Tests failed: " << g_tests_failed << "\n";
    std::cout << "========================================\n";

    return g_tests_failed == 0 ? 0 : 1;
}
