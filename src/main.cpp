// ADSWITCH experiment runner
// Exercises ADSwitchBasic, ADSwitchFast, and QBLSingle on six environment types and
// reports cumulative regret, number of epoch resets, runtime, and writes
// accumulated-regret SVG plots with reset markers to plots/.
//
// Usage: adswitch_runner [--fast-only] [T] [K]
//   --fast-only  run only ADSwitchFast experiments
//   T            time horizon   (default: 10000)
//   K            number of arms (default: 5)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "adswitch_basic.hpp"
#include "adswitch_fast.hpp"
#include "environments.hpp"
#include "qbl_single.hpp"

class UniformRandomBaseline {
public:
    UniformRandomBaseline(int K, unsigned seed = 12345)
        : K_(K), rng_(seed), dist_(0, K - 1) {}

    int select_arm(int /*t*/) {
        return dist_(rng_);
    }

    bool update(int /*arm*/, double /*reward*/) {
        return false;
    }

    int num_resets() const {
        return 0;
    }

private:
    int K_;
    std::mt19937 rng_;
    std::uniform_int_distribution<int> dist_;
};

// ---------------------------------------------------------------------------
// Run a single algorithm on a single environment for T rounds.
// Returns {cumulative_regret, num_resets, elapsed_ms}.
// ---------------------------------------------------------------------------
template <typename Algorithm>
struct RunResult {
    double cumulative_regret;
    int    num_resets;
    double elapsed_ms;
    std::vector<double> regret_trace;
    std::vector<int>    reset_rounds;
};

struct PlotSeries {
    std::string         label;
    std::string         color;
    std::vector<double> regret_trace;
    std::vector<int>    reset_rounds;
};

static std::string sanitize_name(const std::string& name) {
    std::string out;
    out.reserve(name.size());
    for (unsigned char ch : name) {
        if (std::isalnum(ch)) {
            out.push_back(static_cast<char>(std::tolower(ch)));
        } else {
            out.push_back('_');
        }
    }
    return out;
}

static std::string format_double(double value, int precision = 2) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

static void write_regret_plot_svg(const std::string& env_name,
                                  const std::vector<PlotSeries>& series_list,
                                  const std::vector<int>& shift_rounds,
                                  int T) {
    if (series_list.empty() || T <= 0) return;

    const int width = 1000;
    const int height = 620;
    const int left = 80;
    const int right = 30;
    const int top = 80;
    const int bottom = 70;
    const int plot_width = width - left - right;
    const int plot_height = height - top - bottom;
    const int reset_lane_gap = 12;
    const int reset_lane_height = 8;

    double max_regret = 0.0;
    double min_positive_regret = -1.0;
    for (const auto& series : series_list) {
        for (double regret : series.regret_trace) {
            max_regret = std::max(max_regret, regret);
            if (regret > 0.0 &&
                (min_positive_regret < 0.0 || regret < min_positive_regret)) {
                min_positive_regret = regret;
            }
        }
    }
    if (max_regret <= 0.0) max_regret = 1.0;

    const double x_min = static_cast<double>(std::min(std::max(1000, 1), std::max(T, 1)));
    const double x_max = static_cast<double>(std::max(T, 1));
    const double y_min = (min_positive_regret > 0.0)
                             ? std::min(min_positive_regret, max_regret)
                             : 1e-3;
    const double y_max = std::max(max_regret, y_min * 10.0);

    const double log_x_min = std::log10(x_min);
    const double log_x_max = std::log10(x_max);
    const double log_y_min = std::log10(y_min);
    const double log_y_max = std::log10(y_max);

    auto x_to_svg = [&](int t) -> double {
        if (T <= 1 || log_x_max <= log_x_min) return static_cast<double>(left);
        const double clamped_t = static_cast<double>(std::max(t, static_cast<int>(x_min)));
        const double log_x = std::log10(clamped_t);
        return static_cast<double>(left) +
               static_cast<double>(plot_width) * (log_x - log_x_min) /
                   (log_x_max - log_x_min);
    };
    auto y_to_svg = [&](double regret) -> double {
        const double clamped_regret = std::max(regret, y_min);
        const double log_y = std::log10(clamped_regret);
        return static_cast<double>(top) +
               static_cast<double>(plot_height) * (1.0 - (log_y - log_y_min) /
                                                             (log_y_max - log_y_min));
    };

    std::filesystem::create_directories("plots");
    const std::string file_name = "plots/" + sanitize_name(env_name) + "_regret.svg";
    std::ofstream out(file_name);
    if (!out) {
        std::cerr << "Failed to write plot: " << file_name << "\n";
        return;
    }

    out << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width
        << " " << height << "\">\n";
    out << "  <rect x=\"0\" y=\"0\" width=\"" << width
        << "\" height=\"" << height << "\" fill=\"white\"/>\n";
    out << "  <text x=\"" << (width / 2) << "\" y=\"28\" text-anchor=\"middle\""
        << " font-family=\"sans-serif\" font-size=\"22\">Accumulated regret (log-log) – "
        << env_name << "</text>\n";

    const int reset_lane_top = top - 8 -
        static_cast<int>(series_list.size()) * reset_lane_gap;

    const int y_tick_start = static_cast<int>(std::floor(log_y_min));
    const int y_tick_end = static_cast<int>(std::ceil(log_y_max));
    for (int exp = y_tick_start; exp <= y_tick_end; ++exp) {
        const double value = std::pow(10.0, static_cast<double>(exp));
        if (value < y_min || value > y_max * 1.000001) continue;
        const double y = y_to_svg(value);
        out << "  <line x1=\"" << left << "\" y1=\"" << y
            << "\" x2=\"" << (left + plot_width) << "\" y2=\"" << y
            << "\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << (left - 10) << "\" y=\"" << (y + 5)
            << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
            << "1e" << exp << "</text>\n";
    }

    const int x_tick_start = static_cast<int>(std::ceil(log_x_min));
    const int x_tick_end = static_cast<int>(std::floor(log_x_max));
    for (int exp = x_tick_start; exp <= x_tick_end; ++exp) {
        int t = static_cast<int>(std::pow(10.0, static_cast<double>(exp)));
        if (t < static_cast<int>(x_min) || t > T) continue;
        const double x = x_to_svg(t);
        out << "  <line x1=\"" << x << "\" y1=\"" << top
            << "\" x2=\"" << x << "\" y2=\"" << (top + plot_height)
            << "\" stroke=\"#f3f4f6\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << x << "\" y=\"" << (top + plot_height + 24)
            << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
            << t << "</text>\n";
    }
    if (T > static_cast<int>(x_min)) {
        const double x = x_to_svg(T);
        out << "  <line x1=\"" << x << "\" y1=\"" << top
            << "\" x2=\"" << x << "\" y2=\"" << (top + plot_height)
            << "\" stroke=\"#f3f4f6\" stroke-width=\"1\"/>\n";
        out << "  <text x=\"" << x << "\" y=\"" << (top + plot_height + 24)
            << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
            << T << "</text>\n";
    }

    out << "  <line x1=\"" << left << "\" y1=\"" << (top + plot_height)
        << "\" x2=\"" << (left + plot_width) << "\" y2=\"" << (top + plot_height)
        << "\" stroke=\"#111827\" stroke-width=\"2\"/>\n";
    out << "  <line x1=\"" << left << "\" y1=\"" << top
        << "\" x2=\"" << left << "\" y2=\"" << (top + plot_height)
        << "\" stroke=\"#111827\" stroke-width=\"2\"/>\n";
    out << "  <text x=\"" << (left + plot_width / 2) << "\" y=\"" << (height - 20)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Round (log scale)</text>\n";
    out << "  <text x=\"24\" y=\"" << (top + plot_height / 2)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\""
        << " transform=\"rotate(-90 24 " << (top + plot_height / 2)
        << ")\">Accumulated regret (log scale)</text>\n";

    for (std::size_t idx = 0; idx < series_list.size(); ++idx) {
        const auto& series = series_list[idx];
        const int lane_y = reset_lane_top + static_cast<int>(idx) * reset_lane_gap;
        out << "  <line x1=\"" << left << "\" y1=\"" << lane_y
            << "\" x2=\"" << (left + plot_width) << "\" y2=\"" << lane_y
            << "\" stroke=\"#f3f4f6\" stroke-width=\"1\"/>\n";
        for (int reset_t : series.reset_rounds) {
            if (reset_t < static_cast<int>(x_min) || reset_t > T) continue;
            const double x = x_to_svg(reset_t);
            out << "  <line x1=\"" << x << "\" y1=\"" << (lane_y - reset_lane_height / 2)
                << "\" x2=\"" << x << "\" y2=\"" << (lane_y + reset_lane_height / 2)
                << "\" stroke=\"" << series.color
                << "\" stroke-width=\"1.5\" opacity=\"0.85\"/>\n";
        }
    }

    for (int shift_t : shift_rounds) {
        if (shift_t < static_cast<int>(x_min) || shift_t > T) continue;
        const double x = x_to_svg(shift_t);
        out << "  <line x1=\"" << x << "\" y1=\"" << top
            << "\" x2=\"" << x << "\" y2=\"" << (top + plot_height)
            << "\" stroke=\"#10b981\" stroke-width=\"2\" opacity=\"0.65\"/>\n";
    }

    for (const auto& series : series_list) {
        if (series.regret_trace.empty()) continue;
        out << "  <polyline fill=\"none\" stroke=\"" << series.color
            << "\" stroke-width=\"2.5\" points=\"";
        for (int t = 1; t <= static_cast<int>(series.regret_trace.size()); ++t) {
            out << format_double(x_to_svg(t), 2) << ","
                << format_double(y_to_svg(series.regret_trace[t - 1]), 2);
            if (t != static_cast<int>(series.regret_trace.size())) out << " ";
        }
        out << "\"/>\n";
    }

    const int legend_x = left + 12;
    int legend_y = top + 18;
    if (!shift_rounds.empty()) {
        out << "  <line x1=\"" << legend_x << "\" y1=\"" << legend_y
            << "\" x2=\"" << (legend_x + 24) << "\" y2=\"" << legend_y
            << "\" stroke=\"#10b981\" stroke-width=\"2\" opacity=\"0.65\"/>\n";
        out << "  <text x=\"" << (legend_x + 32) << "\" y=\"" << (legend_y + 4)
            << "\" font-family=\"sans-serif\" font-size=\"13\">true shifts</text>\n";
        legend_y += 24;
    }
    for (const auto& series : series_list) {
        out << "  <line x1=\"" << legend_x << "\" y1=\"" << legend_y
            << "\" x2=\"" << (legend_x + 24) << "\" y2=\"" << legend_y
            << "\" stroke=\"" << series.color << "\" stroke-width=\"3\"/>\n";
        out << "  <text x=\"" << (legend_x + 32) << "\" y=\"" << (legend_y + 4)
            << "\" font-family=\"sans-serif\" font-size=\"13\">"
            << series.label << "</text>\n";
        if (!series.reset_rounds.empty()) {
            const int dash_y = legend_y + 16;
            for (int offset = 0; offset <= 24; offset += 6) {
                out << "  <line x1=\"" << (legend_x + offset) << "\" y1=\"" << (dash_y - 4)
                    << "\" x2=\"" << (legend_x + offset) << "\" y2=\"" << (dash_y + 4)
                    << "\" stroke=\"" << series.color
                    << "\" stroke-width=\"1.5\" opacity=\"0.85\"/>\n";
            }
            out << "  <text x=\"" << (legend_x + 32) << "\" y=\"" << (dash_y + 4)
                << "\" font-family=\"sans-serif\" font-size=\"12\">reset ticks</text>\n";
            legend_y += 38;
        } else {
            legend_y += 24;
        }
    }

    out << "</svg>\n";
    std::cout << "Plot saved: " << file_name << "\n";
}

template <typename Algorithm>
RunResult<Algorithm> run_experiment(Algorithm& algo, BanditEnvironment& env, int T) {
    double cum_regret = 0.0;
    std::vector<double> regret_trace;
    std::vector<int> reset_rounds;
    regret_trace.reserve(T);

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
        const bool reset = algo.update(arm, reward);
        if (reset) {
            reset_rounds.push_back(t);
        }
        regret_trace.push_back(cum_regret);

        // 5. Advance the environment (non-stationary update).
        env.advance();
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    return {cum_regret, algo.num_resets(), elapsed_ms,
            std::move(regret_trace), std::move(reset_rounds)};
}

// ---------------------------------------------------------------------------
// Run the selected algorithm variants on the given environment.
// ---------------------------------------------------------------------------
static void compare(const std::string& env_name,
                    const std::function<std::unique_ptr<BanditEnvironment>()>& make_env,
                    int T, int K, bool fast_only) {
    std::vector<PlotSeries> plot_series;
    const std::vector<int> shift_rounds = make_env()->shift_rounds();

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
        auto env = make_env();
        UniformRandomBaseline algo(K, /*seed=*/2026);
        auto res = run_experiment(algo, *env, T);
        print_row("Uniform", res.cumulative_regret, res.num_resets, res.elapsed_ms);
        plot_series.push_back({"Uniform", "#6b7280",
                               std::move(res.regret_trace), std::move(res.reset_rounds)});
    }

    {
        auto env = make_env();
        QBLSingle algo(K, /*gamma=*/0.1, /*seed=*/2027);
        auto res = run_experiment(algo, *env, T);
        print_row("QBL", res.cumulative_regret, res.num_resets, res.elapsed_ms);
        plot_series.push_back({"QBL", "#7c3aed",
                               std::move(res.regret_trace), std::move(res.reset_rounds)});
    }

    if (!fast_only) {
        auto env  = make_env();
        ADSwitchBasic algo(K, T, /*delta=*/0.05);
        auto res = run_experiment(algo, *env, T);
        print_row("Basic", res.cumulative_regret, res.num_resets, res.elapsed_ms);
        plot_series.push_back({"Basic", "#2563eb",
                               std::move(res.regret_trace), std::move(res.reset_rounds)});
    }
    {
        auto env  = make_env();
        ADSwitchFast  algo(K, T, /*delta=*/0.05);
        auto res = run_experiment(algo, *env, T);
        print_row("Fast (R3)", res.cumulative_regret, res.num_resets, res.elapsed_ms);
        plot_series.push_back({"Fast (R3)", "#dc2626",
                               std::move(res.regret_trace), std::move(res.reset_rounds)});
    }

    write_regret_plot_svg(env_name, plot_series, shift_rounds, T);
}

// ---------------------------------------------------------------------------
// Generate K stationary means spread in [0.3, 0.8].
// ---------------------------------------------------------------------------
static std::vector<double> make_stationary_means(int K) {
    std::vector<double> m(K);
    if (K == 1) {
        m[0] = 0.55;
    } else {
        for (int i = 0; i < K; ++i)
            m[i] = 0.3 + 0.5 * static_cast<double>(i) / (K - 1);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Generate sharp-switch phases for K arms with num_phases phases over T.
// Each phase picks a different best arm (cycling through 0..K-1).
// ---------------------------------------------------------------------------
static std::vector<std::pair<int, std::vector<double>>>
make_sharp_phases(int K, int T, int num_phases) {
    using Phase = std::pair<int, std::vector<double>>;
    std::vector<Phase> phases;
    for (int p = 0; p < num_phases; ++p) {
        int start = (p == 0) ? 0 : p * T / num_phases;
        int best  = p % K;
        std::vector<double> means(K, 0.3);
        means[best] = 0.9;
        phases.push_back({start, means});
    }
    return phases;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int T = 10000;
    int K = 5;
    bool fast_only = false;
    int positional = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--fast-only") {
            fast_only = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--fast-only] [T] [K]\n"
                      << "  --fast-only  run only ADSwitchFast experiments\n"
                      << "  T            time horizon (positive integer, default 10000)\n"
                      << "  K            number of arms (positive integer, default 5)\n";
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown flag: " << arg << "\n"
                      << "Usage: " << argv[0] << " [--fast-only] [T] [K]\n";
            return 1;
        } else if (positional == 0) {
            T = std::atoi(argv[i]);
            positional++;
        } else if (positional == 1) {
            K = std::atoi(argv[i]);
            positional++;
        } else {
            std::cerr << "Too many positional arguments.\n"
                      << "Usage: " << argv[0] << " [--fast-only] [T] [K]\n";
            return 1;
        }
    }

    if (T <= 0 || K <= 0) {
        std::cerr << "Usage: " << argv[0] << " [--fast-only] [T] [K]\n"
                  << "  --fast-only  run only ADSwitchFast experiments\n"
                  << "  T = time horizon (positive integer, default 10000)\n"
                  << "  K = number of arms (positive integer, default 5)\n";
        return 1;
    }

    std::cout << "ADSWITCH Experiment Runner\n";
    std::cout << "Arms K=" << K << "  Horizon T=" << T << "\n";
    if (fast_only) {
        std::cout << "Mode: Fast implementation only\n";
    }

    // ------------------------------------------------------------------
    // 1. Stochastic (stationary) environment
    // ------------------------------------------------------------------
    compare("Stochastic",
            [K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<StochasticEnv>(
                    make_stationary_means(K), /*seed=*/1);
            },
            T, K, fast_only);

    // ------------------------------------------------------------------
    // 2. Drifting stochastic environment
    // ------------------------------------------------------------------
    compare("Drifting",
            [K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<DriftingEnv>(
                    make_stationary_means(K),
                    /*drift_rate=*/0.002, /*seed=*/2);
            },
            T, K, fast_only);

    // ------------------------------------------------------------------
    // 3. Sharp-switch environment (generated phases)
    // ------------------------------------------------------------------
    compare("Sharp-switch",
            [K, T]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<SharpSwitchEnv>(
                    make_sharp_phases(K, T, /*num_phases=*/4), /*seed=*/3);
            },
            T, K, fast_only);

    // ------------------------------------------------------------------
    // 4. Big-switch environment (large gap, several switches)
    // ------------------------------------------------------------------
    compare("Big-switch",
            [T, K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<BigSwitchEnv>(K, T, /*num_switches=*/5,
                                                      /*gap=*/0.4, /*seed=*/4);
            },
            T, K, fast_only);

    // ------------------------------------------------------------------
    // 5. Small-switch environment (small gap, several switches)
    // ------------------------------------------------------------------
    compare("Small-switch",
            [T, K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<SmallSwitchEnv>(K, T, /*num_switches=*/5,
                                                        /*gap=*/0.1, /*seed=*/5);
            },
            T, K, fast_only);

    // ------------------------------------------------------------------
    // 6. Mod2 environment (parity swaps with increasingly long phases)
    // ------------------------------------------------------------------
    compare("Mod2",
            [T, K]() -> std::unique_ptr<BanditEnvironment> {
                return std::make_unique<Mod2Env>(K, T, /*seed=*/6);
            },
            T, K, fast_only);

    std::cout << "\nDone.\n";
    return 0;
}
