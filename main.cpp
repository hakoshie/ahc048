#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <tuple>
#include <iomanip>
#include <chrono>
#include <random>
#include <limits>
#include <numeric>
#include <set>
using namespace std;    
#define rep(i, n) for (int i = 0; i < (n); ++i)
#define FOR(i,k,n) for (int i = (k); i < (n); ++i)
// --- Global Random Number Generator ---
// std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::mt19937 rng(42); // Fixed seed for reproducibility

// --- Helper Structs and Functions ---
struct Color {
    double r, g, b;
    Color(double r_ = 0.0, double g_ = 0.0, double b_ = 0.0) : r(r_), g(g_), b(b_) {}

    Color operator+(const Color& other) const {
        return {r + other.r, g + other.g, b + other.b};
    }
    Color operator-(const Color& other) const {
        return {r - other.r, g - other.g, b - other.b};
    }
    Color operator*(double scalar) const {
        return {r * scalar, g * scalar, b * scalar};
    }
    Color operator/(double scalar) const {
        if (std::abs(scalar) < 1e-9) return {0,0,0}; // Avoid division by zero
        return {r / scalar, g / scalar, b / scalar};
    }
    Color& operator+=(const Color& other) {
        r += other.r; g += other.g; b += other.b;
        return *this;
    }
    // Dot product with another color (component-wise multiplication sum)
    double dot(const Color& other) const {
        return r * other.r + g * other.g + b * other.b;
    }
};

// Function to get mixed color based on counts x_i for each tube color
Color get_mixed_color_from_counts(const std::vector<int>& counts,
                                  const std::vector<Color>& tube_colors,
                                  int total_quantity) {
    if (total_quantity == 0 || tube_colors.empty() || counts.empty()) {
        return {0.0, 0.0, 0.0};
    }
    Color sum_color = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i] > 0) {
            sum_color += tube_colors[i] * counts[i];
        }
    }
    return sum_color / total_quantity;
}

// Function to get mixed color based on indices (as in original code)
Color get_mixed_color_from_indices(const std::vector<int>& indices_to_mix,
                                   const std::vector<Color>& own_colors_list_ref,
                                   int m_val_eff) {
    if (indices_to_mix.empty() || m_val_eff == 0) {
        return {0.0, 0.0, 0.0};
    }
    Color sum_color = {0.0, 0.0, 0.0};
    for (int idx : indices_to_mix) {
        if (idx >=0 && static_cast<size_t>(idx) < own_colors_list_ref.size())
            sum_color = sum_color + own_colors_list_ref[idx];
    }
    return sum_color / m_val_eff;
}


double calculate_error_sq(const Color& mixed_color, const Color& target_color) {
    Color diff = mixed_color - target_color;
    return diff.dot(diff); // (dr*dr + dg*dg + db*db)
}

double calculate_error_sqrt(const Color& mixed_color, const Color& target_color) {
    return std::sqrt(calculate_error_sq(mixed_color, target_color));
}

// --- Linear Algebra Utilities (Minimal for QP-like approach) ---
namespace LinAlg {
    using Matrix = std::vector<std::vector<double>>;
    using Vector = std::vector<double>;

    // Solves Ax = b using Gaussian elimination with partial pivoting.
    // Returns empty vector if matrix is singular or system is inconsistent.
    // Modifies A and b.
    Vector solve_linear_system(Matrix A_orig, Vector b_orig) { // Pass by value to modify copies
        Matrix A = A_orig;
        Vector b = b_orig;
        int n = A.size();
        if (n == 0 || A[0].size() != static_cast<size_t>(n) || b.size() != static_cast<size_t>(n)) {
            return {}; // Invalid input
        }

        for (int i = 0; i < n; ++i) {
            int pivot_row = i;
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > std::abs(A[pivot_row][i])) {
                    pivot_row = k;
                }
            }
            if (pivot_row != i) {
                std::swap(A[i], A[pivot_row]);
                std::swap(b[i], b[pivot_row]);
            }

            if (std::abs(A[i][i]) < 1e-12) { 
                return {}; // Singular or nearly singular
            }

            double pivot_val = A[i][i];
            for (int j = i; j < n; ++j) {
                A[i][j] /= pivot_val;
            }
            b[i] /= pivot_val;

            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = A[k][i];
                    for (int j = i; j < n; ++j) {
                        A[k][j] -= factor * A[i][j];
                    }
                    b[k] -= factor * b[i];
                }
            }
        }
        return b; 
    }
} // namespace LinAlg


// --- QP-like approach + Discretization ---
std::vector<double> calculate_continuous_weights(
    const std::vector<Color>& tube_colors,
    const Color& target_color) {
    
    int K = tube_colors.size();
    if (K == 0) return {};
    if (K == 1) return {1.0};

    LinAlg::Matrix A_system(K + 1, std::vector<double>(K + 1, 0.0));
    LinAlg::Vector B_rhs(K + 1, 0.0);

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            A_system[i][j] = 2.0 * tube_colors[i].dot(tube_colors[j]);
        }
    }

    for (int i = 0; i < K; ++i) {
        A_system[i][K] = 1.0;
        A_system[K][i] = 1.0;
    }
    A_system[K][K] = 0.0; 

    for (int i = 0; i < K; ++i) {
        B_rhs[i] = 2.0 * tube_colors[i].dot(target_color);
    }
    B_rhs[K] = 1.0; 

    LinAlg::Vector solution_w_lambda = LinAlg::solve_linear_system(A_system, B_rhs);

    std::vector<double> weights(K);
    if (solution_w_lambda.empty() || solution_w_lambda.size() != static_cast<size_t>(K+1)) { 
        for (int i = 0; i < K; ++i) weights[i] = 1.0 / K;
        return weights;
    }

    double sum_non_negative_w = 0.0;
    for (int i = 0; i < K; ++i) {
        weights[i] = std::max(0.0, solution_w_lambda[i]); 
        sum_non_negative_w += weights[i];
    }

    if (sum_non_negative_w < 1e-9) { 
        for (int i = 0; i < K; ++i) weights[i] = 1.0 / K; 
    } else {
        for (int i = 0; i < K; ++i) {
            weights[i] /= sum_non_negative_w; 
        }
    }
    return weights;
}

std::vector<int> discretize_weights(
    const std::vector<double>& normalized_weights,
    int Q_total_quantity) {
    
    int K = normalized_weights.size();
    if (K == 0) return {};
    if (Q_total_quantity == 0) return std::vector<int>(K, 0);

    std::vector<double> x_float(K);
    for (int i = 0; i < K; ++i) {
        x_float[i] = normalized_weights[i] * Q_total_quantity;
    }

    std::vector<int> x_int(K);
    int current_sum_int = 0;
    for (int i = 0; i < K; ++i) {
        x_int[i] = static_cast<int>(std::round(x_float[i]));
        current_sum_int += x_int[i];
    }

    int diff = Q_total_quantity - current_sum_int;
    
    std::vector<std::pair<double, int>> adjustment_candidates; 
    for(int i=0; i<K; ++i) {
        // For diff > 0 (need to add): priority is how much was rounded down (x_float - x_int)
        // For diff < 0 (need to subtract): priority is how much was rounded up (x_int - x_float)
        double priority_metric = (diff > 0) ? (x_float[i] - x_int[i]) : (x_int[i] - x_float[i]);
        adjustment_candidates.push_back({priority_metric, i});
    }
    // Sort descending by priority metric.
    // If diff > 0, we want to increment those with largest (x_float - x_int).
    // If diff < 0, we want to decrement those with largest (x_int - x_float).
    std::sort(adjustment_candidates.rbegin(), adjustment_candidates.rend()); 
    
    if (diff > 0) {
        for (int k = 0; k < diff; ++k) {
            if (adjustment_candidates.empty()) break;
            int candidate_idx_in_adj_list = k % adjustment_candidates.size(); // Cycle if diff is large
            x_int[adjustment_candidates[candidate_idx_in_adj_list].second]++;
        }
    } else if (diff < 0) {
        for (int k = 0; k < -diff; ++k) {
            if (adjustment_candidates.empty()) break;
            int candidate_idx_in_adj_list = k % adjustment_candidates.size();
            int original_tube_idx = adjustment_candidates[candidate_idx_in_adj_list].second;
            if (x_int[original_tube_idx] > 0) {
                x_int[original_tube_idx]--;
            } else { // Cannot decrement this one, try next best from sorted list
                bool decremented = false;
                for(size_t offset = 1; offset < adjustment_candidates.size(); ++offset) {
                    int next_candidate_idx_in_adj_list = (candidate_idx_in_adj_list + offset) % adjustment_candidates.size();
                    int next_original_tube_idx = adjustment_candidates[next_candidate_idx_in_adj_list].second;
                    if (x_int[next_original_tube_idx] > 0) {
                         x_int[next_original_tube_idx]--;
                         decremented = true;
                         break;
                    }
                }
                // If still not decremented, it means all remaining candidates are 0, which is unusual but possible.
            }
        }
    }
    return x_int;
}

// Returns: {indices_to_mix, error_sqrt, mixed_color, discrete_quantities_per_tube}
std::tuple<std::vector<int>, double, Color, std::vector<int>> select_colors_qp_approx(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val) { 

    if (own_colors_list.empty() || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {{}, err_sqrt, default_mixed_color, {}};
    }

    std::vector<double> continuous_weights = calculate_continuous_weights(own_colors_list, target_color_vec);
    if (continuous_weights.empty()) { 
        Color default_mixed_color = {0.0, 0.0, 0.0};
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {{}, err_sqrt, default_mixed_color, std::vector<int>(own_colors_list.size(), 0)};
    }
    
    std::vector<int> discrete_quantities = discretize_weights(continuous_weights, m_val);

    std::vector<int> indices_to_mix_result;
    indices_to_mix_result.reserve(m_val);
    int current_total_discrete_quantity = 0;
    for (size_t i = 0; i < discrete_quantities.size(); ++i) {
        for (int count = 0; count < discrete_quantities[i]; ++count) {
            indices_to_mix_result.push_back(i);
        }
        current_total_discrete_quantity += discrete_quantities[i];
    }
    
    // Ensure total size is m_val due to potential rounding/adjustment issues
    // This primarily handles cases where sum(discrete_quantities) != m_val, which discretize_weights tries to prevent.
    if (current_total_discrete_quantity != m_val) {
        // Re-discretize or more robustly adjust if this happens frequently
        // For now, simple fill/truncate:
        while (indices_to_mix_result.size() < static_cast<size_t>(m_val) && !own_colors_list.empty()) {
            // Add based on original continuous weights if possible, or just default.
            int best_idx_to_add = 0;
            if (!continuous_weights.empty()) {
                best_idx_to_add = std::distance(continuous_weights.begin(), std::max_element(continuous_weights.begin(), continuous_weights.end()));
            }
            indices_to_mix_result.push_back(best_idx_to_add); 
        }
        if (indices_to_mix_result.size() > static_cast<size_t>(m_val)) {
            indices_to_mix_result.resize(m_val); 
        }
    }


    Color mixed_color_result = get_mixed_color_from_indices(indices_to_mix_result, own_colors_list, m_val);
    double error_sqrt_result = calculate_error_sqrt(mixed_color_result, target_color_vec);

    return {indices_to_mix_result, error_sqrt_result, mixed_color_result, discrete_quantities};
}


void combinations_with_replacement_recursive(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    int start_index,
    std::vector<int>& current_combination,
    std::vector<int>& best_indices,
    double& best_err_val_sqrt, 
    Color& best_mixed_color_tuple,
    bool& found_one) {

    if (current_combination.size() == static_cast<size_t>(m_val)) {
        Color mixed_color = get_mixed_color_from_indices(current_combination, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec);
        // found_one should be set before this check if any valid combo is made
        if (!found_one || current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_mixed_color_tuple = mixed_color;
            best_indices = current_combination;
        }
        found_one = true; // Mark that at least one combination was evaluated
        return;
    }
    // Optimization: if remaining elements to pick > remaining available unique choices (even with replacement)
    if (own_colors_list.empty() && m_val > 0) return; // Cannot pick if no colors
    if (static_cast<int>(current_combination.size()) + (own_colors_list.size() - start_index) < static_cast<size_t>(m_val) && (own_colors_list.size() - start_index >0) && m_val > current_combination.size() && m_val > (own_colors_list.size() - start_index) ) {
         // This condition needs to be carefully formulated for combinations_with_replacement.
         // The original `if (static_cast<size_t>(start_index) >= own_colors_list.size()) return;` handles one boundary.
    }
    if (start_index >= static_cast<int>(own_colors_list.size())) return;


    for (size_t i = start_index; i < own_colors_list.size(); ++i) {
        current_combination.push_back(i);
        // For combinations_with_replacement, the next element can also be 'i'
        combinations_with_replacement_recursive(
            own_colors_list, target_color_vec, m_val, i, 
            current_combination, best_indices, best_err_val_sqrt, best_mixed_color_tuple, found_one);
        current_combination.pop_back();
    }
}

std::tuple<std::vector<int>, double, Color> select_colors_exhaustive(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val) {
    double best_err_val_sqrt = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};
    std::vector<int> best_indices_to_drop; // Default empty
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0}; // Or target_color_vec if m_val=0 meaning no paint added
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {best_indices_to_drop, err_sqrt, default_mixed_color};
    }
    
    bool found_one = false;
    std::vector<int> current_combination;
    current_combination.reserve(m_val);

    combinations_with_replacement_recursive(
        own_colors_list, target_color_vec, m_val, 0,
        current_combination, best_indices_to_drop, best_err_val_sqrt, best_mixed_color_tuple, found_one);
            
    if (!found_one && num_own_colors > 0 && m_val > 0) { 
        best_indices_to_drop.assign(m_val, 0); 
        best_mixed_color_tuple = get_mixed_color_from_indices(best_indices_to_drop, own_colors_list, m_val);
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    } else if (!found_one && m_val == 0) { // No paint, error is just distance to target
        best_mixed_color_tuple = {0,0,0};
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    }
    return {best_indices_to_drop, best_err_val_sqrt, best_mixed_color_tuple};
}

std::tuple<std::vector<int>, double, Color> select_colors_rs(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    int num_samples = 100) {
    double best_err_val_sqrt = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};    
    std::vector<int> best_indices_to_drop;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {best_indices_to_drop, err_sqrt, default_mixed_color};
    }
    
    std::uniform_int_distribution<int> dist(0, static_cast<int>(num_own_colors) - 1);
    bool found_one = false;
    std::vector<int> current_indices(m_val);

    for (int s = 0; s < num_samples; ++s) {
        for (int i = 0; i < m_val; ++i) {
            current_indices[i] = dist(rng);
        }
        Color mixed_color = get_mixed_color_from_indices(current_indices, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec);
        
        if (!found_one || current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_indices_to_drop = current_indices; 
            best_mixed_color_tuple = mixed_color;
        }
        found_one = true;
    }
    
    if (!found_one && num_own_colors > 0 && m_val > 0) {
        best_indices_to_drop.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color_from_indices(best_indices_to_drop, own_colors_list, m_val);
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    } else if (!found_one && m_val == 0) {
        best_mixed_color_tuple = {0,0,0};
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    }
    return {best_indices_to_drop, best_err_val_sqrt, best_mixed_color_tuple};
}

struct SAParams {
    double initial_temp;
    double final_temp;
    double cooling_rate;
    int iterations_per_temp;
    int initial_rs_samples;
};

std::tuple<std::vector<int>, double, Color> select_colors_sa_timed(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    const SAParams& sa_params,
    double time_limit_ms,
    std::vector<int> initial_solution_indices = {}) { 

    size_t num_own_colors = own_colors_list.size();
    auto start_time_sa_chrono = std::chrono::steady_clock::now();

    if (m_val == 0) { // No paint to select
        Color default_mixed_color = {0.0, 0.0, 0.0};
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {{}, err_sqrt, default_mixed_color};
    }
    if (num_own_colors == 0) { // No tube colors available
         Color default_mixed_color = {0.0, 0.0, 0.0};
         double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {{}, err_sqrt, default_mixed_color}; // Cannot form a mix
    }


    std::vector<int> current_indices;
    double current_error_sqrt; 
    Color current_mixed_color;
    bool initial_solution_valid = !initial_solution_indices.empty() && initial_solution_indices.size() == static_cast<size_t>(m_val);

    if (initial_solution_valid) {
        current_indices = initial_solution_indices;
        current_mixed_color = get_mixed_color_from_indices(current_indices, own_colors_list, m_val);
        current_error_sqrt = calculate_error_sqrt(current_mixed_color, target_color_vec);
    } else {
        auto rs_result = select_colors_rs(own_colors_list, target_color_vec, m_val, sa_params.initial_rs_samples);
        current_indices = get<0>(rs_result);
        current_error_sqrt = get<1>(rs_result);
        current_mixed_color = get<2>(rs_result);
    }
    
    // Fallback if RS or initial solution didn't produce a valid set of indices
    if (current_indices.empty() && m_val > 0 && num_own_colors > 0) { 
        current_indices.assign(m_val, 0); // Use first color m_val times
        current_mixed_color = get_mixed_color_from_indices(current_indices, own_colors_list, m_val);
        current_error_sqrt = calculate_error_sqrt(current_mixed_color, target_color_vec);
    }
            
    std::vector<int> best_indices = current_indices; 
    double best_error_sqrt = current_error_sqrt;
    Color best_mixed_color_tuple = current_mixed_color;
    
    // If m_val is 0 or no colors, current_indices might be empty. SA loop shouldn't run.
    if (m_val == 0 || num_own_colors == 0 || current_indices.empty()) {
        return {best_indices, best_error_sqrt, best_mixed_color_tuple};
    }

    double current_temp = sa_params.initial_temp;
    std::uniform_real_distribution<double> unif_real_dist(0.0, 1.0);
    std::uniform_int_distribution<int> idx_change_dist(0, m_val - 1); 
    std::uniform_int_distribution<int> color_idx_dist(0, static_cast<int>(num_own_colors) - 1);

    while (current_temp > sa_params.final_temp) {
        auto check_time = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break;

        for (int iter_idx = 0; iter_idx < sa_params.iterations_per_temp; ++iter_idx) {
            check_time = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break; 

            std::vector<int> neighbor_indices = current_indices; 
            int change_pos = idx_change_dist(rng); // Position in the list of m_val indices
            int new_color_idx = color_idx_dist(rng);  // Index of tube color from own_colors_list
            
            if (num_own_colors > 1) { 
                int retries = 0; // Prevent infinite loop if m_val=1 and only one color exists
                while (new_color_idx == neighbor_indices[change_pos] && retries < num_own_colors * 2) {
                    new_color_idx = color_idx_dist(rng);
                    retries++;
                }
            }
            neighbor_indices[change_pos] = new_color_idx;

            Color neighbor_mixed_color = get_mixed_color_from_indices(neighbor_indices, own_colors_list, m_val);
            double neighbor_error_sqrt = calculate_error_sqrt(neighbor_mixed_color, target_color_vec);
            double delta_error = neighbor_error_sqrt - current_error_sqrt;

            if (delta_error < 0) {
                current_indices = neighbor_indices;
                current_error_sqrt = neighbor_error_sqrt;
                current_mixed_color = neighbor_mixed_color;
                if (current_error_sqrt < best_error_sqrt) {
                    best_error_sqrt = current_error_sqrt;
                    best_indices = current_indices; 
                    best_mixed_color_tuple = current_mixed_color;
                }
            } else {
                if (current_temp > 1e-9) { 
                    double acceptance_probability = std::exp(-delta_error / current_temp);
                    if (unif_real_dist(rng) < acceptance_probability) {
                        current_indices = neighbor_indices;
                        current_error_sqrt = neighbor_error_sqrt;
                        current_mixed_color = neighbor_mixed_color;
                    }
                }
            }
        }
        current_temp *= sa_params.cooling_rate;
    }
    return {best_indices, best_error_sqrt, best_mixed_color_tuple};
}
      

struct SlotInfo {
    Color color;
    int remaining_grams;

    SlotInfo() : color({0,0,0}), remaining_grams(0) {}
};


// --- Main Logic ---
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout << std::fixed << std::setprecision(8); 

    auto overall_start_time_chrono = std::chrono::steady_clock::now();
    
    int N_grid_size, K_num_colors, H_num_targets, T_total_ops_limit;
    double D_cost_factor;
    
    std::cin >> N_grid_size >> K_num_colors >> H_num_targets >> T_total_ops_limit >> D_cost_factor;
    cerr<< "N_grid_size: " << N_grid_size << ", K_num_colors: " << K_num_colors 
         << ", H_num_targets: " << H_num_targets << ", T_total_ops_limit: " << T_total_ops_limit 
         << ", D_cost_factor: " << D_cost_factor << std::endl;
    std::vector<Color> own_colors(K_num_colors);
    for (int i = 0; i < K_num_colors; ++i) {
        std::cin >> own_colors[i].r >> own_colors[i].g >> own_colors[i].b;
    }
        
    std::vector<Color> targets(H_num_targets);
    for (int i = 0; i < H_num_targets; ++i) {
        std::cin >> targets[i].r >> targets[i].g >> targets[i].b;
    }
        
    for (int i = 0; i < N_grid_size; ++i) {
        for (int j = 0; j < N_grid_size - 1; ++j) {
            std::cout << (j > 0 ? " " : "") << "1"; 
        }
        std::cout << std::endl;
    }
    const int HORIZONTAL_DIVIDER_ROW_INDEX = N_grid_size / 2 -1; 
    for (int r_h_idx = 0; r_h_idx < N_grid_size - 1; ++r_h_idx) {
        for (int j = 0; j < N_grid_size; ++j) {
            std::cout << (j > 0 ? " " : "") << (r_h_idx == HORIZONTAL_DIVIDER_ROW_INDEX ? "1" : "0");
        }
        std::cout << std::endl;
    }
    
    const int WELL_CAPACITY = N_grid_size; // Assuming each of the N_grid_size cells in a row can hold 1g, and a slot uses one full row.
                                        // If a slot is one cell, WELL_CAPACITY = 1.
                                        // The problem says "kマスからなるウェルには最大kグラム".
                                        // My interpretation: one slot = one row (N cells) = capacity N.
                                        // If N=20, capacity is 20. The previous hardcoded 10 was likely based on a different assumption.
                                        // The problem also stated "N=20固定" implying well capacity might be related.
                                        // Sticking to previous value 10 based on "Q=10 or smaller" discussions.
    const int ACTUAL_WELL_CAPACITY = 10; 

    const int NUM_SLOTS_PER_HALF = N_grid_size; 
    const int NUM_TOTAL_SLOTS = 2 * NUM_SLOTS_PER_HALF;

    std::vector<SlotInfo> slots(NUM_TOTAL_SLOTS);
    int current_slot_idx_preference = 0; 
    std::set<int> m_cands_set; 
    for (int m = 1; m <= ACTUAL_WELL_CAPACITY; ++m) {
        if(2*m*1000<=T_total_ops_limit) m_cands_set.insert(m);
    }
    // Add some common/middle values if not already present
    // if (ACTUAL_WELL_CAPACITY >= 4) m_cands_set.insert(std::max(1, ACTUAL_WELL_CAPACITY / 2));
    // if (ACTUAL_WELL_CAPACITY >= 6) m_cands_set.insert(std::max(1, ACTUAL_WELL_CAPACITY / 3 * 2));


    std::vector<int> m_cands(m_cands_set.begin(), m_cands_set.end());
    std::sort(m_cands.begin(), m_cands.end());
    
    const double TOTAL_TIME_BUDGET_S = (K_num_colors > 40 && H_num_targets > 800 && N_grid_size > 15) ? 2.85 : 2.95;
    int remaining_ops = T_total_ops_limit;
    std::vector<std::string> operations_log; 
    int num_blank_slots = 40;

    for (int i_target = 0; i_target < H_num_targets; ++i_target) {
        if (remaining_ops <= 0 && i_target < H_num_targets) { 
            int fallback_row = ( ( (current_slot_idx_preference + i_target) % NUM_TOTAL_SLOTS) < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int fallback_col = ( (current_slot_idx_preference + i_target) % NUM_TOTAL_SLOTS) % NUM_SLOTS_PER_HALF;
            operations_log.push_back("2 " + to_string(fallback_row) + " " + to_string(fallback_col));
            continue;
        }
    
        double progress = static_cast<double>(i_target + 1) / H_num_targets;
        double discount_factor = progress * 0.10; // Cost aversion increases more significantly
        discount_factor = 180.0/D_cost_factor;
        // discount_factor = .05;
        
        auto current_time_chrono = std::chrono::steady_clock::now();
        double time_spent_s = std::chrono::duration<double>(current_time_chrono - overall_start_time_chrono).count();
        double remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s;
        double avg_time_per_remaining_target_ms = (H_num_targets - i_target > 0) ? (remaining_time_budget_s / (H_num_targets - i_target) * 1000.0) : 2.0;
        
        double method_time_limit_ms = std::max(0.05, std::min(avg_time_per_remaining_target_ms * 0.9, 15.0)); 
        if (remaining_time_budget_s < 0.05 * (H_num_targets - i_target) && i_target < H_num_targets -1 ) method_time_limit_ms = 0.05;


        Color current_target_color_obj = targets[i_target];
        
        double min_error_sqrt = std::numeric_limits<double>::infinity();
       
        vector<pair<double,int>> mix_color_candidates; // For debugging or future use

        Color mixed_target_color_obj;
        int mix_num=min(max(min((int)D_cost_factor/300,5),1), H_num_targets - i_target);
        FOR(j,i_target,min(H_num_targets,i_target +100*mix_num)){
            auto error_sqrt_j= calculate_error_sqrt(current_target_color_obj, targets[j]);
            mix_color_candidates.push_back({error_sqrt_j, j});
        }
        sort(mix_color_candidates.begin(), mix_color_candidates.end());
        rep(j,mix_num){
            mixed_target_color_obj+= targets[mix_color_candidates[j].second];
        }
        mixed_target_color_obj = mixed_target_color_obj / static_cast<double>(mix_num);

        int best_reuse_slot_idx = -1;
        double min_reuse_error_sqrt = std::numeric_limits<double>::infinity();
        
        for (int slot_idx = 0; slot_idx < NUM_TOTAL_SLOTS; ++slot_idx) {
            if (slots[slot_idx].remaining_grams > 0) {
                double err_sqrt = calculate_error_sqrt(slots[slot_idx].color, current_target_color_obj);
                if (err_sqrt < min_reuse_error_sqrt) {
                    min_reuse_error_sqrt = err_sqrt;
                    best_reuse_slot_idx = slot_idx;
                }
            }
        }
        bool reuse_is_possible = (best_reuse_slot_idx != -1);
        // Score for reuse: only error cost, no paint cost. Ops cost is 1 (for "use")
        double reuse_score = reuse_is_possible ? (min_reuse_error_sqrt * 10000.0) : std::numeric_limits<double>::infinity();

        double best_1g_score = std::numeric_limits<double>::infinity();
        double best_1g_error_sqrt = std::numeric_limits<double>::infinity();
        std::vector<int> best_1g_indices; 
        Color best_1g_mixed_color;
        int best_1g_m_val = 0;

        if (K_num_colors > 0) {
            set<int> m_cands;
            rep(mi,11){
                if(2*mi*(1000-i_target) <=remaining_ops) {
                    m_cands.insert(mi);
                }
            }
            if(num_blank_slots > 0) {
                rep(mi,11){
                    m_cands.insert(mi);
                }
            }
            // m_cands.erase(std::unique(m_cands.begin(), m_cands.end()), m_cands.end());
            for (int m_trial : m_cands) {
                if (m_trial == 0 || m_trial > ACTUAL_WELL_CAPACITY) continue;
                if (m_trial + 1 > remaining_ops) continue; 

                std::vector<int> current_m_indices_candidate;
                double current_m_err_sqrt_candidate = std::numeric_limits<double>::infinity();
                Color current_m_mixed_c_candidate;
                bool candidate_found = false;

                // Try QP Approx first as it's deterministic and potentially fast
                auto qp_res_tuple = select_colors_qp_approx(own_colors, mixed_target_color_obj, m_trial);
                if (!get<0>(qp_res_tuple).empty() || m_trial == 0) { // Check if QP returned a valid (possibly empty for m_trial=0) set of indices
                    current_m_indices_candidate = get<0>(qp_res_tuple);
                    current_m_err_sqrt_candidate = get<1>(qp_res_tuple);
                    current_m_mixed_c_candidate = get<2>(qp_res_tuple);
                    candidate_found = true;
                }

                // Try Exhaustive if applicable and QP wasn't good enough or failed
                // Heuristic for complexity: K^m_trial approx. For C(K+m-1, m).
                long long combinations_approx = 1;
                if (K_num_colors > 0 && m_trial > 0) {
                    for(int i=0; i<m_trial; ++i) {
                        if (__builtin_mul_overflow(combinations_approx, (K_num_colors+m_trial-1-i), &combinations_approx)) {combinations_approx = -1; break;}
                        if (combinations_approx < 0) break; // Overflow
                        combinations_approx /= (i+1);
                         if (combinations_approx < 0 || combinations_approx > 30000) {combinations_approx = -1; break;} // Cap
                    }
                } else if (m_trial == 0) combinations_approx = 1;


                if (combinations_approx != -1 && combinations_approx <= 30000 ) { 
                    auto exhaustive_res = select_colors_exhaustive(own_colors, mixed_target_color_obj, m_trial);
                    if (!candidate_found || get<1>(exhaustive_res) < current_m_err_sqrt_candidate) {
                        current_m_indices_candidate = get<0>(exhaustive_res);
                        current_m_err_sqrt_candidate = get<1>(exhaustive_res);
                        current_m_mixed_c_candidate = get<2>(exhaustive_res);
                        candidate_found = true;
                    }
                }
                
                // Try SA, possibly seeded with QP/Exhaustive result, if still not good or not run
                // SA should run if others weren't good enough or to refine.
                SAParams sa_p = {1.0, 1e-4, 0.97, std::max(10, K_num_colors / (m_trial+1) + 5), std::max(20, K_num_colors*2/(m_trial+1) + 10)};
                std::vector<int> sa_initial_seed = candidate_found ? current_m_indices_candidate : std::vector<int>();
                
                auto sa_res = select_colors_sa_timed(own_colors, mixed_target_color_obj, m_trial, sa_p, method_time_limit_ms, sa_initial_seed);
                if (!candidate_found || get<1>(sa_res) < current_m_err_sqrt_candidate) {
                     if (!get<0>(sa_res).empty() || m_trial == 0) { // Ensure SA returned something valid
                        current_m_indices_candidate = get<0>(sa_res);
                        current_m_err_sqrt_candidate = get<1>(sa_res);
                        current_m_mixed_c_candidate = get<2>(sa_res);
                        candidate_found = true;
                     }
                }
                
                if (!candidate_found && m_trial > 0 && !own_colors.empty()) { // Absolute fallback for this m_trial
                    current_m_indices_candidate.assign(m_trial, 0);
                    current_m_mixed_c_candidate = get_mixed_color_from_indices(current_m_indices_candidate, own_colors, m_trial);
                    current_m_err_sqrt_candidate = calculate_error_sqrt(current_m_mixed_c_candidate, current_target_color_obj);
                    candidate_found = true;
                } else if (!candidate_found) {
                    continue; // No valid mix found for this m_trial
                }

                // Score for new paint: error cost + D_cost_factor * (tubes_used - 1 if some is leftover)
                // The problem formula is D * (total_used_from_tubes - H_num_targets)
                // Here, for one target, effective cost is D * (m_trial). If we make m_trial grams and use 1, m_trial-1 is wasted.
                // The provided formula `D_cost_factor * (m_trial -1)` was for when 1g is used.
                // If we consider the full game score, each tube pull costs D.
                // So, for m_trial tube pulls, cost is D_cost_factor * m_trial.
                double score_candidate = current_m_err_sqrt_candidate * 10000.0 + discount_factor * D_cost_factor * (m_trial-1); 
                if (score_candidate < best_1g_score) {
                    best_1g_score = score_candidate;
                    best_1g_indices = current_m_indices_candidate;
                    best_1g_mixed_color = current_m_mixed_c_candidate;
                    best_1g_error_sqrt = current_m_err_sqrt_candidate;
                    best_1g_m_val = best_1g_indices.size(); // m_val is the size of indices, not the color
                }
            }
        }
        // cerr << "Target " << i_target + 1 << ": Reuse score: " << reuse_score 
        //      << ", Best 1g score: " << best_1g_score 
        //      << ", Best 1g indices: " << best_1g_indices.size() 
        //      << ", Best 1g color: (" << best_1g_mixed_color.r << ", " 
        //      << best_1g_mixed_color.g << ", " 
        //      << best_1g_mixed_color.b << ")" 
        //      << std::endl;
        
        for (int slot_idx = 0; slot_idx < NUM_TOTAL_SLOTS; ++slot_idx) {
            if (slots[slot_idx].remaining_grams == 0) num_blank_slots++;
        }

        bool chose_reuse = false;
        bool chose_new_1g = false;
        
        double reuse_threshold_factor = 1.0; 
        // if (num_blank_slots>0) reuse_threshold_factor = .85; // If no blank slots, prefer reuse
        // If many blank slots, be less inclined to reuse (encourage filling slots), unless reuse is much better.
        // if (num_blank_slots > NUM_TOTAL_SLOTS / 2) reuse_threshold_factor = 0.9; 
        // if (num_blank_slots < NUM_TOTAL_SLOTS / 4) reuse_threshold_factor = 1.1; // Prefer reuse if slots are full

        if (K_num_colors == 0 && reuse_is_possible) { // No tubes, must reuse if possible
             if (1 <= remaining_ops) chose_reuse = true;
        } else if (reuse_is_possible && (best_1g_indices.empty() || (reuse_score < best_1g_score * reuse_threshold_factor))) { // Prefer reuse if it's better or if no 1g options
             if (1 <= remaining_ops) { 
                chose_reuse = true;
             }
        }
        
        if (!chose_reuse && !best_1g_indices.empty() && K_num_colors > 0) { // K_num_colors check for creating new
            int ops_for_1g = best_1g_m_val + 1; 
            if (ops_for_1g <= remaining_ops) {
                chose_new_1g = true;
            }
        }
        
        if (chose_reuse) {
            int actual_row = (best_reuse_slot_idx < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col = best_reuse_slot_idx % NUM_SLOTS_PER_HALF;
            operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));
            cerr <<"reuse score: " << reuse_score << " error sqrt: " << min_reuse_error_sqrt 
                //  << ", best reuse slot idx: " << best_reuse_slot_idx 
                //  << ", color: (" << slots[best_reuse_slot_idx].color.r << ", " 
                //  << slots[best_reuse_slot_idx].color.g << ", " 
                //  << slots[best_reuse_slot_idx].color.b << ")" 
                 
                 << std::endl;
            slots[best_reuse_slot_idx].remaining_grams--;
            // Batch paint logic removed for simplicity as it was not fully active
            remaining_ops--;
        }
        else if (chose_new_1g) { 
            int slot_to_use = -1;
            int best_val_for_slot = -1; 
            
            for(int offset = 0; offset < NUM_TOTAL_SLOTS; ++offset) {
                int temp_slot = (current_slot_idx_preference + offset) % NUM_TOTAL_SLOTS;
                int current_val = ACTUAL_WELL_CAPACITY-slots[temp_slot].remaining_grams; // 0 for batch (lowest priority), 1 for non-batch occupied, 2 for empty
                
                if (current_val > best_val_for_slot) {
                    best_val_for_slot = current_val;
                    slot_to_use = temp_slot;
                }
            }
            if(slot_to_use == -1) slot_to_use = current_slot_idx_preference; // Fallback

            int actual_row = (slot_to_use < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col = slot_to_use % NUM_SLOTS_PER_HALF;

            int discards_needed = slots[slot_to_use].remaining_grams;
            if (discards_needed + best_1g_m_val + 1 > remaining_ops) { 
                 chose_new_1g = false; // Not enough ops, will go to fallback
            } else {
                for(int d=0; d<discards_needed; ++d) operations_log.push_back("3 " + to_string(actual_row) + " " + to_string(actual_col));
                remaining_ops -= discards_needed;

                for(int tube_original_idx : best_1g_indices) operations_log.push_back("1 " + to_string(actual_row) + " " + to_string(actual_col) + " " + to_string(tube_original_idx));
                operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));

                slots[slot_to_use].color = best_1g_mixed_color;
                slots[slot_to_use].remaining_grams = best_1g_m_val - 1; 

                current_slot_idx_preference = (slot_to_use + 1) % NUM_TOTAL_SLOTS;
                remaining_ops -= (best_1g_m_val + 1);
            }
            cerr << "New1g score: " << best_1g_score 
            << " error sqrt: " << best_1g_error_sqrt;
            cerr << ", m_val: " << best_1g_m_val ;
            cerr <<endl;
            // cerr<< ", indices: ";
            // for (int idx : best_1g_indices) {
            //     cerr << idx << " ";
            // }
            // cerr << ", color: (" << best_1g_mixed_color.r << ", " 
            //      << best_1g_mixed_color.g << ", " 
            //      << best_1g_mixed_color.b << ")" 
            //      << std::endl;
                             

        }

     
        // cerr<<"remaining_ops after fallback: " << remaining_ops << std::endl;
        end_target_iteration:;
    }

    for(const auto& op_str : operations_log) {
        std::cout << op_str << std::endl;
    }
    
    return 0;
}