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
#include <numeric> // For std::iota (not strictly used here but good for similar tasks)
#include <set>     // For std::set to get unique sorted m_cands

using namespace std;
#define rep (i, n) for (int i = 0; i < (n); ++i)
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
// --- Global Random Number Generator ---
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

// --- Helper Structs and Functions ---
struct Color {
    double r, g, b;

    // Default constructor
    Color(double r_ = 0.0, double g_ = 0.0, double b_ = 0.0) : r(r_), g(g_), b(b_) {}
};

Color get_mixed_color(const std::vector<int>& indices_to_mix,
                      const std::vector<Color>& own_colors_list_ref,
                      int m_val_eff) {
    if (indices_to_mix.empty() || m_val_eff == 0) {
        return {0.0, 0.0, 0.0};
    }
    double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
    for (int idx : indices_to_mix) {
        // Basic bounds check, though ideally indices should always be valid
        if (idx < 0 || idx >= own_colors_list_ref.size()) {
             // This case should ideally not happen if logic is correct
             // std::cerr << "Warning: Invalid index " << idx << " in get_mixed_color." << std::endl;
             // Depending on problem rules, either skip, error, or use a default.
             // For now, we assume valid indices as per Python's direct access.
        }
        const auto& color = own_colors_list_ref[idx];
        sum_r += color.r;
        sum_g += color.g;
        sum_b += color.b;
    }
    return {sum_r / m_val_eff, sum_g / m_val_eff, sum_b / m_val_eff};
}

double calculate_error(const Color& mixed_color_tuple, const Color& target_color_tuple) {
    double err_sq = 0.0;
    double dr = mixed_color_tuple.r - target_color_tuple.r;
    double dg = mixed_color_tuple.g - target_color_tuple.g;
    double db = mixed_color_tuple.b - target_color_tuple.b;
    err_sq = dr * dr + dg * dg + db * db;
    return std::sqrt(err_sq);
}

// --- Color Selection Strategies ---

// Helper for select_colors_exhaustive
void combinations_with_replacement_recursive(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    int start_index,
    std::vector<int>& current_combination,
    std::vector<int>& best_indices,
    double& best_err_val,
    Color& best_mixed_color_tuple,
    bool& found_one) {

    if (current_combination.size() == static_cast<size_t>(m_val)) {
        Color mixed_color = get_mixed_color(current_combination, own_colors_list, m_val);
        double current_err = calculate_error(mixed_color, target_color_vec);
        found_one = true;
        if (current_err < best_err_val) {
            best_err_val = current_err;
            best_mixed_color_tuple = mixed_color;
            best_indices = current_combination;
        }
        return;
    }

    if (static_cast<size_t>(start_index) >= own_colors_list.size()) {
         // This condition might be hit if m_val is larger than can be formed,
         // or if own_colors_list is empty and start_index is 0.
         // The outer function should handle empty own_colors_list.
        return;
    }
    // To prevent going too deep if m_val is large relative to stack size,
    // ensure remaining spots can be filled.
    if (own_colors_list.empty() && m_val > current_combination.size()) return;


    for (size_t i = start_index; i < own_colors_list.size(); ++i) {
        current_combination.push_back(i);
        combinations_with_replacement_recursive(
            own_colors_list, target_color_vec, m_val,
            i, // For replacement, start from i, not i+1
            current_combination, best_indices, best_err_val, best_mixed_color_tuple, found_one
        );
        current_combination.pop_back();
    }
}


std::tuple<std::vector<int>, double, Color> select_colors_exhaustive(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val) {

    double best_err_val = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};
    std::vector<int> best_indices_to_drop;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        if (m_val > 0 && num_own_colors > 0) { // Should not happen if num_own_colors == 0
             // This path means m_val == 0.
             // If m_val == 0, indices are empty. Mixed color is (0,0,0)
        } else if (m_val == 0) {
            // default_mixed_color is already (0,0,0)
        } else { // num_own_colors == 0 and m_val > 0
            // Cannot pick any colors. default_mixed_color is (0,0,0)
        }
        // The Python version has own_colors_list[0] if m_val > 0 and num_own_colors > 0.
        // This case is m_val == 0 OR num_own_colors == 0.
        // If m_val == 0: result is empty list, (0,0,0) color. Error is against target.
        // If num_own_colors == 0 (and m_val > 0): result is empty list, (0,0,0) color. Error is against target.
        std::vector<int> default_indices; // Empty for these cases
        double err = calculate_error(default_mixed_color, target_color_vec);
        return {default_indices, err, default_mixed_color};
    }
    
    bool found_one = false;
    std::vector<int> current_combination;
    current_combination.reserve(m_val);

    combinations_with_replacement_recursive(
        own_colors_list, target_color_vec, m_val, 0,
        current_combination, best_indices_to_drop, best_err_val, best_mixed_color_tuple, found_one
    );
            
    if (!found_one && num_own_colors > 0 && m_val > 0) { 
        best_indices_to_drop.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val);
        best_err_val = calculate_error(best_mixed_color_tuple, target_color_vec);
    } else if (!found_one) {
        // This case implies m_val > 0 but own_colors_list was empty, handled by initial if.
        // Or, m_val was 0, also handled.
        // If it gets here, it means combinations_with_replacement_recursive didn't find anything,
        // which is unusual if num_own_colors > 0 and m_val > 0.
        // The Python code's 'pass' suggests no change to defaults if nothing found.
        // But best_err_val is inf, best_indices_to_drop is empty, best_mixed_color_tuple is (0,0,0).
        // If m_val > 0 and num_own_colors > 0, the [0]*m_val fallback is critical.
    }
    return {best_indices_to_drop, best_err_val, best_mixed_color_tuple};
}

std::tuple<std::vector<int>, double, Color> select_colors_rs(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    int num_samples = 100) {
    
    double best_err_val = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};    
    std::vector<int> best_indices_to_drop;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
         std::vector<int> default_indices;
        // Python logic: if m_val > 0 and num_own_colors > 0: default_mixed_color = own_colors_list[0]
        // This specific condition isn't hit by the outer if.
        // If m_val == 0: default_indices is empty, color (0,0,0)
        // If num_own_colors == 0 (and m_val > 0): default_indices is empty, color (0,0,0)
        double err = calculate_error(default_mixed_color, target_color_vec);
        return {default_indices, err, default_mixed_color};
    }
    
    std::uniform_int_distribution<int> dist(0, num_own_colors - 1);
    bool found_one = false;
    std::vector<int> current_indices(m_val);

    for (int s = 0; s < num_samples; ++s) {
        for (int i = 0; i < m_val; ++i) {
            current_indices[i] = dist(rng);
        }
        Color mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
        double current_err = calculate_error(mixed_color, target_color_vec);
        found_one = true;
        if (current_err < best_err_val) {
            best_err_val = current_err;
            best_indices_to_drop = current_indices; 
            best_mixed_color_tuple = mixed_color;
        }
    }
    
    if (!found_one && num_own_colors > 0 && m_val > 0) {
        best_indices_to_drop.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val);
        best_err_val = calculate_error(best_mixed_color_tuple, target_color_vec);
    } // else if !found_one: (similar to exhaustive, should be covered by initial checks or the [0]*m_val fallback)
    
    return {best_indices_to_drop, best_err_val, best_mixed_color_tuple};
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
    double time_limit_ms) {

    size_t num_own_colors = own_colors_list.size();
    auto start_time_sa_chrono = std::chrono::steady_clock::now();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        std::vector<int> default_indices;
        // if m_val > 0 and num_own_colors > 0: default_mixed_color = own_colors_list[0]
        // This part is tricky. If m_val == 0, indices empty, color (0,0,0).
        // If num_own_colors == 0 (and m_val > 0), indices empty, color (0,0,0).
        // The Python `([0] * m_val if num_own_colors > 0 and m_val > 0 else [])`
        // suggests if num_own_colors == 0 OR m_val == 0, the indices are empty.
        // The default_mixed_color is (0,0,0) in these cases.
        double err = calculate_error(default_mixed_color, target_color_vec);
        return {default_indices, err, default_mixed_color};
    }

    std::vector<int> current_indices;
    double current_error;
    Color current_mixed_color;

    std::tie(current_indices, current_error, current_mixed_color) = select_colors_rs(
        own_colors_list, target_color_vec, m_val, sa_params.initial_rs_samples
    );
    // If RS returns empty indices (e.g. m_val=0), current_indices will be empty.
    // If m_val > 0 and num_own_colors > 0, RS should return non-empty indices.
    // The Python SA code has a check: `if not current_indices: if m_val > 0: current_indices = [0] * m_val`.
    // This implies RS might return empty. If current_indices is empty and m_val > 0, initialize it.
    if (current_indices.empty() && m_val > 0 && num_own_colors > 0) {
        current_indices.assign(m_val, 0);
        current_mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
        current_error = calculate_error(current_mixed_color, target_color_vec);
    }
            
    std::vector<int> best_indices = current_indices; 
    double best_error = current_error;
    Color best_mixed_color_tuple = current_mixed_color;
    
    double current_temp = sa_params.initial_temp;
    std::uniform_real_distribution<double> unif_real_dist(0.0, 1.0);
    std::uniform_int_distribution<int> idx_change_dist(0, m_val - 1); // If m_val=0, this is problematic, but handled.
    std::uniform_int_distribution<int> color_idx_dist(0, num_own_colors - 1);


    while (current_temp > sa_params.final_temp) {
        auto check_time = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break;

        for (int iter_idx = 0; iter_idx < sa_params.iterations_per_temp; ++iter_idx) {
            check_time = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break; 

            if (current_indices.empty()) { // Should only happen if m_val was 0 initially
                if (m_val > 0 && num_own_colors > 0) { // Recover if somehow became empty
                    current_indices.assign(m_val, 0);
                    current_mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
                    current_error = calculate_error(current_mixed_color, target_color_vec);
                    if (current_error < best_error) { // Unlikely path, but for completeness
                         best_error = current_error;
                         best_indices = current_indices;
                         best_mixed_color_tuple = current_mixed_color;
                    }
                } else {
                    break; // Cannot proceed if m_val is 0 or no colors
                }
            }
            if (current_indices.empty()) break; // Double check after recovery attempt

            std::vector<int> neighbor_indices = current_indices; 
            
            int idx_to_change = idx_change_dist(rng); // Requires m_val > 0
            int new_color_idx = color_idx_dist(rng);  // Requires num_own_colors > 0
            if (num_own_colors > 1) { // Ensure change if possible
                while (new_color_idx == neighbor_indices[idx_to_change]) {
                    new_color_idx = color_idx_dist(rng);
                }
            }
            neighbor_indices[idx_to_change] = new_color_idx;

            Color neighbor_mixed_color = get_mixed_color(neighbor_indices, own_colors_list, m_val);
            double neighbor_error = calculate_error(neighbor_mixed_color, target_color_vec);
            
            double delta_error = neighbor_error - current_error;

            if (delta_error < 0) {
                current_indices = neighbor_indices;
                current_error = neighbor_error;
                current_mixed_color = neighbor_mixed_color;
                if (current_error < best_error) {
                    best_error = current_error;
                    best_indices = current_indices; // Python uses current_indices[:]
                    best_mixed_color_tuple = current_mixed_color;
                }
            } else {
                if (current_temp > 1e-9) { // Avoid division by zero or issues with very small temp
                    double acceptance_probability = std::exp(-delta_error / current_temp);
                    if (unif_real_dist(rng) < acceptance_probability) {
                        current_indices = neighbor_indices;
                        current_error = neighbor_error;
                        current_mixed_color = neighbor_mixed_color;
                    }
                }
            }
        }
        if (current_indices.empty() && m_val > 0) break; // Safety break if state becomes invalid
        current_temp *= sa_params.cooling_rate;
    }
            
    return {best_indices, best_error, best_mixed_color_tuple};
}


std::tuple<std::vector<int>, double, Color> select_colors_gd(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val_max_count, 
    double D_cost_threshold 
) {
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val_max_count == 0) {
        Color empty_mix_color = {0.0, 0.0, 0.0};
        double err = calculate_error(empty_mix_color, target_color_vec);
        return {{}, err, empty_mix_color}; // Return empty vector of indices
    }

    std::vector<int> current_selected_indices;
    double error_before_this_iteration = calculate_error({0.0, 0.0, 0.0}, target_color_vec);
    
    // std::cerr << "GD Start: m_val_max=" << m_val_max_count << " D_cost_thresh=" << D_cost_threshold << std::endl;
    // std::cerr << "GD Initial error_before: " << error_before_this_iteration << std::endl;


    for (int count = 0; count < m_val_max_count; ++count) {
        int best_candidate_idx_for_this_step = -1;
        double min_error_found_this_step = error_before_this_iteration; 
        // Color best_mixed_color_this_step; // Not strictly needed to store here

        // std::cerr << "GD Iteration " << count << ", error_before_this_iter=" << error_before_this_iteration << std::endl;

        for (size_t i_candidate = 0; i_candidate < num_own_colors; ++i_candidate) {
            std::vector<int> potential_indices = current_selected_indices;
            potential_indices.push_back(i_candidate);
            
            Color potential_mixed_color = get_mixed_color(potential_indices, own_colors_list, potential_indices.size());
            double potential_error = calculate_error(potential_mixed_color, target_color_vec);
            
            // std::cerr << "  Candidate " << i_candidate << ", potential_error=" << potential_error << ", current_min_err_this_step=" << min_error_found_this_step << std::endl;

            if (potential_error < min_error_found_this_step) {
                min_error_found_this_step = potential_error;
                best_candidate_idx_for_this_step = i_candidate;
                // best_mixed_color_this_step = potential_mixed_color;
            }
        }
        
        // std::cerr << "GD Iteration " << count << ", best_cand_idx=" << best_candidate_idx_for_this_step << ", min_err_found_this_step=" << min_error_found_this_step << std::endl;


        if (best_candidate_idx_for_this_step != -1) {
            double improvement_delta = error_before_this_iteration - min_error_found_this_step;
             // std::cerr << "  Improvement delta: " << improvement_delta << " (scaled: " << improvement_delta * 10000.0 << ")" << std::endl;
            if ((improvement_delta * 10000.0) > D_cost_threshold) { // Python: if delta *10000 > D_cost_factor_unused
                current_selected_indices.push_back(best_candidate_idx_for_this_step);
                error_before_this_iteration = min_error_found_this_step;
                // std::cerr << "  Added index " << best_candidate_idx_for_this_step << ". New error_before_next_iter=" << error_before_this_iteration << std::endl;
            } else {
                // std::cerr << "  Improvement not good enough or negative. Breaking." << std::endl;
                break; 
            }
        } else {
            // std::cerr << "  No candidate improved error. Breaking." << std::endl;
            break; 
        }
    }

    Color final_mixed_color = {0.0, 0.0, 0.0};
    double final_error = calculate_error(final_mixed_color, target_color_vec); // Error if list is empty

    if (!current_selected_indices.empty()) {
        final_mixed_color = get_mixed_color(current_selected_indices, own_colors_list, current_selected_indices.size());
        final_error = calculate_error(final_mixed_color, target_color_vec);
    }
    // std::cerr << "GD End: selected_indices_count=" << current_selected_indices.size() << ", final_error=" << final_error << std::endl;
    
    return {current_selected_indices, final_error, final_mixed_color};
}


// --- Main Logic ---
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout << std::fixed << std::setprecision(8); // For color values if ever printed

    auto overall_start_time_chrono = std::chrono::steady_clock::now();
    
    int N_grid_size, K_num_colors, H_num_targets, T_total_ops_limit;
    double D_cost_factor;
    
    std::cin >> N_grid_size >> K_num_colors >> H_num_targets >> T_total_ops_limit >> D_cost_factor;
    
    std::vector<Color> own_colors(K_num_colors);
    for (int i = 0; i < K_num_colors; ++i) {
        std::cin >> own_colors[i].r >> own_colors[i].g >> own_colors[i].b;
    }
        
    std::vector<Color> targets(H_num_targets);
    for (int i = 0; i < H_num_targets; ++i) {
        std::cin >> targets[i].r >> targets[i].g >> targets[i].b;
    }
        
    // --- 初期仕切り出力 ---
    for (int i = 0; i < N_grid_size; ++i) {
        for (int j = 0; j < N_grid_size - 1; ++j) {
            std::cout << (j > 0 ? " " : "") << "1";
        }
        std::cout << std::endl;
    }
    const int HORIZONTAL_DIVIDER_ROW_INDEX = 9; // 0-indexed
    for (int r_h_idx = 0; r_h_idx < N_grid_size - 1; ++r_h_idx) {
        for (int j = 0; j < N_grid_size; ++j) {
            std::cout << (j > 0 ? " " : "") << (r_h_idx == HORIZONTAL_DIVIDER_ROW_INDEX ? "1" : "0");
        }
        std::cout << std::endl;
    }
    
    // --- 状態管理 ---
    const int NUM_TOTAL_SLOTS = 2 * N_grid_size;
    std::vector<int> remaining_paint_in_slot(NUM_TOTAL_SLOTS, 0);
    std::vector<Color> current_colors_in_slot(NUM_TOTAL_SLOTS, {0.0, 0.0, 0.0});
    int current_slot_idx_to_fill = 0; 
    
    std::set<int> m_cands_set; // Use set for automatic sorting and uniqueness
    int max_m_val_heuristic = 0; // Renamed from max_m_val to avoid confusion
    for (int m = 1; m <= 10; ++m) {
        // The condition 2*m*1000 <= T_total_ops_limit is unusual.
        // If T_total_ops_limit is ~20000, then 2*m <= 20, so m <= 10.
        // This seems to be a heuristic from the Python code.
        if (2LL * m * 1000 <= T_total_ops_limit) { 
            max_m_val_heuristic = m;
            m_cands_set.insert(m);
        }
    }
    // Python also adds max_m_val // i for i=1,2. Max_m_val here is max_m_val_heuristic.
    // if (max_m_val_heuristic > 0) {
    //     m_cands_set.insert(max_m_val_heuristic);
    //     m_cands_set.insert(std::max(max_m_val_heuristic / 2, 1));
    //     m_cands_set.insert(std::max(max_m_val_heuristic / 3, 1)); // Python code was i=1,2
    // }
    // The python code is actually:
    // for i in range(1,3): m_cands.append(max(max_m_val // i,1))
    // This implies the original `max_m_val` (which is `max_m_val_heuristic` here) and then `max_m_val // 2`.
    if (max_m_val_heuristic > 0) { // The original m_cands already contains max_m_val_heuristic
         m_cands_set.insert(std::max(max_m_val_heuristic / 2, 1));
    }


    if (m_cands_set.empty() && K_num_colors > 0) { // Ensure at least m=1 if possible and colors exist
        // A basic check for m=1: needs 1 paint op, 1 use op. Total 2 ops.
        // Discard ops are separate.
        if (1 + 1 <= T_total_ops_limit) { // Cost for m=1: 1 paint, 1 use
             m_cands_set.insert(1);
        }
    }
    std::vector<int> m_cands(m_cands_set.begin(), m_cands_set.end());


    // --- メインループ ---
    const double TOTAL_TIME_BUDGET_S = 2.85; // Python has 2.85
    int remaining_ops = T_total_ops_limit;

    for (int i_target = 0; i_target < H_num_targets; ++i_target) {
        auto current_time_chrono = std::chrono::steady_clock::now();
        double time_spent_s = std::chrono::duration<double>(current_time_chrono - overall_start_time_chrono).count();
        double remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s;
        
        if (remaining_ops <= 0) {
            int best_reuse_slot_idx = -1;
            double min_reuse_error = std::numeric_limits<double>::infinity();
            for (int slot_idx = 0; slot_idx < NUM_TOTAL_SLOTS; ++slot_idx) {
                if (remaining_paint_in_slot[slot_idx] > 0) {
                    double err = calculate_error(current_colors_in_slot[slot_idx], targets[i_target]);
                    if (err < min_reuse_error) {
                        min_reuse_error = err;
                        best_reuse_slot_idx = slot_idx;
                    }
                }
            }
            if (best_reuse_slot_idx != -1) {
                int actual_row_ops = (best_reuse_slot_idx < N_grid_size) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                int actual_col_ops = best_reuse_slot_idx % N_grid_size;
                std::cout << "2 " << actual_row_ops << " " << actual_col_ops << std::endl;
                remaining_paint_in_slot[best_reuse_slot_idx]--;
                // remaining_ops--; // Python code has this, but if remaining_ops is already <=0, this is just for accounting
            } else {
                 // No paint left and no ops left: output a dummy command or handle as per problem spec if this is an issue
                 // For now, assume problem allows not fulfilling a target if ops run out.
                 // Or, if a dummy command is needed (e.g., use slot 0,0 even if empty/wrong)
                 std::cout << "2 0 0" << std::endl; // Fallback, may not be optimal
            }
            if (remaining_ops > 0) remaining_ops--; // Decrement if an op was actually performed
            continue;
        }
        
        double avg_time_per_remaining_target_ms = (remaining_time_budget_s / (H_num_targets - i_target) * 1000.0);
        if (H_num_targets - i_target <= 0) avg_time_per_remaining_target_ms = 0;
        
        double sa_time_limit_ms_for_this_target = std::max(0.01, std::min(avg_time_per_remaining_target_ms, 2.7));
        // Python: if remaining_time_budget_s < 0.05 and i_target < H_num_targets -1: sa_time_limit_ms_for_this_target = 0.03
        // This specific Python condition is not directly translated but the general idea of adaptive time limit is.
        // The 2.7ms cap might be too small. Original Python had 2.7. Let's try something larger like 50ms or related to avg.
        // The Python code has 2.7, perhaps it's a fixed small value for quick SA runs.
        // Let's keep 2.7 as per Python, could be a typo for 27 or 270.
        
        const Color& target_color = targets[i_target];

        int best_reuse_slot_idx = -1;
        double min_reuse_error = std::numeric_limits<double>::infinity();
        for (int slot_idx = 0; slot_idx < NUM_TOTAL_SLOTS; ++slot_idx) {
            if (remaining_paint_in_slot[slot_idx] > 0) {
                double err = calculate_error(current_colors_in_slot[slot_idx], target_color);
                if (err < min_reuse_error) {
                    min_reuse_error = err;
                    best_reuse_slot_idx = slot_idx;
                }
            }
        }
        bool reuse_possible = (best_reuse_slot_idx != -1);
        
        std::vector<int> m_candidate_list_for_target = m_cands; // Make a copy
        
        double best_m_score = std::numeric_limits<double>::infinity();
        int best_m_val_actual = 0; // Actual number of colors used for the best m
        std::vector<int> best_m_indices;
        Color best_m_mixed_color = {0,0,0};
        // double best_m_err = std::numeric_limits<double>::infinity(); // Already captured in best_m_score effectively

        if (!m_candidate_list_for_target.empty() && K_num_colors > 0) { // Only try to create if we have colors and candidates for m
            for (int m_trial : m_candidate_list_for_target) {
                if (m_trial == 0) continue; // m must be > 0
                if (m_trial + 1 > remaining_ops) continue; // Basic check: m paints + 1 use
                                                           // This doesn't include discards yet.

                std::vector<int> current_m_indices;
                double current_m_err;
                Color current_m_mixed_color;

                if (m_trial <= 3) { // Small m, use exhaustive
                    std::tie(current_m_indices, current_m_err, current_m_mixed_color) = 
                        select_colors_exhaustive(own_colors, target_color, m_trial);
                } else { // Larger m, use SA (as per final Python version)
                    // Or GD, as per Python comments:
                    // std::tie(current_m_indices, current_m_err, current_m_mixed_color) =
                    //    select_colors_gd(own_colors, target_color, m_trial, D_cost_factor);

                    SAParams sa_p = {10, 1e-3, 0.97, 10, 10}; // Default SA params from Python
                    std::tie(current_m_indices, current_m_err, current_m_mixed_color) = 
                        select_colors_sa_timed(own_colors, target_color, m_trial, sa_p, sa_time_limit_ms_for_this_target);
                }
                
                if (current_m_indices.empty() && m_trial > 0) { // Fallback if a strategy failed to return indices for m_trial > 0
                    if (!own_colors.empty()) { // Can only do this if own_colors exist
                        current_m_indices.assign(m_trial, 0); // Use first color m_trial times
                        current_m_mixed_color = get_mixed_color(current_m_indices, own_colors, m_trial);
                        current_m_err = calculate_error(current_m_mixed_color, target_color);
                    } else { // No colors to pick, error will be for (0,0,0)
                        current_m_err = calculate_error({0,0,0}, target_color);
                        current_m_mixed_color = {0,0,0};
                        // current_m_indices remains empty
                    }
                }


                int actual_num_colors_mixed = current_m_indices.size();
                double mixing_cost = 0;
                if (actual_num_colors_mixed > 0) {
                    mixing_cost = D_cost_factor * (actual_num_colors_mixed -1);
                }
                 // If actual_num_colors_mixed is 0 (e.g. GD decided not to pick any), error is high, mixing_cost is 0.
                
                double score_candidate = current_m_err * 10000.0 + mixing_cost;
                
                // std::cerr << "Target " << i_target << " m_trial=" << m_trial << " actual_len=" << actual_num_colors_mixed << " err=" << current_m_err << " score=" << score_candidate << std::endl;

                if (score_candidate < best_m_score) {
                    for(auto idx : current_m_indices) {
                        std::cerr << idx << " "; // Debug output for indices
                    }
                    std::cerr << std::endl;
                    std::cerr << "UPDATED OPTM=" << actual_num_colors_mixed << " score=" << score_candidate << " err=" << current_m_err << std::endl;
                    
                    best_m_score = score_candidate;
                    best_m_val_actual = actual_num_colors_mixed;
                    best_m_indices = current_m_indices;
                    best_m_mixed_color = current_m_mixed_color;
                    // best_m_err = current_m_err;
                }
            }
        }


        bool use_existing_paint = false;
        if (reuse_possible) {
            double score_reuse = min_reuse_error * 10000.0; // Cost of reuse is just error (no mixing cost)
            // Python: if min_reuse_error < 0.005 or score_reuse <= best_m_score:
            if (min_reuse_error < 0.005 || score_reuse <= best_m_score) {
                use_existing_paint = true;
            }
        }
        
        // If no best_m_indices found (e.g. K_num_colors=0 or m_cands empty), best_m_val_actual will be 0.
        // In this case, if reuse is not possible/better, we might be stuck.
        // The problem implies we must always output something.

        if (use_existing_paint) {
            int actual_row_ops = (best_reuse_slot_idx < N_grid_size) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col_ops = best_reuse_slot_idx % N_grid_size;
            std::cout << "2 " << actual_row_ops << " " << actual_col_ops << std::endl;
            remaining_paint_in_slot[best_reuse_slot_idx]--;
            remaining_ops--;
            // std::cerr << "Target " << i_target << ": Reusing slot " << best_reuse_slot_idx << " Error: " << min_reuse_error << std::endl;
            continue; 
        }

        // If we reach here, we create new paint.
        // Handle case where no new paint could be determined (best_m_val_actual == 0)
        // This could happen if K_num_colors == 0 or all m_cands are too costly/error-prone.
        if (best_m_val_actual == 0) {
            // Cannot create new. If reuse was also not chosen, we are in a tricky spot.
            // Fallback: try to use any existing paint if not chosen by score, or just a dummy command.
            if (best_reuse_slot_idx != -1) { // Reuse even if score wasn't better, as a last resort
                 int actual_row_ops = (best_reuse_slot_idx < N_grid_size) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                 int actual_col_ops = best_reuse_slot_idx % N_grid_size;
                 std::cout << "2 " << actual_row_ops << " " << actual_col_ops << std::endl;
                 remaining_paint_in_slot[best_reuse_slot_idx]--;
                 remaining_ops--;
                 // std::cerr << "Target " << i_target << ": Fallback reuse slot " << best_reuse_slot_idx << std::endl;
            } else { // Absolute last resort: dummy command
                 std::cout << "2 0 0" << std::endl; // Use (0,0) paint
                 remaining_ops--; // Count this op
                 // std::cerr << "Target " << i_target << ": Fallback dummy 2 0 0" << std::endl;
            }
            continue;
        }


        // Find a slot to create in (prefer empty, otherwise overwrite following current_slot_idx_to_fill)
        int slot_to_create_in = -1;
        int search_start_slot = current_slot_idx_to_fill;
        for(int offset = 0; offset < NUM_TOTAL_SLOTS; ++offset) {
            int temp_slot = (search_start_slot + offset) % NUM_TOTAL_SLOTS;
            if(remaining_paint_in_slot[temp_slot] == 0) {
                slot_to_create_in = temp_slot;
                break;
            }
        }
        if(slot_to_create_in == -1) { // No empty slots, overwrite current_slot_idx_to_fill
            slot_to_create_in = current_slot_idx_to_fill;
        }


        int actual_row_ops_create = (slot_to_create_in < N_grid_size) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
        int actual_col_ops_create = slot_to_create_in % N_grid_size;

        int discard_ops_needed = remaining_paint_in_slot[slot_to_create_in];
        int total_ops_for_new_paint = discard_ops_needed + best_m_val_actual + 1; // discards + paints + 1 use

        if (total_ops_for_new_paint > remaining_ops) {
            // Not enough ops to create this new paint.
            // Try to fall back to reuse if it was possible, even if score was worse.
            if (best_reuse_slot_idx != -1 && remaining_ops >=1 ) { // Check if reuse op itself is possible
                 int actual_row_ops = (best_reuse_slot_idx < N_grid_size) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                 int actual_col_ops = best_reuse_slot_idx % N_grid_size;
                 std::cout << "2 " << actual_row_ops << " " << actual_col_ops << std::endl;
                 remaining_paint_in_slot[best_reuse_slot_idx]--;
                 remaining_ops--;
                // std::cerr << "Target " << i_target << ": Fallback (due to ops) reuse slot " << best_reuse_slot_idx << std::endl;

            } else { // Absolute last resort due to ops: dummy command
                 std::cout << "2 0 0" << std::endl; 
                 if (remaining_ops > 0) remaining_ops--;
                // std::cerr << "Target " << i_target << ": Fallback (due to ops) dummy 2 0 0" << std::endl;
            }
            continue;
        }

        // Proceed with creating new paint
        // std::cerr << "Target " << i_target << ": Creating new in slot " << slot_to_create_in << " with m_actual=" << best_m_val_actual << std::endl;

        for (int d = 0; d < discard_ops_needed; ++d) {
            std::cout << "3 " << actual_row_ops_create << " " << actual_col_ops_create << std::endl;
        }
        remaining_ops -= discard_ops_needed;
        remaining_paint_in_slot[slot_to_create_in] = 0; // Reset paint amount
            
        for (int c_idx : best_m_indices) {
            std::cout << "1 " << actual_row_ops_create << " " << actual_col_ops_create << " " << c_idx << std::endl;
        }
        std::cout << "2 " << actual_row_ops_create << " " << actual_col_ops_create << std::endl;
        
        current_colors_in_slot[slot_to_create_in] = best_m_mixed_color;
        remaining_paint_in_slot[slot_to_create_in] = best_m_val_actual - 1; // One use consumed
        current_slot_idx_to_fill = (slot_to_create_in + 1) % NUM_TOTAL_SLOTS; // Advance preferred next slot
        remaining_ops -= (best_m_val_actual + 1); // paints + 1 use
    }
    
    return 0;
}