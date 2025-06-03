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

// --- Global Random Number Generator ---
// std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::mt19937 rng(42); // Fixed seed for reproducibility, change to a random seed if needed

// --- Helper Structs and Functions ---
struct Color {
    double r, g, b;
    Color(double r_ = 0.0, double g_ = 0.0, double b_ = 0.0) : r(r_), g(g_), b(b_) {}

    Color operator+(const Color& other) const {
        return {r + other.r, g + other.g, b + other.b};
    }
    Color operator/(double scalar) const {
        return {r / scalar, g / scalar, b / scalar};
    }
};

Color get_mixed_color(const std::vector<int>& indices_to_mix,
                      const std::vector<Color>& own_colors_list_ref,
                      int m_val_eff) {
    if (indices_to_mix.empty() || m_val_eff == 0) {
        return {0.0, 0.0, 0.0};
    }
    Color sum_color = {0.0, 0.0, 0.0};
    for (int idx : indices_to_mix) {
        sum_color = sum_color + own_colors_list_ref[idx];
    }
    return sum_color / m_val_eff;
}

double calculate_error_sq(const Color& mixed_color, const Color& target_color) {
    double dr = mixed_color.r - target_color.r;
    double dg = mixed_color.g - target_color.g;
    double db = mixed_color.b - target_color.b;
    return dr * dr + dg * dg + db * db;
}

double calculate_error_sqrt(const Color& mixed_color, const Color& target_color) {
    return std::sqrt(calculate_error_sq(mixed_color, target_color));
}


// --- Color Selection Strategies (Exhaustive, RS, SA - from previous code) ---
void combinations_with_replacement_recursive(
    const std::vector<Color>& own_colors_list,
    const Color& target_color_vec,
    int m_val,
    int start_index,
    std::vector<int>& current_combination,
    std::vector<int>& best_indices,
    double& best_err_val_sqrt, // now stores sqrt error
    Color& best_mixed_color_tuple,
    bool& found_one) {

    if (current_combination.size() == static_cast<size_t>(m_val)) {
        Color mixed_color = get_mixed_color(current_combination, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec); // Use sqrt
        found_one = true;
        if (current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_mixed_color_tuple = mixed_color;
            best_indices = current_combination;
        }
        return;
    }
    if (static_cast<size_t>(start_index) >= own_colors_list.size()) return;
    if (own_colors_list.empty() && m_val > current_combination.size()) return;

    for (size_t i = start_index; i < own_colors_list.size(); ++i) {
        current_combination.push_back(i);
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
    std::vector<int> best_indices_to_drop;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        std::vector<int> default_indices;
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {default_indices, err_sqrt, default_mixed_color};
    }
    
    bool found_one = false;
    std::vector<int> current_combination;
    current_combination.reserve(m_val);

    combinations_with_replacement_recursive(
        own_colors_list, target_color_vec, m_val, 0,
        current_combination, best_indices_to_drop, best_err_val_sqrt, best_mixed_color_tuple, found_one);
            
    if (!found_one && num_own_colors > 0 && m_val > 0) { 
        best_indices_to_drop.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val);
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
        std::vector<int> default_indices;
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {default_indices, err_sqrt, default_mixed_color};
    }
    
    std::uniform_int_distribution<int> dist(0, num_own_colors - 1);
    bool found_one = false;
    std::vector<int> current_indices(m_val);

    for (int s = 0; s < num_samples; ++s) {
        for (int i = 0; i < m_val; ++i) {
            current_indices[i] = dist(rng);
        }
        Color mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec);
        found_one = true;
        if (current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_indices_to_drop = current_indices; 
            best_mixed_color_tuple = mixed_color;
        }
    }
    
    if (!found_one && num_own_colors > 0 && m_val > 0) {
        best_indices_to_drop.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val);
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
    double time_limit_ms) {

    size_t num_own_colors = own_colors_list.size();
    auto start_time_sa_chrono = std::chrono::steady_clock::now();

    if (num_own_colors == 0 || m_val == 0) {
        Color default_mixed_color = {0.0, 0.0, 0.0};
        std::vector<int> default_indices;
        double err_sqrt = calculate_error_sqrt(default_mixed_color, target_color_vec);
        return {default_indices, err_sqrt, default_mixed_color};
    }

    std::vector<int> current_indices;
    double current_error_sqrt; // Store sqrt error
    Color current_mixed_color;

    std::tie(current_indices, current_error_sqrt, current_mixed_color) = select_colors_rs(
        own_colors_list, target_color_vec, m_val, sa_params.initial_rs_samples);
    
    if (current_indices.empty() && m_val > 0 && num_own_colors > 0) {
        current_indices.assign(m_val, 0);
        current_mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
        current_error_sqrt = calculate_error_sqrt(current_mixed_color, target_color_vec);
    }
            
    std::vector<int> best_indices = current_indices; 
    double best_error_sqrt = current_error_sqrt;
    Color best_mixed_color_tuple = current_mixed_color;
    
    double current_temp = sa_params.initial_temp;
    std::uniform_real_distribution<double> unif_real_dist(0.0, 1.0);
    std::uniform_int_distribution<int> idx_change_dist(0, m_val - 1); 
    std::uniform_int_distribution<int> color_idx_dist(0, num_own_colors - 1);

    while (current_temp > sa_params.final_temp) {
        auto check_time = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break;

        for (int iter_idx = 0; iter_idx < sa_params.iterations_per_temp; ++iter_idx) {
            check_time = std::chrono::steady_clock::now();
            if (std::chrono::duration<double, std::milli>(check_time - start_time_sa_chrono).count() > time_limit_ms) break; 

            if (current_indices.empty()) { 
                if (m_val > 0 && num_own_colors > 0) { 
                    current_indices.assign(m_val, 0);
                    current_mixed_color = get_mixed_color(current_indices, own_colors_list, m_val);
                    current_error_sqrt = calculate_error_sqrt(current_mixed_color, target_color_vec);
                    if (current_error_sqrt < best_error_sqrt) { 
                         best_error_sqrt = current_error_sqrt;
                         best_indices = current_indices;
                         best_mixed_color_tuple = current_mixed_color;
                    }
                } else { break; }
            }
            if (current_indices.empty()) break;

            std::vector<int> neighbor_indices = current_indices; 
            int idx_to_change = idx_change_dist(rng); 
            int new_color_idx = color_idx_dist(rng);  
            if (num_own_colors > 1) { 
                while (new_color_idx == neighbor_indices[idx_to_change]) {
                    new_color_idx = color_idx_dist(rng);
                }
            }
            neighbor_indices[idx_to_change] = new_color_idx;

            Color neighbor_mixed_color = get_mixed_color(neighbor_indices, own_colors_list, m_val);
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
        if (current_indices.empty() && m_val > 0) break; 
        current_temp *= sa_params.cooling_rate;
    }
    return {best_indices, best_error_sqrt, best_mixed_color_tuple};
}

// --- Parameters for Batch Creation ---
const double BATCH_SIMILARITY_THRESHOLD_SQ = 0.005 * 0.005; // Squared error threshold for similarity
const int BATCH_MAX_LOOK_AHEAD = 5;         // Max future targets to consider for a batch
const int BATCH_MIN_SIZE = 2;               // Minimum number of targets in a batch (including current)
const int MAX_PAINT_IN_WELL = 10;           // Max grams a well can hold (based on N=10, assuming 1 cell per gram)
                                            // This should be N_grid_size (palette size) if using single cell wells.
                                            // Or (N_grid_size / NUM_TOTAL_SLOTS) if wells are N_grid_size / NUM_TOTAL_SLOTS big
                                            // For the current slot setup (2*N slots, each 10 cells), capacity is 10.

struct SlotInfo {
    Color color;
    int remaining_grams;
    bool is_batch_paint; // True if this paint was created for a batch
    int batch_targets_remaining; // How many more targets this batch paint is intended for

    SlotInfo() : color({0,0,0}), remaining_grams(0), is_batch_paint(false), batch_targets_remaining(0) {}
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
    const int HORIZONTAL_DIVIDER_ROW_INDEX = N_grid_size / 2 -1; // e.g. 9 for N=20
    for (int r_h_idx = 0; r_h_idx < N_grid_size - 1; ++r_h_idx) {
        for (int j = 0; j < N_grid_size; ++j) {
             // Keep upper and lower half separate, but allow flow within each half by default
            std::cout << (j > 0 ? " " : "") << (r_h_idx == HORIZONTAL_DIVIDER_ROW_INDEX ? "1" : "0");
        }
        std::cout << std::endl;
    }
    
    const int WELL_CAPACITY = 10; // Each slot (well) uses N_grid_size cells from one row
    const int NUM_SLOTS_PER_HALF = N_grid_size; // N slots in upper half, N in lower half
    const int NUM_TOTAL_SLOTS = 2 * NUM_SLOTS_PER_HALF;

    std::vector<SlotInfo> slots(NUM_TOTAL_SLOTS);
    int current_slot_idx_preference = 0; 
    
    std::set<int> m_cands_set; 
    int max_m_val_heuristic = 0; 
    for (int m = 1; m <= WELL_CAPACITY; ++m) { // m_val cannot exceed well capacity
        if (2LL * m * 1000 <= T_total_ops_limit) { 
            max_m_val_heuristic = m;
        }
    }
    if (max_m_val_heuristic > 0) {
        for (int m = 1; m <= max_m_val_heuristic; ++m) m_cands_set.insert(m);
        m_cands_set.insert(std::max(max_m_val_heuristic / 2, 1));
    }
    if (m_cands_set.empty() && K_num_colors > 0) {
        if (1 <= T_total_ops_limit) m_cands_set.insert(1);
    }
    std::vector<int> m_cands(m_cands_set.begin(), m_cands_set.end());
    std::sort(m_cands.begin(), m_cands.end()); // Ensure sorted
    
    const double TOTAL_TIME_BUDGET_S = 2.95;
    int remaining_ops = T_total_ops_limit;
    std::vector<std::string> operations_log; // Store operations before printing
    double progress=0;
    for (int i_target = 0; i_target < H_num_targets; ++i_target) {
        if (remaining_ops <= 0 && i_target < H_num_targets) { // Must still output H use operations
             // Try to use any paint, even if 0 grams left, just to output "2 x y"
            int fallback_row = ( (i_target % NUM_TOTAL_SLOTS) < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int fallback_col = (i_target % NUM_TOTAL_SLOTS) % NUM_SLOTS_PER_HALF;
            operations_log.push_back("2 " + to_string(fallback_row) + " " + to_string(fallback_col));
            // No actual op cost deduction as remaining_ops is already 0
            continue;
        }
        progress= static_cast<double>(i_target + 1) / H_num_targets;
        double discount_factor;
        // discount_factor = (0.1 + progress *9/ 10.0);
        discount_factor=0.15;
        
        auto current_time_chrono = std::chrono::steady_clock::now();
        double time_spent_s = std::chrono::duration<double>(current_time_chrono - overall_start_time_chrono).count();
        double remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s;
        double avg_time_per_remaining_target_ms = 0;
        if (H_num_targets - i_target > 0) {
            avg_time_per_remaining_target_ms = (remaining_time_budget_s / (H_num_targets - i_target) * 1000.0);
        }
        double sa_time_limit_ms = std::max(0.01, std::min(avg_time_per_remaining_target_ms*.9, 2.7));
        if (remaining_time_budget_s < 0.05 && i_target < H_num_targets -1 ) sa_time_limit_ms = 0.03;


        const Color& current_target_color_obj = targets[i_target];

        // --- Strategy 1: Batch Creation ---
        bool do_batch_creation = false;
        Color batch_avg_target_color;
        int batch_num_targets_covered = 0;
        std::vector<int> batch_m_indices;
        Color batch_mixed_color;
        double batch_score = std::numeric_limits<double>::infinity();
        int batch_m_val_chosen = 0;

        if (i_target + BATCH_MIN_SIZE -1 < H_num_targets && K_num_colors > 0) { // Enough targets left for a minimal batch and colors exist
            std::vector<const Color*> targets_for_batch;
            targets_for_batch.push_back(&targets[i_target]);
            Color temp_avg_color = targets[i_target];

            for (int k = 1; k < BATCH_MAX_LOOK_AHEAD && (i_target + k) < H_num_targets; ++k) {
                if (calculate_error_sq(targets[i_target+k], targets[i_target]) < BATCH_SIMILARITY_THRESHOLD_SQ) {
                    targets_for_batch.push_back(&targets[i_target+k]);
                    temp_avg_color = temp_avg_color + targets[i_target+k];
                } else {
                    break;
                }
            }

        //     if (targets_for_batch.size() >= BATCH_MIN_SIZE) {
        //         batch_num_targets_covered = targets_for_batch.size();
        //         batch_avg_target_color = temp_avg_color / batch_num_targets_covered;

        //         // Find best m_val for this batch_avg_target_color, ensuring m_val >= batch_num_targets_covered
        //         double current_best_batch_creation_score = std::numeric_limits<double>::infinity();
                
        //         for (int m_trial : m_cands) {
        //             if (m_trial == 0 || m_trial < batch_num_targets_covered || m_trial > WELL_CAPACITY) continue;
                    
        //             std::vector<int> temp_indices;
        //             double temp_err_sqrt;
        //             Color temp_mixed_c;
        //             // For batch, always use SA for now, or a faster method if time is critical
        //             SAParams sa_p_batch = {5, 1e-3, 0.98, 5, 5}; // Faster SA for batch eval
        //             std::tie(temp_indices, temp_err_sqrt, temp_mixed_c) = 
        //                 select_colors_sa_timed(own_colors, batch_avg_target_color, m_trial, sa_p_batch, sa_time_limit_ms/2.0); // Use part of time budget

        //             if (temp_indices.empty() && m_trial > 0 && !own_colors.empty()) { // Fallback
        //                 temp_indices.assign(m_trial, 0);
        //                 temp_mixed_c = get_mixed_color(temp_indices, own_colors, m_trial);
        //                 temp_err_sqrt = calculate_error_sqrt(temp_mixed_c, batch_avg_target_color);
        //             } else if (temp_indices.empty()) continue;


        //             double total_err_for_batch_sq = 0;
        //             for(const Color* target_ptr : targets_for_batch){
        //                 total_err_for_batch_sq += calculate_error_sq(temp_mixed_c, *target_ptr);
        //             }
        //             double avg_err_sqrt_for_batch = std::sqrt(total_err_for_batch_sq / batch_num_targets_covered);
                    
        //             // Score for creating m_trial paint, using it for batch_num_targets_covered times
        //             double cost_paint_tubes =discount_factor* D_cost_factor * m_trial;
        //             // Benefit is using it batch_num_targets_covered times
        //             // Effective cost per target for paint: D_cost_factor * m_trial / batch_num_targets_covered
        //             // Or, cost_waste = D_cost_factor * (m_trial - batch_num_targets_covered)
        //             // Score = avg_err_sqrt * 10000 + D_cost_factor * (m_trial - batch_num_targets_covered) / batch_num_targets_covered; // Heuristic
        //             double this_batch_candidate_score = avg_err_sqrt_for_batch * 10000.0 +discount_factor* D_cost_factor * (m_trial - batch_num_targets_covered);


        //             if (this_batch_candidate_score < current_best_batch_creation_score) {
        //                 current_best_batch_creation_score = this_batch_candidate_score;
        //                 batch_m_indices = temp_indices;
        //                 batch_mixed_color = temp_mixed_c;
        //                 batch_score = this_batch_candidate_score; // Store the overall score for this batch option
        //                 batch_m_val_chosen = m_trial;
        //             }
        //         }
        //         if (!batch_m_indices.empty()) {
        //             do_batch_creation = true; // Found a viable batch creation
        //         }
        //     }
        }

        // --- Strategy 2: Reuse existing paint ---
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
        double reuse_score = reuse_is_possible ? (min_reuse_error_sqrt * 10000.0) : std::numeric_limits<double>::infinity();

        // --- Strategy 3: Create new 1-gram paint ---
        double best_1g_score = std::numeric_limits<double>::infinity();
        std::vector<int> best_1g_indices;
        Color best_1g_mixed_color;
        int best_1g_m_val = 0;

        if (K_num_colors > 0) { // Only if colors exist
            for (int m_trial : m_cands) {
                if (m_trial == 0 || m_trial > WELL_CAPACITY) continue;

                std::vector<int> current_m_indices;
                double current_m_err_sqrt;
                Color current_m_mixed_c;

                SAParams sa_p = {10, 1e-4, 0.97, 10, 10}; // Default SA
                if (m_trial<=5 or (m_trial<=6 and K_num_colors<=8)) { // Small m, small K, use exhaustive
                     std::tie(current_m_indices, current_m_err_sqrt, current_m_mixed_c) = 
                        select_colors_exhaustive(own_colors, current_target_color_obj, m_trial);
                } else {
                    std::tie(current_m_indices, current_m_err_sqrt, current_m_mixed_c) = 
                        select_colors_sa_timed(own_colors, current_target_color_obj, m_trial, sa_p, sa_time_limit_ms);
                }

                if (current_m_indices.empty() && m_trial > 0 && !own_colors.empty()) { // Fallback
                    current_m_indices.assign(m_trial, 0);
                    current_m_mixed_c = get_mixed_color(current_m_indices, own_colors, m_trial);
                    current_m_err_sqrt = calculate_error_sqrt(current_m_mixed_c, current_target_color_obj);
                } else if (current_m_indices.empty()) continue;

                double score_candidate = current_m_err_sqrt * 10000.0 + discount_factor*D_cost_factor * (m_trial - 1);

                if (score_candidate < best_1g_score) {
                    best_1g_score = score_candidate;
                    best_1g_indices = current_m_indices;
                    best_1g_mixed_color = current_m_mixed_c;
                    best_1g_m_val = m_trial;
                }
            }
        }
        
        // --- Decide Strategy ---
        bool chose_batch = false;
        bool chose_reuse = false;
        bool chose_new_1g = false;

        // Normalize batch_score to be comparable (score for the first target of the batch)
        double batch_first_target_err_sqrt = std::numeric_limits<double>::infinity();
        if(do_batch_creation){
            batch_first_target_err_sqrt = calculate_error_sqrt(batch_mixed_color, targets[i_target]);
        }
        // Simplified comparison:
        // Consider cost of paint tubes for the *first* use.
        // Batch: D_cost_factor * (batch_m_val_chosen - 1)  (if we assume 1 use, then rest is bonus)
        // 1G: D_cost_factor * (best_1g_m_val - 1)
        double batch_comparable_score = do_batch_creation ? (batch_first_target_err_sqrt * 10000.0 + D_cost_factor * (batch_m_val_chosen -1) ) 
                                                         : std::numeric_limits<double>::infinity();


        if (do_batch_creation && batch_comparable_score <= reuse_score && batch_comparable_score <= best_1g_score) {
            // More sophisticated check: is the *total benefit* of batching worth it?
            // Total error for batch targets vs total error for individual creation, considering paint costs.
            // For now, use the simplified score for the first target.
            // And ensure enough ops for the full batch creation.
            int ops_for_batch = batch_m_val_chosen + 1; // paints + 1 use (discards handled separately)
            if (ops_for_batch <= remaining_ops) {
                 chose_batch = true;
            }
        }
        
        if (!chose_batch && reuse_is_possible && reuse_score <= best_1g_score) {
             if (1 <= remaining_ops) { // Ops for use
                chose_reuse = true;
             }
        }
        
        if (!chose_batch && !chose_reuse && !best_1g_indices.empty()) {
            int ops_for_1g = best_1g_m_val + 1; // paints + 1 use
            if (ops_for_1g <= remaining_ops) {
                chose_new_1g = true;
            }
        }


        // --- Execute Chosen Strategy ---
        if (chose_batch) {
            int slot_to_use = -1;
            // Find/clear slot
            int search_start_pref = current_slot_idx_preference;
            for(int offset = 0; offset < NUM_TOTAL_SLOTS; ++offset) {
                int temp_slot = (search_start_pref + offset) % NUM_TOTAL_SLOTS;
                if(slots[temp_slot].remaining_grams == 0) { slot_to_use = temp_slot; break; }
            }
            if(slot_to_use == -1) slot_to_use = current_slot_idx_preference;

            int actual_row = (slot_to_use < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col = slot_to_use % NUM_SLOTS_PER_HALF;

            int discards_needed = slots[slot_to_use].remaining_grams;
            if (discards_needed + batch_m_val_chosen + 1 > remaining_ops) { // Re-check with discards
                chose_batch = false; // Cannot do it, try other options
                 if (reuse_is_possible && reuse_score <= best_1g_score && 1 <= remaining_ops) chose_reuse = true;
                 else if (!best_1g_indices.empty() && (best_1g_m_val + 1 <= remaining_ops)) chose_new_1g = true;
                 else chose_new_1g = false; // Reset if it was true but now ops are too low
            } else {
                for(int d=0; d<discards_needed; ++d) operations_log.push_back("3 " + to_string(actual_row) + " " + to_string(actual_col));
                remaining_ops -= discards_needed;
                
                for(int c_idx : batch_m_indices) operations_log.push_back("1 " + to_string(actual_row) + " " + to_string(actual_col) + " " + to_string(c_idx));
                operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));
                
                slots[slot_to_use].color = batch_mixed_color;
                slots[slot_to_use].remaining_grams = batch_m_val_chosen - 1;
                slots[slot_to_use].is_batch_paint = true;
                slots[slot_to_use].batch_targets_remaining = batch_num_targets_covered - 1;
                current_slot_idx_preference = (slot_to_use + 1) % NUM_TOTAL_SLOTS;
                remaining_ops -= (batch_m_val_chosen + 1);
                i_target += (batch_num_targets_covered - 1); // Advance i_target
            }
        }
        
        if (chose_reuse && !chose_batch) { // Ensure batch wasn't chosen then failed ops check
            int actual_row = (best_reuse_slot_idx < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col = best_reuse_slot_idx % NUM_SLOTS_PER_HALF;
            operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));
            slots[best_reuse_slot_idx].remaining_grams--;
            if (slots[best_reuse_slot_idx].is_batch_paint) {
                slots[best_reuse_slot_idx].batch_targets_remaining--;
                if (slots[best_reuse_slot_idx].batch_targets_remaining <= 0) {
                    slots[best_reuse_slot_idx].is_batch_paint = false; // Batch fulfilled
                }
            }
            remaining_ops--;
        }
        
        if (chose_new_1g && !chose_batch && !chose_reuse) {
            int slot_to_use = -1;
            int search_start_pref = current_slot_idx_preference;
            for(int offset = 0; offset < NUM_TOTAL_SLOTS; ++offset) {
                int temp_slot = (search_start_pref + offset) % NUM_TOTAL_SLOTS;
                 // Prefer non-batch slots or empty slots
                if(!slots[temp_slot].is_batch_paint || slots[temp_slot].remaining_grams == 0) {
                    if (slots[temp_slot].remaining_grams == 0) {slot_to_use = temp_slot; break;}
                    if (slot_to_use == -1) slot_to_use = temp_slot; // Keep first non-batch, non-empty
                }
            }
            if(slot_to_use == -1) slot_to_use = current_slot_idx_preference; // Fallback to overwrite preferred

            int actual_row = (slot_to_use < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
            int actual_col = slot_to_use % NUM_SLOTS_PER_HALF;

            int discards_needed = slots[slot_to_use].remaining_grams;
            if (discards_needed + best_1g_m_val + 1 > remaining_ops) { // Re-check with discards
                 chose_new_1g = false; // Ops fail, try final fallback
            } else {
                for(int d=0; d<discards_needed; ++d) operations_log.push_back("3 " + to_string(actual_row) + " " + to_string(actual_col));
                remaining_ops -= discards_needed;

                for(int c_idx : best_1g_indices) operations_log.push_back("1 " + to_string(actual_row) + " " + to_string(actual_col) + " " + to_string(c_idx));
                operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));

                slots[slot_to_use].color = best_1g_mixed_color;
                slots[slot_to_use].remaining_grams = best_1g_m_val - 1;
                slots[slot_to_use].is_batch_paint = false;
                slots[slot_to_use].batch_targets_remaining = 0;
                current_slot_idx_preference = (slot_to_use + 1) % NUM_TOTAL_SLOTS;
                remaining_ops -= (best_1g_m_val + 1);
            }
        }

        if (!chose_batch && !chose_reuse && !chose_new_1g) { // Fallback if no strategy chosen or ops failed
            if (reuse_is_possible && 1 <= remaining_ops) { // Fallback to reuse if possible
                int actual_row = (best_reuse_slot_idx < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                int actual_col = best_reuse_slot_idx % NUM_SLOTS_PER_HALF;
                operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));
                slots[best_reuse_slot_idx].remaining_grams--;
                 if (slots[best_reuse_slot_idx].is_batch_paint) slots[best_reuse_slot_idx].batch_targets_remaining--;
                remaining_ops--;
            } else if (!best_1g_indices.empty() && (1 <= remaining_ops)) { // Fallback to create 1g if its indices determined but ops failed earlier
                 // This is a bit risky, as we might not have enough ops for paint *and* use
                 // Simplified: just do a "use" from a default slot
                int fallback_row = ( (i_target % NUM_TOTAL_SLOTS) < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                int fallback_col = (i_target % NUM_TOTAL_SLOTS) % NUM_SLOTS_PER_HALF;
                operations_log.push_back("2 " + to_string(fallback_row) + " " + to_string(fallback_col));
                if (remaining_ops > 0) remaining_ops--;

            } else { // Absolute fallback
                int fallback_row = ( (i_target % NUM_TOTAL_SLOTS) < NUM_SLOTS_PER_HALF) ? 0 : (HORIZONTAL_DIVIDER_ROW_INDEX + 1);
                int fallback_col = (i_target % NUM_TOTAL_SLOTS) % NUM_SLOTS_PER_HALF;
                operations_log.push_back("2 " + to_string(fallback_row) + " " + to_string(fallback_col));
                if (remaining_ops > 0) remaining_ops--;
            }
        }
    }

    // Output all logged operations
    for(const auto& op_str : operations_log) {
        std::cout << op_str << std::endl;
    }
    
    return 0;
}