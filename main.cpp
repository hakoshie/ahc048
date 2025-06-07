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
    
    // ===== START: Palette Layout Configuration =====
    // 縦仕切り v[i][j]: 全て閉じる
    for (int i = 0; i < N_grid_size; ++i) {
        for (int j = 0; j < N_grid_size - 1; ++j) {
            std::cout << (j > 0 ? " " : "") << "1"; 
        }
        std::cout << std::endl;
    }

    const int WELL_HEIGHT = 5; // 各ウェルの高さを5マスに設定
    // 横仕切り h[i][j]: 5マスごとに区切る
    for (int r_h_idx = 0; r_h_idx < N_grid_size - 1; ++r_h_idx) {
        // r_h_idx = 4, 9, 14 ... の時に仕切りを閉じる (1)
        bool close_divider = ((r_h_idx + 1) % WELL_HEIGHT == 0);
        for (int j = 0; j < N_grid_size; ++j) {
            std::cout << (j > 0 ? " " : "") << (close_divider ? "1" : "0");
        }
        std::cout << std::endl;
    }
    
    const int ACTUAL_WELL_CAPACITY = WELL_HEIGHT; 
    const int NUM_SLOTS_PER_REGION = N_grid_size;
    const int NUM_REGIONS = N_grid_size / ACTUAL_WELL_CAPACITY;
    const int NUM_TOTAL_SLOTS = NUM_REGIONS * NUM_SLOTS_PER_REGION; // 4 * 20 = 80 slots

    // ヘルパー関数: スロットインデックスからパレットの座標を取得
    auto get_coords_from_slot_idx = [&](int slot_idx) {
        int region_index = slot_idx / NUM_SLOTS_PER_REGION; // 0, 1, 2, 3
        int col_in_region = slot_idx % NUM_SLOTS_PER_REGION; // 0-19
        int start_row = region_index * ACTUAL_WELL_CAPACITY;
        return make_pair(start_row, col_in_region);
    };
    // ===== END: Palette Layout Configuration =====
    
    std::vector<SlotInfo> slots(NUM_TOTAL_SLOTS);
    int current_slot_idx_preference = 0; 
    std::set<int> m_cands_set; 
    for (int m = 1; m <= ACTUAL_WELL_CAPACITY; ++m) {
        if(2*m*1000<=T_total_ops_limit) m_cands_set.insert(m);
    }

    std::vector<int> m_cands(m_cands_set.begin(), m_cands_set.end());
    std::sort(m_cands.begin(), m_cands.end());
    
    const double TOTAL_TIME_BUDGET_S = (K_num_colors > 40 && H_num_targets > 800 && N_grid_size > 15) ? 2.85 : 2.95;
    int remaining_ops = T_total_ops_limit;
    std::vector<std::string> operations_log; 
    int num_blank_slots = NUM_TOTAL_SLOTS;

    for (int i_target = 0; i_target < H_num_targets; ++i_target) {
        if (remaining_ops <= 0 && i_target < H_num_targets) { 
            int fallback_slot_idx = (current_slot_idx_preference + i_target) % NUM_TOTAL_SLOTS;
            auto coords = get_coords_from_slot_idx(fallback_slot_idx);
            operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));
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
        int mix_num=min(max(min((int)D_cost_factor/180,5),1), H_num_targets - i_target);//300?
        FOR(j,i_target,min(H_num_targets,i_target +180*mix_num)){// 200? 100
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
        double reuse_score = reuse_is_possible ? (min_reuse_error_sqrt * 10000.0) : std::numeric_limits<double>::infinity();

        double best_1g_score = std::numeric_limits<double>::infinity();
        double best_1g_error_sqrt = std::numeric_limits<double>::infinity();
        std::vector<int> best_1g_indices; 
        Color best_1g_mixed_color;
        int best_1g_m_val = 0;

        if (K_num_colors > 0) {
            set<int> m_cands;
            rep(mi,ACTUAL_WELL_CAPACITY + 1){ // mの候補はウェル容量まで
                if(2*mi*(1000-i_target) <=remaining_ops) {
                    m_cands.insert(mi);
                }
            }
            if(num_blank_slots > 0) {
                rep(mi,ACTUAL_WELL_CAPACITY + 1){
                    m_cands.insert(mi);
                }
            }
            for (int m_trial : m_cands) {
                if (m_trial == 0 || m_trial > ACTUAL_WELL_CAPACITY) continue;
                if (m_trial + 1 > remaining_ops) continue; 

                std::vector<int> current_m_indices_candidate;
                double current_m_err_sqrt_candidate = std::numeric_limits<double>::infinity();
                Color current_m_mixed_c_candidate;
                bool candidate_found = false;

                // auto qp_res_tuple = select_colors_qp_approx(own_colors, mixed_target_color_obj, m_trial);
                // if (!get<0>(qp_res_tuple).empty() || m_trial == 0) {
                //     current_m_indices_candidate = get<0>(qp_res_tuple);
                //     current_m_err_sqrt_candidate = get<1>(qp_res_tuple);
                //     current_m_mixed_c_candidate = get<2>(qp_res_tuple);
                //     candidate_found = true;
                // }

                long long combinations_approx = 1;
                if (K_num_colors > 0 && m_trial > 0) {
                    for(int i=0; i<m_trial; ++i) {
                        if (__builtin_mul_overflow(combinations_approx, (K_num_colors+m_trial-1-i), &combinations_approx)) {combinations_approx = -1; break;}
                        if (combinations_approx < 0) break; 
                        combinations_approx /= (i+1);
                         if (combinations_approx < 0 || combinations_approx > 30000) {combinations_approx = -1; break;}
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
                
                SAParams sa_p = {1.0, 1e-4, 0.97, std::max(10, K_num_colors / (m_trial+1) + 5), std::max(20, K_num_colors*2/(m_trial+1) + 10)};
                std::vector<int> sa_initial_seed = candidate_found ? current_m_indices_candidate : std::vector<int>();
                
                auto sa_res = select_colors_sa_timed(own_colors, mixed_target_color_obj, m_trial, sa_p, method_time_limit_ms, sa_initial_seed);
                if (!candidate_found || get<1>(sa_res) < current_m_err_sqrt_candidate) {
                     if (!get<0>(sa_res).empty() || m_trial == 0) { 
                        current_m_indices_candidate = get<0>(sa_res);
                        current_m_err_sqrt_candidate = get<1>(sa_res);
                        current_m_mixed_c_candidate = get<2>(sa_res);
                        candidate_found = true;
                     }
                }
                
                if (!candidate_found && m_trial > 0 && !own_colors.empty()) {
                    current_m_indices_candidate.assign(m_trial, 0);
                    current_m_mixed_c_candidate = get_mixed_color_from_indices(current_m_indices_candidate, own_colors, m_trial);
                    current_m_err_sqrt_candidate = calculate_error_sqrt(current_m_mixed_c_candidate, current_target_color_obj);
                    candidate_found = true;
                } else if (!candidate_found) {
                    continue; 
                }

                double score_candidate = current_m_err_sqrt_candidate * 10000.0 + discount_factor * D_cost_factor * (m_trial-1); 
                if (score_candidate < best_1g_score) {
                    best_1g_score = score_candidate;
                    best_1g_indices = current_m_indices_candidate;
                    best_1g_mixed_color = current_m_mixed_c_candidate;
                    best_1g_error_sqrt = current_m_err_sqrt_candidate;
                    best_1g_m_val = best_1g_indices.size(); 
                }
            }
        }
        
        num_blank_slots = 0;
        for (int slot_idx = 0; slot_idx < NUM_TOTAL_SLOTS; ++slot_idx) {
            if (slots[slot_idx].remaining_grams == 0) num_blank_slots++;
        }

        bool chose_reuse = false;
        bool chose_new_1g = false;
        
        double reuse_threshold_factor = 1.0; 

        if (K_num_colors == 0 && reuse_is_possible) {
             if (1 <= remaining_ops) chose_reuse = true;
        } else if (reuse_is_possible && (best_1g_indices.empty() || (reuse_score < best_1g_score * reuse_threshold_factor))) {
             if (1 <= remaining_ops) { 
                chose_reuse = true;
             }
        }
        
        if (!chose_reuse && !best_1g_indices.empty() && K_num_colors > 0) {
            int ops_for_1g = best_1g_m_val + 1; 
            if (ops_for_1g <= remaining_ops) {
                chose_new_1g = true;
            }
        }
        
        if (chose_reuse) {
            auto coords = get_coords_from_slot_idx(best_reuse_slot_idx);
            int actual_row = coords.first;
            int actual_col = coords.second;
            operations_log.push_back("2 " + to_string(actual_row) + " " + to_string(actual_col));
            cerr <<"reuse score: " << reuse_score << " error sqrt: " << min_reuse_error_sqrt << std::endl;
            slots[best_reuse_slot_idx].remaining_grams--;
            remaining_ops--;
        }
        else if (chose_new_1g) { 
            int slot_to_use = -1;
            int best_val_for_slot = -1; 
            
            for(int offset = 0; offset < NUM_TOTAL_SLOTS; ++offset) {
                int temp_slot = (current_slot_idx_preference + offset) % NUM_TOTAL_SLOTS;
                int current_val = ACTUAL_WELL_CAPACITY-slots[temp_slot].remaining_grams;
                
                if (current_val > best_val_for_slot) {
                    best_val_for_slot = current_val;
                    slot_to_use = temp_slot;
                }
            }
            if(slot_to_use == -1) slot_to_use = current_slot_idx_preference; // Fallback

            auto coords = get_coords_from_slot_idx(slot_to_use);
            int actual_row = coords.first;
            int actual_col = coords.second;

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
            cerr << "New1g score: " << best_1g_score << " error sqrt: " << best_1g_error_sqrt;
            cerr << ", m_val: " << best_1g_m_val << endl;
        }

        end_target_iteration:;
    }

    for(const auto& op_str : operations_log) {
        std::cout << op_str << std::endl;
    }
    
    return 0;
}