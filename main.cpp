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
std::mt19937 rng(42); // Fixed seed for reproducibility

// --- Helper Structs and Functions ---
struct Color {
    double r, g, b;
    Color(double r_ = 0.0, double g_ = 0.0, double b_ = 0.0) : r(r_), g(g_), b(b_) {}

    Color operator+(const Color& other) const { return {r + other.r, g + other.g, b + other.b}; }
    Color operator-(const Color& other) const { return {r - other.r, g - other.g, b - other.b}; }
    Color operator*(double scalar) const { return {r * scalar, g * scalar, b * scalar}; }
    Color operator/(double scalar) const {
        if (std::abs(scalar) < 1e-9) return {0,0,0};
        return {r / scalar, g / scalar, b / scalar};
    }
    Color& operator+=(const Color& other) { r += other.r; g += other.g; b += other.b; return *this; }
    double dot(const Color& other) const { return r * other.r + g * other.g + b * other.b; }
};

Color get_mixed_color_from_indices(const std::vector<int>& indices_to_mix, const std::vector<Color>& own_colors_list_ref, int m_val_eff) {
    if (indices_to_mix.empty() || m_val_eff == 0) return {0.0, 0.0, 0.0};
    Color sum_color = {0.0, 0.0, 0.0};
    for (int idx : indices_to_mix) {
        if (idx >=0 && static_cast<size_t>(idx) < own_colors_list_ref.size())
            sum_color += own_colors_list_ref[idx];
    }
    return sum_color / m_val_eff;
}

double calculate_error_sqrt(const Color& mixed_color, const Color& target_color) {
    Color diff = mixed_color - target_color;
    return std::sqrt(diff.dot(diff));
}

// --- Color Selection Algorithms ---

void combinations_with_replacement_recursive(
    const std::vector<Color>& own_colors_list, const Color& target_color_vec, int m_val, int start_index,
    std::vector<int>& current_combination, std::vector<int>& best_indices,
    double& best_err_val_sqrt, Color& best_mixed_color_tuple, bool& found_one) {

    if (current_combination.size() == static_cast<size_t>(m_val)) {
        Color mixed_color = get_mixed_color_from_indices(current_combination, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec);
        if (!found_one || current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_mixed_color_tuple = mixed_color;
            best_indices = current_combination;
        }
        found_one = true;
        return;
    }
    if (start_index >= static_cast<int>(own_colors_list.size())) return;

    for (size_t i = start_index; i < own_colors_list.size(); ++i) {
        current_combination.push_back(i);
        combinations_with_replacement_recursive(own_colors_list, target_color_vec, m_val, i,
                                                current_combination, best_indices, best_err_val_sqrt, best_mixed_color_tuple, found_one);
        current_combination.pop_back();
    }
}

std::tuple<std::vector<int>, double, Color> select_colors_exhaustive(
    const std::vector<Color>& own_colors_list, const Color& target_color_vec, int m_val) {
    double best_err_val_sqrt = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};
    std::vector<int> best_indices;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        return {best_indices, calculate_error_sqrt({0,0,0}, target_color_vec), {0,0,0}};
    }
    
    bool found_one = false;
    std::vector<int> current_combination;
    current_combination.reserve(m_val);

    combinations_with_replacement_recursive(own_colors_list, target_color_vec, m_val, 0,
                                            current_combination, best_indices, best_err_val_sqrt, best_mixed_color_tuple, found_one);
            
    if (!found_one && m_val > 0) { 
        best_indices.assign(m_val, 0); 
        best_mixed_color_tuple = get_mixed_color_from_indices(best_indices, own_colors_list, m_val);
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    }
    return {best_indices, best_err_val_sqrt, best_mixed_color_tuple};
}

std::tuple<std::vector<int>, double, Color> select_colors_rs(
    const std::vector<Color>& own_colors_list, const Color& target_color_vec, int m_val, int num_samples = 100) {
    double best_err_val_sqrt = std::numeric_limits<double>::infinity();
    Color best_mixed_color_tuple = {0.0, 0.0, 0.0};    
    std::vector<int> best_indices;
    size_t num_own_colors = own_colors_list.size();

    if (num_own_colors == 0 || m_val == 0) {
        return {best_indices, calculate_error_sqrt({0,0,0}, target_color_vec), {0,0,0}};
    }
    
    std::uniform_int_distribution<int> dist(0, static_cast<int>(num_own_colors) - 1);
    bool found_one = false;
    std::vector<int> current_indices(m_val);

    for (int s = 0; s < num_samples; ++s) {
        for (int i = 0; i < m_val; ++i) current_indices[i] = dist(rng);
        Color mixed_color = get_mixed_color_from_indices(current_indices, own_colors_list, m_val);
        double current_err_sqrt = calculate_error_sqrt(mixed_color, target_color_vec);
        
        if (!found_one || current_err_sqrt < best_err_val_sqrt) {
            best_err_val_sqrt = current_err_sqrt;
            best_indices = current_indices; 
            best_mixed_color_tuple = mixed_color;
        }
        found_one = true;
    }
    
    if (!found_one && m_val > 0) {
        best_indices.assign(m_val, 0);
        best_mixed_color_tuple = get_mixed_color_from_indices(best_indices, own_colors_list, m_val);
        best_err_val_sqrt = calculate_error_sqrt(best_mixed_color_tuple, target_color_vec);
    }
    return {best_indices, best_err_val_sqrt, best_mixed_color_tuple};
}

struct SAParams { double t_init, t_final, cool_rate; int iter_per_temp, rs_samples; };

std::tuple<std::vector<int>, double, Color> select_colors_sa_timed(
    const std::vector<Color>& own_colors, const Color& target, int m, const SAParams& sa_p, double time_limit_ms, std::vector<int> init_indices = {}) {
    
    size_t K = own_colors.size();
    auto start_time = std::chrono::steady_clock::now();

    if (m == 0 || K == 0) return {{}, calculate_error_sqrt({0,0,0}, target), {0,0,0}};

    std::vector<int> current_indices;
    double current_err;
    Color current_mixed_c;

    if (!init_indices.empty() && init_indices.size() == static_cast<size_t>(m)) {
        current_indices = init_indices;
    } else {
        auto rs_res = select_colors_rs(own_colors, target, m, sa_p.rs_samples);
        current_indices = get<0>(rs_res);
    }

    if (current_indices.empty() && m > 0) current_indices.assign(m, 0);
    
    current_mixed_c = get_mixed_color_from_indices(current_indices, own_colors, m);
    current_err = calculate_error_sqrt(current_mixed_c, target);

    std::vector<int> best_indices = current_indices;
    double best_err = current_err;
    Color best_mixed_c = current_mixed_c;

    double temp = sa_p.t_init;
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::uniform_int_distribution<int> idx_dist(0, m - 1);
    std::uniform_int_distribution<int> color_dist(0, K - 1);

    while (temp > sa_p.t_final) {
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double, std::milli>(now - start_time).count() > time_limit_ms) break;

        for (int iter = 0; iter < sa_p.iter_per_temp; ++iter) {
            auto neighbor_indices = current_indices;
            int change_pos = idx_dist(rng);
            int new_color_idx = color_dist(rng);
            if(K > 1) while(new_color_idx == neighbor_indices[change_pos]) new_color_idx = color_dist(rng);
            neighbor_indices[change_pos] = new_color_idx;

            Color neighbor_mixed_c = get_mixed_color_from_indices(neighbor_indices, own_colors, m);
            double neighbor_err = calculate_error_sqrt(neighbor_mixed_c, target);
            double delta_err = neighbor_err - current_err;

            if (delta_err < 0 || (temp > 1e-9 && unif(rng) < std::exp(-delta_err / temp))) {
                current_indices = neighbor_indices;
                current_err = neighbor_err;
                if (current_err < best_err) {
                    best_err = current_err;
                    best_indices = current_indices;
                    best_mixed_c = neighbor_mixed_c;
                }
            }
        }
        temp *= sa_p.cool_rate;
    }
    return {best_indices, best_err, best_mixed_c};
}
      
// --- Data Structures ---
struct SlotInfo {
    Color color;
    int remaining_grams;
    SlotInfo() : color({0,0,0}), remaining_grams(0) {}
};

struct PaletteConfig {
    string name;
    int well_capacity;
    int num_slots;
    int layout_type; // 0: Horizontal (10x40, 5x80), 1: Vertical (20x20)
};

// --- Main Logic ---
int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout << std::fixed << std::setprecision(8);

    auto overall_start_time_chrono = std::chrono::steady_clock::now();
    
    int N, K, H, T;
    double D;
    
    std::cin >> N >> K >> H >> T >> D;
    std::vector<Color> own_colors(K);
    for (int i = 0; i < K; ++i) std::cin >> own_colors[i].r >> own_colors[i].g >> own_colors[i].b;
    std::vector<Color> targets(H);
    for (int i = 0; i < H; ++i) std::cin >> targets[i].r >> targets[i].g >> targets[i].b;
        
    // ===== Dynamic Palette Layout Selection =====
    PaletteConfig config;
    double d_high_thresh = 3000.0, d_low_thresh = 250.0;
    int t_low_thresh = 10000, t_high_thresh = 30000;

    if (D < d_low_thresh && T > t_high_thresh) {
        config = {"20x20", 20, 20, 0};
    } else if (D > d_high_thresh || T < t_low_thresh) {
        config = {"5x80", 5, 80, 0};
    } else {
        config = {"10x40", 10, 40, 0};
    }
    
    cerr << "Strategy: " << config.name << " (D=" << D << ", T=" << T << ")" << endl;

    if (config.layout_type == 0) { // Horizontal wells
        for (int i = 0; i < N; ++i) { for (int j = 0; j < N - 1; ++j) cout << (j ? " " : "") << "1"; cout << endl; }
        int well_height = N * N / config.num_slots;
        for (int r = 0; r < N - 1; ++r) { bool close = ((r + 1) % well_height == 0); for (int j = 0; j < N; ++j) cout << (j ? " " : "") << (close ? "1" : "0"); cout << endl; }
    } else { // Vertical wells
        int well_width = N * N / config.num_slots;
        for (int i = 0; i < N; ++i) { for (int j = 0; j < N - 1; ++j) { bool close = ((j + 1) % well_width == 0); cout << (j ? " " : "") << (close ? "1" : "0"); } cout << endl; }
        for (int r = 0; r < N - 1; ++r) { for (int j = 0; j < N; ++j) cout << (j ? " " : "") << "1"; cout << endl; }
    }
    
    auto get_coords_from_slot_idx = [&](int slot_idx) {
        if (config.layout_type == 0) {
            int well_height = N * N / config.num_slots;
            int region_idx = slot_idx / N;
            int col = slot_idx % N;
            return make_pair(region_idx * well_height, col);
        } else {
            return make_pair(slot_idx, 0);
        }
    };
    
    std::vector<SlotInfo> slots(config.num_slots);
    int current_slot_idx_preference = 0;
    
    const double TOTAL_TIME_BUDGET_S = (K > 40 && H > 800 && N > 15) ? 2.85 : 2.95;
    int remaining_ops = T;
    std::vector<std::string> operations_log;
    int num_blank_slots = config.num_slots;

    // ======================================================================
    // ===== START OF THE CORE LOGIC LOOP (FROM YOUR PROVIDED CODE) =====
    // ======================================================================
    for (int i_target = 0; i_target < H; ++i_target) {
        if (remaining_ops <= 0 && i_target < H) { 
            int fallback_slot_idx = (current_slot_idx_preference + i_target) % config.num_slots;
            auto coords = get_coords_from_slot_idx(fallback_slot_idx);
            operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));
            continue;
        }
    
        // This part is kept as you provided
        double progress = static_cast<double>(i_target + 1) / H;
        double discount_factor = progress * 0.10;
        double discount_thres;
        if(D<1000){
            discount_thres = 140.0;
        }else{
            discount_thres=180;
        }
     
        discount_factor = discount_thres/D;
        
        auto current_time_chrono = std::chrono::steady_clock::now();
        double time_spent_s = std::chrono::duration<double>(current_time_chrono - overall_start_time_chrono).count();
        double remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s;
        double avg_time_per_remaining_target_ms = (H - i_target > 0) ? (remaining_time_budget_s / (H - i_target) * 1000.0) : 2.0;
        
        double method_time_limit_ms = std::max(0.05, std::min(avg_time_per_remaining_target_ms , 15.0)); 
        if (remaining_time_budget_s < 0.05 * (H - i_target) && i_target < H - 1) method_time_limit_ms = 0.05;

        Color current_target_color_obj = targets[i_target];
        
        double min_error_sqrt = std::numeric_limits<double>::infinity();
       
        vector<pair<double,int>> mix_color_candidates;
        Color mixed_target_color_obj;
        int mix_num;
        if(D>5000){
            mix_num = 5;
        }else if(D>3000){
            mix_num = 4;
        }else if (D>1000){
            mix_num = 3;
        }else if (D>500){
            mix_num = 2;
        }else{
            mix_num = 1;
        }
        mix_num=min(mix_num, H - i_target);
        mix_num=min(mix_num, config.well_capacity);
        vector<double>weight(mix_num);
        double sum_weight = 0.0;
        rep(j, mix_num){
            // weight[j] = pow(0.9, j);
            // weight[j] = pow(0.95, j); // Slightly more aggressive discounting
            // weight[j] = pow(1.05, j); // Slightly more aggressive discounting
            // weight[j] = 1;
            sum_weight += weight[j];
        }
        rep(j, mix_num) weight[j] /= sum_weight;


        FOR(j,i_target,min(H, i_target +150*mix_num)){
            auto error_sqrt_j= calculate_error_sqrt(current_target_color_obj, targets[j]);
            mix_color_candidates.push_back({error_sqrt_j, j});
        }
        sort(mix_color_candidates.begin(), mix_color_candidates.end());
        int actual_mix_num = 0;
        rep(j,mix_num){
            // if(mix_color_candidates[j].first <0.3){
                mixed_target_color_obj+= targets[mix_color_candidates[j].second];
                actual_mix_num++;
            // }
        }
        mixed_target_color_obj = mixed_target_color_obj / (double)actual_mix_num;

        int best_reuse_slot_idx = -1;
        double min_reuse_error_sqrt = std::numeric_limits<double>::infinity();
        
        for (int slot_idx = 0; slot_idx < config.num_slots; ++slot_idx) {
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

        if (K > 0) {
            set<int> m_cands;
            rep(mi, config.well_capacity + 1){
                if(2*mi*(H-i_target) <= remaining_ops) m_cands.insert(mi);
            }
            if(num_blank_slots > 0) {
                rep(mi, config.well_capacity + 1) m_cands.insert(mi);
            }

            for (int m_trial : m_cands) {
                if (m_trial == 0 || m_trial > config.well_capacity) continue;
                if (m_trial + 1 > remaining_ops) continue; 

                std::vector<int> current_m_indices_candidate;
                double current_m_err_sqrt_candidate = std::numeric_limits<double>::infinity();
                Color current_m_mixed_c_candidate;
                bool candidate_found = false;

                long long combinations_approx = 1;
                int comb_max=30000;
                if (K > 0 && m_trial > 0) {
                    for(int i_comb=0; i_comb<m_trial; ++i_comb) {
                        if (__builtin_mul_overflow(combinations_approx, (K+m_trial-1-i_comb), &combinations_approx)) {combinations_approx = -1; break;}
                        if (combinations_approx < 0) break; 
                        combinations_approx /= (i_comb+1);
                         if (combinations_approx < 0 || combinations_approx > comb_max) {combinations_approx = -1; break;}
                    }
                } else if (m_trial == 0) combinations_approx = 1;

                if (combinations_approx != -1 && combinations_approx <= comb_max) { 
                    auto exhaustive_res = select_colors_exhaustive(own_colors, mixed_target_color_obj, m_trial);
                    if (!candidate_found || get<1>(exhaustive_res) < current_m_err_sqrt_candidate) {
                        current_m_indices_candidate = get<0>(exhaustive_res);
                        current_m_err_sqrt_candidate = get<1>(exhaustive_res);
                        current_m_mixed_c_candidate = get<2>(exhaustive_res);
                        candidate_found = true;
                    }
                }
                
                SAParams sa_p = {1.0, 1e-4, 0.97, std::max(10, K / (m_trial+1) + 5), std::max(20, K*2/(m_trial+1) + 10)};
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
                
                // ===== START: SCORE CALCULATION FIX =====
                double score_candidate;
                if(D<discount_thres){
                    score_candidate = current_m_err_sqrt_candidate * 10000.0 + D*(m_trial - 1); 
                }else{
                    score_candidate = current_m_err_sqrt_candidate * 10000.0 + D * (m_trial - 1) *  (discount_factor);
                }
                // ===== END: SCORE CALCULATION FIX =====

                if (score_candidate < best_1g_score) {
                    best_1g_score = score_candidate;
                    best_1g_indices = current_m_indices_candidate;
                    best_1g_mixed_color = current_m_mixed_c_candidate;
                    best_1g_error_sqrt = current_m_err_sqrt_candidate;
                    best_1g_m_val = m_trial; 
                }
            }
        }
        
        num_blank_slots = 0;
        for (int slot_idx = 0; slot_idx < config.num_slots; ++slot_idx) {
            if (slots[slot_idx].remaining_grams == 0) num_blank_slots++;
        }

        bool chose_reuse = false;
        bool chose_new_1g = false;
        
        double reuse_threshold_factor = 1.0; 

        if (K == 0 && reuse_is_possible) {
             if (1 <= remaining_ops) chose_reuse = true;
        } else if (reuse_is_possible && (best_1g_indices.empty() || (reuse_score < best_1g_score * reuse_threshold_factor))) {
             if (1 <= remaining_ops) { 
                chose_reuse = true;
             }
        }
        
        if (!chose_reuse && !best_1g_indices.empty() && K > 0) {
            int ops_for_1g = best_1g_m_val + 1; 
            if (ops_for_1g <= remaining_ops) {
                chose_new_1g = true;
            }
        }
        
        if (chose_reuse) {
            auto coords = get_coords_from_slot_idx(best_reuse_slot_idx);
            operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));
            cerr <<"reuse score: " << reuse_score << " error sqrt: " << min_reuse_error_sqrt << std::endl;
            slots[best_reuse_slot_idx].remaining_grams--;
            remaining_ops--;
        }
        else if (chose_new_1g) { 
            int slot_to_use = -1;
            int best_val_for_slot = -1; 
            double min_sum_diff = std::numeric_limits<double>::infinity();

            for(int offset = 0; offset < config.num_slots; ++offset) {
                int temp_slot = (current_slot_idx_preference + offset) % config.num_slots;
                // 他の残っているものとの違いが一番小さいスロットを選ぶ
                if (slots[temp_slot].remaining_grams == 0) {
                    slot_to_use = temp_slot;
                    break; 
                }
                double sum_diff = std::numeric_limits<double>::infinity();
                rep(j, config.num_slots){
                    if(j == temp_slot ) continue;
                    auto diff = calculate_error_sqrt(slots[temp_slot].color, slots[j].color);
                    sum_diff+= diff;
                }
                if (sum_diff < min_sum_diff) {
                    min_sum_diff = sum_diff;
                    slot_to_use = temp_slot;
                    // best_val_for_slot = slots[temp_slot].remaining_grams;
                }
            }
            if(slot_to_use == -1) slot_to_use = current_slot_idx_preference;

            auto coords = get_coords_from_slot_idx(slot_to_use);
            int discards_needed = slots[slot_to_use].remaining_grams;
            if (discards_needed + best_1g_m_val + 1 > remaining_ops) { 
                 chose_new_1g = false;
                 if(reuse_is_possible && 1 <= remaining_ops) {
                    auto reuse_coords = get_coords_from_slot_idx(best_reuse_slot_idx);
                    operations_log.push_back("2 " + to_string(reuse_coords.first) + " " + to_string(reuse_coords.second));
                    slots[best_reuse_slot_idx].remaining_grams--; remaining_ops--;
                 } else { 
                    auto last_coords = get_coords_from_slot_idx(i_target % config.num_slots);
                    operations_log.push_back("2 " + to_string(last_coords.first) + " " + to_string(last_coords.second));
                    if(remaining_ops > 0) remaining_ops--;
                 }

            } else {
                for(int d=0; d<discards_needed; ++d) operations_log.push_back("3 " + to_string(coords.first) + " " + to_string(coords.second));
                remaining_ops -= discards_needed;

                for(int tube_original_idx : best_1g_indices) operations_log.push_back("1 " + to_string(coords.first) + " " + to_string(coords.second) + " " + to_string(tube_original_idx));
                operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));

                slots[slot_to_use].color = best_1g_mixed_color;
                slots[slot_to_use].remaining_grams = best_1g_m_val - 1; 

                current_slot_idx_preference = (slot_to_use + 1) % config.num_slots;
                remaining_ops -= (best_1g_m_val + 1);
            }
            if (chose_new_1g) {
                cerr << "New1g score: " << best_1g_score << " error sqrt: " << best_1g_error_sqrt;
                cerr << ", m_val: " << best_1g_m_val << endl;
            }
        } else { // Fallback
            if (reuse_is_possible && 1 <= remaining_ops) {
                auto coords = get_coords_from_slot_idx(best_reuse_slot_idx);
                operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));
                slots[best_reuse_slot_idx].remaining_grams--;
                remaining_ops--;
            } else {
                auto coords = get_coords_from_slot_idx(i_target % config.num_slots);
                operations_log.push_back("2 " + to_string(coords.first) + " " + to_string(coords.second));
                if (remaining_ops > 0) remaining_ops--;
            }
        }
    }
    // ====================================================================
    // ===== END OF THE CORE LOGIC LOOP =====
    // ====================================================================

    for(const auto& op_str : operations_log) {
        std::cout << op_str << std::endl;
    }
    
    return 0;
}