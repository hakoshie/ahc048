import sys
import math
import random
# import copy
import itertools
import time
import numpy as np

# --- Helper Functions (変更なし) ---
def get_mixed_color(indices_to_mix, own_colors_list_ref, m_val_eff):
    # ... (省略) ...
    if not indices_to_mix or m_val_eff == 0:
        return (0.0, 0.0, 0.0)
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    for idx in indices_to_mix:
        color = own_colors_list_ref[idx]
        sum_r += color[0]
        sum_g += color[1]
        sum_b += color[2]
    return (sum_r / m_val_eff, sum_g / m_val_eff, sum_b / m_val_eff)

def calculate_error(mixed_color_tuple, target_color_tuple):
    # ... (省略) ...
    err_sq = 0.0
    for i in range(3):
        err_sq += (mixed_color_tuple[i] - target_color_tuple[i])**2
    return math.sqrt(err_sq)

# --- Color Selection Strategies (変更なし) ---
def select_colors_exhaustive(own_colors_list, target_color_vec, m_val):
    # ... (省略、戻り値に best_mixed_color_tuple を含む) ...
    best_err_val = float('inf')
    best_mixed_color_tuple = (0.0, 0.0, 0.0)
    best_indices_to_drop = []
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0 or m_val == 0:
        default_mixed_color = (0.0,0.0,0.0)
        if m_val > 0 and num_own_colors > 0: default_mixed_color = own_colors_list[0]
        err = calculate_error(default_mixed_color, target_color_vec)
        return ([0] * m_val if num_own_colors > 0 and m_val > 0 else []), err, default_mixed_color

    found_one = False
    for current_indices_tuple in itertools.combinations_with_replacement(range(num_own_colors), m_val):
        mixed_color = get_mixed_color(current_indices_tuple, own_colors_list, m_val)
        current_err = calculate_error(mixed_color, target_color_vec)
        found_one = True
        if current_err < best_err_val:
            best_err_val = current_err
            best_mixed_color_tuple = mixed_color
            best_indices_to_drop = list(current_indices_tuple)
            
    if not found_one and num_own_colors > 0 and m_val > 0: 
        best_indices_to_drop = [0] * m_val
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val)
        best_err_val = calculate_error(best_mixed_color_tuple, target_color_vec)
    elif not found_one : 
        pass
    return best_indices_to_drop, best_err_val, best_mixed_color_tuple

def select_colors_rs(own_colors_list, target_color_vec, m_val, num_samples=100):
    # ... (省略、戻り値に best_mixed_color_tuple を含む) ...
    best_err_val = float('inf')
    best_mixed_color_tuple = (0.0, 0.0, 0.0)    
    best_indices_to_drop = []
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0 or m_val == 0:
        default_mixed_color = (0.0,0.0,0.0)
        if m_val > 0 and num_own_colors > 0: default_mixed_color = own_colors_list[0]
        err = calculate_error(default_mixed_color, target_color_vec)
        return ([0] * m_val if num_own_colors > 0 and m_val > 0 else []), err, default_mixed_color
    
    found_one = False
    for _ in range(num_samples):
        current_indices = [random.randint(0, num_own_colors - 1) for _ in range(m_val)]
        mixed_color = get_mixed_color(current_indices, own_colors_list, m_val)
        current_err = calculate_error(mixed_color, target_color_vec)
        found_one = True
        if current_err < best_err_val:
            best_err_val = current_err
            best_indices_to_drop = current_indices 
            best_mixed_color_tuple = mixed_color
    
    if not found_one and num_own_colors > 0 and m_val > 0:
        best_indices_to_drop = [0] * m_val
        best_mixed_color_tuple = get_mixed_color(best_indices_to_drop, own_colors_list, m_val)
        best_err_val = calculate_error(best_mixed_color_tuple, target_color_vec)
    elif not found_one:
        pass
    return best_indices_to_drop, best_err_val, best_mixed_color_tuple


def select_colors_sa_timed(own_colors_list, target_color_vec, m_val, sa_params, time_limit_ms):
    # ... (省略、戻り値に best_mixed_color_tuple を含む) ...
    num_own_colors = len(own_colors_list)
    start_time_sa = time.perf_counter()

    default_mixed_color = (0.0,0.0,0.0)
    if m_val > 0 and num_own_colors > 0: default_mixed_color = own_colors_list[0]
    if num_own_colors == 0 or m_val == 0:
        err = calculate_error(default_mixed_color, target_color_vec)
        return ([0] * m_val if num_own_colors > 0 and m_val > 0 else []), err, default_mixed_color

    current_indices, current_error, current_mixed_color = select_colors_rs(
        own_colors_list, target_color_vec, m_val, num_samples=sa_params.get('initial_rs_samples', 10)
    )
            
    best_indices = current_indices[:] 
    best_error = current_error
    best_mixed_color_tuple = current_mixed_color
    
    current_temp = sa_params['initial_temp']
    final_temp = sa_params['final_temp']
    cooling_rate = sa_params['cooling_rate']
    iterations_per_temp = sa_params['iterations_per_temp']

    while current_temp > final_temp:
        if (time.perf_counter() - start_time_sa) * 1000 > time_limit_ms: break

        for iter_idx in range(iterations_per_temp):
            if (time.perf_counter() - start_time_sa) * 1000 > time_limit_ms: break 

            if not current_indices: 
                if m_val > 0: current_indices = [0] * m_val 
                else: break 

            neighbor_indices = current_indices[:] 
            
            idx_to_change = random.randint(0, m_val - 1)
            new_color_idx = random.randint(0, num_own_colors - 1)
            if num_own_colors > 1:
                while new_color_idx == neighbor_indices[idx_to_change]:
                    new_color_idx = random.randint(0, num_own_colors - 1)
            neighbor_indices[idx_to_change] = new_color_idx

            neighbor_mixed_color = get_mixed_color(neighbor_indices, own_colors_list, m_val)
            neighbor_error = calculate_error(neighbor_mixed_color, target_color_vec)
            
            delta_error = neighbor_error - current_error

            if delta_error < 0:
                current_indices = neighbor_indices
                current_error = neighbor_error
                current_mixed_color = neighbor_mixed_color
                if current_error < best_error:
                    best_error = current_error
                    best_indices = current_indices[:]
                    best_mixed_color_tuple = current_mixed_color
            else:
                if current_temp > 1e-9: 
                    acceptance_probability = math.exp(-delta_error / current_temp)
                    if random.random() < acceptance_probability:
                        current_indices = neighbor_indices
                        current_error = neighbor_error
                        current_mixed_color = neighbor_mixed_color
        current_temp *= cooling_rate
            
    return best_indices, best_error, best_mixed_color_tuple

# --- Main Logic ---
def main():
    overall_start_time = time.perf_counter()
    
    data = sys.stdin.read().split()
    if not data: return
    
    ptr = 0
    N_grid_size = int(data[ptr]); ptr+=1 # グリッドの1辺のサイズ (例: 20)
    K_num_colors = int(data[ptr]); ptr+=1
    H_num_targets = int(data[ptr]); ptr+=1
    T_total_ops_limit = int(data[ptr]); ptr+=1
    D_cost_factor = float(data[ptr]); ptr+=1
    
    own_colors = []
    for _ in range(K_num_colors):
        own_colors.append((float(data[ptr]), float(data[ptr+1]), float(data[ptr+2])))
        ptr += 3
        
    targets = []
    for _ in range(H_num_targets):
        targets.append((float(data[ptr]), float(data[ptr+1]), float(data[ptr+2])))
        ptr += 3
        
    # --- M_val 決定ロジック (変更なし) ---
    # (前の回答のM_val決定ロジックをここにペースト)
    possible_m_vals_by_ops = []
    for m_cand in range(1, N_grid_size + 1): 
        if 2 * m_cand * H_num_targets <= T_total_ops_limit:
            possible_m_vals_by_ops.append(m_cand)
        else:
            break
    if not possible_m_vals_by_ops: possible_m_vals_by_ops = [1]

    m_val_candidates_to_test = []
    for 대표_m in [1, 2, 3, 4, 5, 7, 10, 15, 20]: # 1も候補に
        if 대표_m <= possible_m_vals_by_ops[-1]: 
            m_val_candidates_to_test.append(대표_m)
    if possible_m_vals_by_ops[-1] not in m_val_candidates_to_test:
        m_val_candidates_to_test.append(possible_m_vals_by_ops[-1])
    m_val_candidates_to_test = sorted(list(set(m_val_candidates_to_test)))
    if not m_val_candidates_to_test: m_val_candidates_to_test = [1]

    best_estimated_score = float('inf')
    chosen_M_val = m_val_candidates_to_test[0]
    N_TRIALS_FOR_M_VAL_EST = max(1, min(10, H_num_targets // 25)) 
    RS_SAMPLES_FOR_M_VAL_EST = 20
    EXHAUSTIVE_THRESHOLD_FOR_M_VAL_EST = 4

    if len(m_val_candidates_to_test) > 1:
        for m_cand_val in m_val_candidates_to_test:
            current_sum_err = 0.0
            for i_trial in range(N_TRIALS_FOR_M_VAL_EST):
                target_idx_for_eval = (i_trial * (H_num_targets // N_TRIALS_FOR_M_VAL_EST if N_TRIALS_FOR_M_VAL_EST > 0 else 1)) % H_num_targets
                if m_cand_val <= EXHAUSTIVE_THRESHOLD_FOR_M_VAL_EST:
                    _, err, _ = select_colors_exhaustive(own_colors, targets[target_idx_for_eval], m_cand_val)
                else:
                    _, err, _ = select_colors_rs(own_colors, targets[target_idx_for_eval], m_cand_val, num_samples=RS_SAMPLES_FOR_M_VAL_EST)
                current_sum_err += err
            avg_err = current_sum_err / N_TRIALS_FOR_M_VAL_EST if N_TRIALS_FOR_M_VAL_EST > 0 else float('inf')
            cost_V = D_cost_factor * (m_cand_val -1) * H_num_targets 
            cost_E = round(10000 * avg_err * H_num_targets) 
            current_estimated_score = cost_V + cost_E
            if current_estimated_score < best_estimated_score:
                best_estimated_score = current_estimated_score
                chosen_M_val = m_cand_val
    else:
        chosen_M_val = m_val_candidates_to_test[0]
    # print(f"Selected M_val: {chosen_M_val}", file=sys.stderr)

    SA_PARAMS_MAIN = {
        'initial_rs_samples': 10,
        'initial_temp': 0.1,   
        'final_temp': 1e-4,
        'cooling_rate': 0.97, 
        'iterations_per_temp': max(3, chosen_M_val // 3)
    }
    EXHAUSTIVE_THRESHOLD_M_VAL_MAIN = 4

    # --- ★ 初期仕切り出力の変更 ---
    # 縦仕切りは全て1 (閉じる)
    for _ in range(N_grid_size): print(" ".join(["1"] * (N_grid_size - 1)))
    # 横仕切り: 9行目と10行目の間 (インデックス9の仕切り) のみ1、他は0
    # N_grid_size行なので、N_grid_size-1 本の横仕切りがある。インデックスは 0 から N_grid_size-2。
    # 9行目と10行目の間の仕切りは h_9,j なので、インデックス9の横仕切りセット。
    # 例: N=20の場合、h_0,j ... h_18,j
    HORIZONTAL_DIVIDER_ROW_INDEX = 9 # このインデックスの横仕切りを閉じる
    for r_h_idx in range(N_grid_size - 1):
        if r_h_idx == HORIZONTAL_DIVIDER_ROW_INDEX:
            print(" ".join(["1"] * N_grid_size)) # この行の仕切りは全て閉じる
        else:
            print(" ".join(["0"] * N_grid_size)) # 他の仕切りは全て開く
    
    # --- 状態管理の変更 ---
    # 各列の上半分と下半分を区別して管理
    # 計 2 * N_grid_size 個のスロットがあるイメージ
    # スロットインデックス: 0..N-1 (上半分パレットの列0..N-1)
    #                    N..2N-1 (下半分パレットの列0..N-1)
    NUM_TOTAL_SLOTS = 2 * N_grid_size
    remaining_paint_in_slot = [0] * NUM_TOTAL_SLOTS
    current_colors_in_slot = [(0.0, 0.0, 0.0)] * NUM_TOTAL_SLOTS
    
    # 各スロットを順番に使うためのインデックス
    current_slot_idx_to_fill = 0 
    
    # --- メインループ ---
    TOTAL_TIME_BUDGET_S = 2.85
    # print(f"H={H_num_targets}, Chosen M_val={chosen_M_val}, Total Slots={NUM_TOTAL_SLOTS}", file=sys.stderr)

    for i_target in range(H_num_targets):
        time_spent_s = time.perf_counter() - overall_start_time
        remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s
        
        avg_time_per_remaining_target_ms = (remaining_time_budget_s / (H_num_targets - i_target) * 1000) if H_num_targets - i_target > 0 else 0
        sa_time_limit_ms_for_this_target = max(0.05, min(avg_time_per_remaining_target_ms * 0.85, 2.7))
        if remaining_time_budget_s < 0.05 and i_target < H_num_targets -1: sa_time_limit_ms_for_this_target = 0.05
        
        target_color = targets[i_target]

        # --- ★ 既存の絵の具を再利用するかの判定 (全スロット対象) ---
        best_reuse_slot_idx = -1
        min_reuse_error = float('inf')

        for slot_idx_check in range(NUM_TOTAL_SLOTS):
            if remaining_paint_in_slot[slot_idx_check] > 0:
                error_if_reused = calculate_error(current_colors_in_slot[slot_idx_check], target_color)
                if error_if_reused < min_reuse_error:
                    min_reuse_error = error_if_reused
                    best_reuse_slot_idx = slot_idx_check
        
        equivalent_d_cost_as_error = (chosen_M_val ) * D_cost_factor / 10000.0 
        # reuse_benefit_threshold = equivalent_d_cost_as_error * 0.8 # Dコストの80%以下の誤差なら (要調整)
        # または、非常に小さい固定の誤差閾値とDコスト比較の組み合わせ
        # 例: 誤差が0.01以下なら無条件で再利用、そうでなければDコストと比較
        use_existing_paint = False
        if best_reuse_slot_idx != -1:
            if min_reuse_error < 0.005 : # 非常に小さい誤差なら優先的に再利用
                 use_existing_paint = True
            elif min_reuse_error < equivalent_d_cost_as_error : # Dコストを払うより得なら再利用
                 use_existing_paint = True


        if use_existing_paint:
            # print(f"T{i_target}: Reusing slot {best_reuse_slot_idx}, Err {min_reuse_error:.3f}", file=sys.stderr)
            actual_row_ops = 0 if best_reuse_slot_idx < N_grid_size else HORIZONTAL_DIVIDER_ROW_INDEX + 1
            actual_col_ops = best_reuse_slot_idx % N_grid_size
            print(f"2 {actual_row_ops} {actual_col_ops}")
            remaining_paint_in_slot[best_reuse_slot_idx] -= 1
            # current_slot_idx_to_fill は進めない (再利用なので新しいスロットは使ってない)
            continue 

        # --- 新規作成 ---
        # 次に使うスロットを決定 (単純なローテーション)
        slot_to_create_in = current_slot_idx_to_fill
        
        actual_row_ops_create = 0 if slot_to_create_in < N_grid_size else HORIZONTAL_DIVIDER_ROW_INDEX + 1
        actual_col_ops_create = slot_to_create_in % N_grid_size

        # print(f"T{i_target}: New in slot {slot_to_create_in} (r={actual_row_ops_create}, c={actual_col_ops_create})", file=sys.stderr)
        for _ in range(remaining_paint_in_slot[slot_to_create_in]):
            print(f"3 {actual_row_ops_create} {actual_col_ops_create}")
        remaining_paint_in_slot[slot_to_create_in] = 0
            
        selected_indices = []
        selected_mixed_color = (0,0,0)

        if chosen_M_val <= EXHAUSTIVE_THRESHOLD_M_VAL_MAIN:
            selected_indices, _, selected_mixed_color = select_colors_exhaustive(own_colors, target_color, chosen_M_val)
        else:
            selected_indices, _, selected_mixed_color = select_colors_sa_timed(own_colors, target_color, chosen_M_val,
                                                       SA_PARAMS_MAIN, sa_time_limit_ms_for_this_target)
        
        if not selected_indices and K_num_colors > 0: # Should be populated
            selected_indices = [0] * chosen_M_val
            selected_mixed_color = get_mixed_color(selected_indices, own_colors, chosen_M_val)

        current_colors_in_slot[slot_to_create_in] = selected_mixed_color

        for c_idx in selected_indices:
            print(f"1 {actual_row_ops_create} {actual_col_ops_create} {c_idx}")
            
        print(f"2 {actual_row_ops_create} {actual_col_ops_create}")
        remaining_paint_in_slot[slot_to_create_in] = chosen_M_val - 1
        
        current_slot_idx_to_fill = (current_slot_idx_to_fill + 1) % NUM_TOTAL_SLOTS
    
    # final_time = time.perf_counter() - overall_start_time
    # print(f"Total time: {final_time:.3f}s", file=sys.stderr)

if __name__ == "__main__":
    main()