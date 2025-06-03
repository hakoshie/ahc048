import sys
import math
import random
import itertools
import time
import numpy as np

# --- Helper Functions ---
def get_mixed_color(indices_to_mix, own_colors_list_ref, m_val_eff):
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
    err_sq = 0.0
    for i in range(3):
        err_sq += (mixed_color_tuple[i] - target_color_tuple[i])**2
    return math.sqrt(err_sq)

# --- Color Selection Strategies ---
def select_colors_exhaustive(own_colors_list, target_color_vec, m_val):
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
    elif not found_one: 
        pass
    return best_indices_to_drop, best_err_val, best_mixed_color_tuple

def select_colors_rs(own_colors_list, target_color_vec, m_val, num_samples=100):
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
def select_colors_gd(own_colors_list, target_color_vec, m_val, D_cost_factor_unused):
    num_own_colors = len(own_colors_list)

    # Handle edge cases
    # if m_val == 0:
    #     # No colors to mix, result is black (or transparent, depending on interpretation)
    #     # Error is calculated against target.
    #     empty_mix_color = (0.0, 0.0, 0.0)
    #     return [], calculate_error(empty_mix_color, target_color_vec), empty_mix_color
    
    # if num_own_colors == 0:
    #     # This case should ideally not be hit if K_num_colors >= 1 (a common constraint).
    #     # If it were possible, and m_val > 0, we cannot select any colors.
    #     empty_mix_color = (0.0, 0.0, 0.0)
    #     return [], calculate_error(empty_mix_color, target_color_vec), empty_mix_color

    current_selected_indices = []
    # The final mixed color and error will be determined after all m_val colors are selected.
    error = float('inf')
    for _iteration_num in range(m_val): # We intend to select m_val colors, one by one
        
        best_candidate_idx_for_this_step = -1
        # Initialize error for this step to a high value.
        min_error_this_step = error
        # best_resulting_mix_this_step = (0.0, 0.0, 0.0) # Not strictly needed to store here if recalculating at end

        # num_own_colors is fixed, so no need to check if own_colors_list is empty inside loop
        # if it was non-empty at the start.
        
        for i_candidate_color_in_own_list in range(num_own_colors):
            # Tentatively add the current candidate color's index to the list of already selected ones
            potential_new_indices = current_selected_indices + [i_candidate_color_in_own_list]
            
            # Calculate the mixed color of this potential new set of indices.
            # Ensure your get_mixed_color uses len(potential_new_indices) for averaging.
            # The third argument to get_mixed_color in your main code might be vestigial.
            potential_mixed_color = get_mixed_color(potential_new_indices, own_colors_list, len(potential_new_indices))
            
            potential_error = calculate_error(potential_mixed_color, target_color_vec)
            delta = min_error_this_step - potential_error
   
            if delta>0:
                min_error_this_step = potential_error
                best_candidate_idx_for_this_step = i_candidate_color_in_own_list
            print(f"Iteration {_iteration_num}, candidate {i_candidate_color_in_own_list}, potential_error={potential_error}, min_error_this_step={min_error_this_step}, delta={delta}", file=sys.stderr)
        delta = error - min_error_this_step
        if delta *10000 > D_cost_factor_unused:
            error = min_error_this_step
            current_selected_indices.append(best_candidate_idx_for_this_step)
            # Update the current selected indices to include the best candidate found in this step.
            # This is the index of the color that gives the minimum error when added.
            # If no candidate was found, best_candidate_idx_for_this_step remains -1.
            # In that case, we will handle it after the loop.
            print(f"Iteration {_iteration_num}, best candidate index: {best_candidate_idx_for_this_step}, min_error_this_step={min_error_this_step}", file=sys.stderr)
        else:
            break
                # best_resulting_mix_this_step = potential_mixed_color # Store if needed for incremental update
        
 
                 
    # After m_val iterations (or breaking if num_own_colors was 0):
    # The current_selected_indices list holds the chosen color indices.
    # Calculate the final mixed color and error from this list.
    # Ensure get_mixed_color uses len(current_selected_indices) for averaging.
    final_mixed_color = get_mixed_color(current_selected_indices, own_colors_list, len(current_selected_indices))
    final_error = calculate_error(final_mixed_color, target_color_vec)
    
    return current_selected_indices, final_error, final_mixed_color
# --- Main Logic ---
def main():
    overall_start_time = time.perf_counter()
    
    data = sys.stdin.read().split()
    if not data: return
    
    ptr = 0
    N_grid_size = int(data[ptr]); ptr+=1
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
        
    # --- 初期仕切り出力 ---
    for _ in range(N_grid_size): print(" ".join(["1"] * (N_grid_size - 1)))
    HORIZONTAL_DIVIDER_ROW_INDEX = 9
    for r_h_idx in range(N_grid_size - 1):
        if r_h_idx == HORIZONTAL_DIVIDER_ROW_INDEX:
            print(" ".join(["1"] * N_grid_size))
        else:
            print(" ".join(["0"] * N_grid_size))
    
    # --- 状態管理 ---
    NUM_TOTAL_SLOTS = 2 * N_grid_size
    remaining_paint_in_slot = [0] * NUM_TOTAL_SLOTS
    current_colors_in_slot = [(0.0, 0.0, 0.0)] * NUM_TOTAL_SLOTS
    current_slot_idx_to_fill = 0 
    m_cands=[]
    max_m_val = 0
    for m in range(1,11):
        if 2*m*1000<=T_total_ops_limit:
            max_m_val = m
            m_cands.append(m)
    # for i in range(1,3):
    #     m_cands.append(max(max_m_val // i,1))
    m_cands = sorted(set(m_cands))  # 重複を排除してソート
    # --- メインループ ---
    TOTAL_TIME_BUDGET_S = 2.85
    remaining_ops = T_total_ops_limit

    for i_target in range(H_num_targets):
        time_spent_s = time.perf_counter() - overall_start_time
        remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s
        
        if remaining_ops <= 0:
            best_reuse_slot_idx = -1
            min_reuse_error = float('inf')
            for slot_idx in range(NUM_TOTAL_SLOTS):
                if remaining_paint_in_slot[slot_idx] > 0:
                    err = calculate_error(current_colors_in_slot[slot_idx], targets[i_target])
                    if err < min_reuse_error:
                        min_reuse_error = err
                        best_reuse_slot_idx = slot_idx
            if best_reuse_slot_idx != -1:
                actual_row_ops = 0 if best_reuse_slot_idx < N_grid_size else HORIZONTAL_DIVIDER_ROW_INDEX + 1
                actual_col_ops = best_reuse_slot_idx % N_grid_size
                print(f"2 {actual_row_ops} {actual_col_ops}")
                remaining_paint_in_slot[best_reuse_slot_idx] -= 1
                remaining_ops -= 1
            continue
        
        avg_time_per_remaining_target_ms = (remaining_time_budget_s / (H_num_targets - i_target) * 1000) if H_num_targets - i_target > 0 else 0
        sa_time_limit_ms_for_this_target = max(0.01, min(avg_time_per_remaining_target_ms * 0.5, 2.7))
        # if remaining_time_budget_s < 0.05 and i_target < H_num_targets -1: 
        #     sa_time_limit_ms_for_this_target = 0.03
        
        target_color = targets[i_target]

        # 再利用可能なスロットを探す
        best_reuse_slot_idx = -1
        min_reuse_error = float('inf')
        for slot_idx in range(NUM_TOTAL_SLOTS):
            if remaining_paint_in_slot[slot_idx] > 0:
                err = calculate_error(current_colors_in_slot[slot_idx], target_color)
                if err < min_reuse_error:
                    min_reuse_error = err
                    best_reuse_slot_idx = slot_idx
        
        reuse_possible = (best_reuse_slot_idx != -1)
        
        # 新規作成の候補M_valを準備 (1〜5)
        m_candidate_list = m_cands[:]
        
        
        best_m_score = float('inf')
        best_m_val = 0
        best_m_indices = [0]
        best_m_mixed_color = (0,0,0)
        best_m_err = float('inf')
        
        if m_candidate_list:
            for m in m_candidate_list:
                if m <= 3:
                    indices, err, mixed_color = select_colors_exhaustive(own_colors, target_color, m)
                else:
                    # indices, err, mixed_color = select_colors_gd(own_colors, target_color, m, D_cost_factor)
                    indices, err, mixed_color = select_colors_sa_timed(
                        own_colors, target_color, m, 
                        sa_params={
                            'initial_temp': 1.0,
                            'final_temp': 1e-3,
                            'cooling_rate': 0.95,
                            'iterations_per_temp': 10,
                            'initial_rs_samples': 10
                        }, 
                        time_limit_ms=sa_time_limit_ms_for_this_target
                    )
                # for index in indices:
                #     print(f"{index}",file=sys.stderr,end=' ')
                # print(f"m={len(indices)} err={err}, len ={len(indices)}", file=sys.stderr)
                score_candidate = err * 10000 + D_cost_factor * (len(indices)-1 )
                
                if score_candidate < best_m_score:
                    print(f"UPDATED OPTM={len(indices)} score={score_candidate} err={err}", file=sys.stderr)
                    best_m_score = score_candidate
                    best_m_val = len(indices)
                    best_m_indices = indices
                    best_m_mixed_color = mixed_color
                    best_m_err = err

        # 再利用と新規作成を比較
        use_existing_paint = False
        if reuse_possible:
            score_reuse = min_reuse_error * 10000
            if min_reuse_error < 0.005 or score_reuse <= best_m_score:
                use_existing_paint = True

        if use_existing_paint:
            actual_row_ops = 0 if best_reuse_slot_idx < N_grid_size else HORIZONTAL_DIVIDER_ROW_INDEX + 1
            actual_col_ops = best_reuse_slot_idx % N_grid_size
            print(f"2 {actual_row_ops} {actual_col_ops}")
            remaining_paint_in_slot[best_reuse_slot_idx] -= 1
            remaining_ops -= 1
            continue 

        # 新規作成
        for sloti in range(len(remaining_paint_in_slot)):
            if remaining_paint_in_slot[sloti]==0:
                current_slot_idx_to_fill = sloti
                break   
        slot_to_create_in = current_slot_idx_to_fill
        actual_row_ops_create = 0 if slot_to_create_in < N_grid_size else HORIZONTAL_DIVIDER_ROW_INDEX + 1
        actual_col_ops_create = slot_to_create_in % N_grid_size

        # 廃棄操作が必要かチェック
        discard_ops = remaining_paint_in_slot[slot_to_create_in]
        total_ops_needed = discard_ops + best_m_val + 1
        
        # if total_ops_needed > remaining_ops:
        #     actual_discard = min(remaining_ops, discard_ops)
        #     for _ in range(actual_discard):
        #         print(f"3 {actual_row_ops_create} {actual_col_ops_create}")
        #     remaining_paint_in_slot[slot_to_create_in] -= actual_discard
        #     remaining_ops -= actual_discard
        #     continue

        # 廃棄操作
        if discard_ops > 0:
            for _ in range(discard_ops):
                print(f"3 {actual_row_ops_create} {actual_col_ops_create}")
            remaining_ops -= discard_ops
            remaining_paint_in_slot[slot_to_create_in] = 0
            
        # 絵の具追加操作
        for c_idx in best_m_indices:
            print(f"1 {actual_row_ops_create} {actual_col_ops_create} {c_idx}")
        print(f"2 {actual_row_ops_create} {actual_col_ops_create}")
        
        # 状態更新
        current_colors_in_slot[slot_to_create_in] = best_m_mixed_color
        remaining_paint_in_slot[slot_to_create_in] = best_m_val - 1
        current_slot_idx_to_fill = (current_slot_idx_to_fill + 1) % NUM_TOTAL_SLOTS
        remaining_ops -= (best_m_val + 1)
    
if __name__ == "__main__":
    main()