import sys
import math
import random
import copy # deepcopyは重いので注意
import itertools
import time
import numpy as np # スコア計算やargminのために残す

# --- Helper Functions for Color Mixing and Error ---
def get_mixed_color(indices_to_mix, own_colors_list_ref, m_val_eff):
    """色のインデックスリストから平均混合色を計算"""
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
    """混合色とターゲット色のユークリッド距離を計算"""
    err_sq = 0.0
    for i in range(3):
        err_sq += (mixed_color_tuple[i] - target_color_tuple[i])**2
    return math.sqrt(err_sq)

# --- Color Selection Strategies ---

def select_colors_exhaustive(own_colors_list, target_color_vec, m_val):
    """M_valが小さい場合に全探索で色を選ぶ (itertools版)"""
    best_err_val = float('inf')
    best_indices_to_drop = []
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0 or m_val == 0:
        err = calculate_error((0,0,0) if m_val == 0 else (own_colors_list[0] if num_own_colors > 0 else (0,0,0)), target_color_vec)
        return ([0] * m_val if num_own_colors > 0 else []), err

    for current_indices_tuple in itertools.combinations_with_replacement(range(num_own_colors), m_val):
        mixed_color = get_mixed_color(current_indices_tuple, own_colors_list, m_val)
        current_err = calculate_error(mixed_color, target_color_vec)

        if current_err < best_err_val:
            best_err_val = current_err
            best_indices_to_drop = list(current_indices_tuple)
            
    if not best_indices_to_drop and num_own_colors > 0 : # Should be populated if num_own_colors > 0
        best_indices_to_drop = [0] * m_val
        mixed_color = get_mixed_color(best_indices_to_drop, own_colors_list, m_val)
        best_err_val = calculate_error(mixed_color, target_color_vec)
    elif not best_indices_to_drop and num_own_colors == 0: # Failsafe
        best_indices_to_drop = []
        best_err_val = calculate_error((0,0,0), target_color_vec)


    return best_indices_to_drop, best_err_val

def select_colors_rs(own_colors_list, target_color_vec, m_val, num_samples=100):
    """ランダムサンプリングで色を選ぶ (M_val決定時やSAの初期解候補に)"""
    best_err_val = float('inf')
    best_indices_to_drop = []
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0 or m_val == 0:
        err = calculate_error((0,0,0) if m_val == 0 else (own_colors_list[0] if num_own_colors > 0 else (0,0,0)), target_color_vec)
        return ([0] * m_val if num_own_colors > 0 else []), err

    for _ in range(num_samples):
        current_indices = [random.randint(0, num_own_colors - 1) for _ in range(m_val)]
        mixed_color = get_mixed_color(current_indices, own_colors_list, m_val)
        current_err = calculate_error(mixed_color, target_color_vec)

        if current_err < best_err_val:
            best_err_val = current_err
            best_indices_to_drop = current_indices
    
    if not best_indices_to_drop and num_own_colors > 0:
        best_indices_to_drop = [0] * m_val
        mixed_color = get_mixed_color(best_indices_to_drop, own_colors_list, m_val)
        best_err_val = calculate_error(mixed_color, target_color_vec)
    elif not best_indices_to_drop and num_own_colors == 0:
        best_indices_to_drop = []
        best_err_val = calculate_error((0,0,0), target_color_vec)

    return best_indices_to_drop, best_err_val

def select_colors_sa_timed(own_colors_list, target_color_vec, m_val,
                           sa_params, # 辞書で渡す: {'initial_temp', 'final_temp', 'cooling_rate', 'iterations_per_temp'}
                           time_limit_ms):
    num_own_colors = len(own_colors_list)
    start_time_sa = time.perf_counter()

    if num_own_colors == 0 or m_val == 0:
        err = calculate_error((0,0,0) if m_val == 0 else (own_colors_list[0] if num_own_colors > 0 else (0,0,0)), target_color_vec)
        return ([0] * m_val if num_own_colors > 0 else []), err

    # 1. 初期解 (簡易RSで)
    current_indices, current_error = select_colors_rs(own_colors_list, target_color_vec, m_val, num_samples=10) # 軽量RS
            
    best_indices = current_indices[:] # 浅いコピーでOK
    best_error = current_error
    
    # 2. 焼きなまし法
    current_temp = sa_params['initial_temp']
    while current_temp > sa_params['final_temp']:
        if (time.perf_counter() - start_time_sa) * 1000 > time_limit_ms: break

        for _ in range(sa_params['iterations_per_temp']):
            if (time.perf_counter() - start_time_sa) * 1000 > time_limit_ms: break # 内側でもチェック

            neighbor_indices = current_indices[:] # 浅いコピー
            if not neighbor_indices: continue

            idx_to_change = random.randint(0, m_val - 1)
            new_color_idx = random.randint(0, num_own_colors - 1)
            neighbor_indices[idx_to_change] = new_color_idx

            neighbor_mixed_color = get_mixed_color(neighbor_indices, own_colors_list, m_val)
            neighbor_error = calculate_error(neighbor_mixed_color, target_color_vec)
            delta_error = neighbor_error - current_error

            if delta_error < 0:
                current_indices = neighbor_indices
                current_error = neighbor_error
                if current_error < best_error:
                    best_error = current_error
                    best_indices = current_indices[:]
            else:
                if current_temp > 1e-9:
                    acceptance_probability = math.exp(-delta_error / current_temp)
                    if random.random() < acceptance_probability:
                        current_indices = neighbor_indices
                        current_error = neighbor_error
        current_temp *= sa_params['cooling_rate']
            
    return best_indices, best_error

# --- Main Logic ---
def main():
    overall_start_time = time.perf_counter()
    
    data = sys.stdin.read().split()
    if not data: return
    
    ptr = 0
    N = int(data[ptr]); ptr+=1
    K = int(data[ptr]); ptr+=1
    H = int(data[ptr]); ptr+=1
    T_total_ops_limit = int(data[ptr]); ptr+=1 # T from problem
    D_cost_factor = float(data[ptr]); ptr+=1
    
    own_colors = []
    for _ in range(K):
        own_colors.append((float(data[ptr]), float(data[ptr+1]), float(data[ptr+2])))
        ptr += 3
        
    targets = []
    for _ in range(H):
        targets.append((float(data[ptr]), float(data[ptr+1]), float(data[ptr+2])))
        ptr += 3
        
    # --- M_val 決定ロジック ---
    max_possible_M_val_by_ops = 0
    m_val_candidates = []
    for m_cand in range(1, N+1): # M_valは最大でも盤面のセル数Nまで (1列使う場合)
                                 # ただし、実質的には2*m*H <= T が制約
        if 2 * m_cand * H <= T_total_ops_limit:
            m_val_candidates.append(m_cand)
            max_possible_M_val_by_ops = m_cand
        else:
            break
    
    # M_val候補 (実験的に調整)
    # あまり多くの候補を試すとM_val決定に時間がかかる
    
    # if max_possible_M_val_by_ops >= 3: m_val_candidates.append(3)
    # if max_possible_M_val_by_ops >= 5: m_val_candidates.append(5)
    # if max_possible_M_val_by_ops >= 8 and 8 not in m_val_candidates: m_val_candidates.append(8)
    # if max_possible_M_val_by_ops >= 12 and 12 not in m_val_candidates: m_val_candidates.append(12)
    # if max_possible_M_val_by_ops >= max_possible_M_val_by_ops and max_possible_M_val_by_ops not in m_val_candidates and max_possible_M_val_by_ops > 0:
    #      if max_possible_M_val_by_ops not in m_val_candidates: m_val_candidates.append(max_possible_M_val_by_ops)
    
    m_val_candidates = sorted(list(set(m_val_candidates))) # 重複削除とソート
    if not m_val_candidates: m_val_candidates = [2] # 最低でも2

    # print(f"Possible M_val candidates by ops: {m_val_candidates}", file=sys.stderr)

    best_estimated_score = float('inf')
    chosen_M_val = m_val_candidates[0]
    
    N_TRIALS_FOR_M_VAL_EST = 50 # ターゲット数に応じて調整
    RS_SAMPLES_FOR_M_VAL_EST = 30 # M_val評価時のRS試行回数 (SAより高速なRSで評価)

    if len(m_val_candidates) > 1: # 候補が複数ある場合のみ評価
        for m_cand_idx, m_cand_val in enumerate(m_val_candidates):
            current_sum_err = 0.0
            for i in range(N_TRIALS_FOR_M_VAL_EST):
                target_idx_for_eval = (i * (H // N_TRIALS_FOR_M_VAL_EST if N_TRIALS_FOR_M_VAL_EST >0 else 1 )) % H # 飛び飛びのターゲットで評価
                
                if m_cand_val <= 3: # 全探索可能な範囲
                    _, err = select_colors_exhaustive(own_colors, targets[target_idx_for_eval], m_cand_val)
                else: # それ以外は軽量RSで評価
                    _, err = select_colors_rs(own_colors, targets[target_idx_for_eval], m_cand_val, num_samples=RS_SAMPLES_FOR_M_VAL_EST)
                current_sum_err += err
            
            avg_err = current_sum_err / N_TRIALS_FOR_M_VAL_EST if N_TRIALS_FOR_M_VAL_EST > 0 else float('inf')
            # スコア計算 (D_cost_factor * (Type1_ops - H) + error_term)
            # Type1_ops = m_cand_val * H
            # (V-H) = (m_cand_val * H - H) = (m_cand_val - 1) * H
            cost_V = D_cost_factor * (m_cand_val -1) * H # 正しいコスト計算
            cost_E = round(10000 * avg_err * H) # 総誤差にスケールアップ
            current_estimated_score = cost_V + cost_E
            # print(f"Eval M_val={m_cand_val}: AvgErr={avg_err:.4f}, EstScore={current_estimated_score}", file=sys.stderr)

            if current_estimated_score < best_estimated_score:
                best_estimated_score = current_estimated_score
                chosen_M_val = m_cand_val
    else:
        chosen_M_val = m_val_candidates[0]

    print(f"Selected M_val: {chosen_M_val}", file=sys.stderr)

    # --- SA パラメータ (本番用) ---
    # これらの値は実験で調整
    SA_PARAMS_MAIN = {
        'initial_temp': 1,   # 誤差のスケールに依存。小さめから試す
        'final_temp': 1e-4,
        'cooling_rate': 0.98, # 0.95-0.995
        'iterations_per_temp': 10
    }
    EXHAUSTIVE_THRESHOLD_M_VAL = 3 # このM_val以下は全探索

    # --- 初期仕切り出力 ---
    for _ in range(N): print(" ".join(["1"] * (N - 1)))
    for _ in range(N - 1): print(" ".join(["0"] * N))
    
    remaining_paint_in_col = [0] * N
    current_col_idx = 0
    
    # --- メインループ ---
    TOTAL_TIME_BUDGET_S = 2.6 # 全体の時間予算 (3秒より少し短く)

    for i in range(H):
        time_spent_s = time.perf_counter() - overall_start_time
        remaining_time_budget_s = TOTAL_TIME_BUDGET_S - time_spent_s
        
        # ターゲットあたりの時間配分 (SA用)
        avg_time_per_remaining_target_ms = 0
        if H - i > 0:
            avg_time_per_remaining_target_ms = (remaining_time_budget_s / (H - i)) * 1000
        
        # SAの実行時間制限 (最低0.5ms, 平均値にキャップ、最大 مثلا 2.8msなど)
        # この値も調整が非常に重要
        sa_time_limit_ms_for_this_target = max(0.2, min(avg_time_per_remaining_target_ms * 0.8, 2.4)) # 少し控えめに

        if remaining_time_budget_s < 0.05 and i < H -1 : # 残り時間が非常に少ない場合はSAを軽量化またはスキップ
             sa_time_limit_ms_for_this_target = 0.2 # 最低限
        
        col_to_use = current_col_idx
        
        # 前の塗料を捨てる
        for _ in range(remaining_paint_in_col[col_to_use]):
            print(f"3 0 {col_to_use}")
        remaining_paint_in_col[col_to_use] = 0
            
        target_color = targets[i]
        
        selected_indices = []
        # error_val = float('inf') # 使わないので不要

        if chosen_M_val <= EXHAUSTIVE_THRESHOLD_M_VAL:
            selected_indices, _ = select_colors_exhaustive(own_colors, target_color, chosen_M_val)
        else:
            selected_indices, _ = select_colors_sa_timed(own_colors, target_color, chosen_M_val,
                                                       SA_PARAMS_MAIN, sa_time_limit_ms_for_this_target)
        
        if not selected_indices and K > 0 : # 万が一空ならデフォルト
            selected_indices = [0] * chosen_M_val
            # print(f"Warning: selected_indices empty for target {i}, M_val {chosen_M_val}", file=sys.stderr)


        for c_idx in selected_indices:
            print(f"1 0 {col_to_use} {c_idx}")
            
        print(f"2 0 {col_to_use}") # 混合して画伯へ
        remaining_paint_in_col[col_to_use] = chosen_M_val - 1 # 1g使ったので残り
        
        current_col_idx = (current_col_idx + 1) % N

    # print(f"Total time: {time.perf_counter() - overall_start_time:.3f}s", file=sys.stderr)

if __name__ == "__main__":
    main()