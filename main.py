import sys
import numpy as np
import math
import random
import sys

def select_colors(own_colors, target, M_val):
    best_error = float('inf')
    best_colors = []
    best_r1 = 0.5
    K = len(own_colors)
    if M_val == 2:
        for c1 in range(K):
            for c2 in range(K):
                vec1 = own_colors[c1]
                vec2 = own_colors[c2]
                colors=[c1, c2]
                r1 = .5
                mixed = [
                    r1 * vec1[0] + (1 - r1) * vec2[0],
                    r1 * vec1[1] + (1 - r1) * vec2[1],
                    r1 * vec1[2] + (1 - r1) * vec2[2]
                ]
                error = 0
                for d in range(3):
                    error += (mixed[d] - target[d])**2
                error = error ** 0.5
                if error < best_error:
                    best_error = error
                    best_colors = colors
    elif M_val == 3:
        for c1 in range(K):
            for c2 in range(K):
                for c3 in range(K):
                    vec1 = own_colors[c1]
                    vec2 = own_colors[c2]
                    vec3 = own_colors[c3]
                    colors = [c1, c2, c3]
                    r1 = .33
                    mixed = [
                        r1*vec1[0]+r1*vec2[0]+(1-2*r1)*vec3[0],
                        r1*vec1[1]+r1*vec2[1]+(1-2*r1)*vec3[1],
                        r1*vec1[2]+r1*vec2[2]+(1-2*r1)*vec3[2]
                    ]
                    error = 0
                    for d in range(3):
                        error += (mixed[d] - target[d])**2
                    error = error ** 0.5
                    if error < best_error:
                        best_error = error
                        best_colors = colors                    
    return (best_colors, best_error)
import copy
# (get_mixed_color と calculate_error_sq は前の回答から流用)
def get_mixed_color(colors_to_mix, own_colors_list_for_hc, m_val_for_hc):
    """色のインデックスリストから平均混合色を計算"""
    if not colors_to_mix or m_val_for_hc == 0:
        return (0.0, 0.0, 0.0)
    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    for idx in colors_to_mix:
        color = own_colors_list_for_hc[idx]
        sum_r += color[0]
        sum_g += color[1]
        sum_b += color[2]
    return (sum_r / m_val_for_hc, sum_g / m_val_for_hc, sum_b / m_val_for_hc)

def calculate_error(mixed_color_tuple, target_color_tuple):
    """混合色とターゲット色のユークリッド距離を計算"""
    err_sq = 0.0
    for i in range(3):
        err_sq += (mixed_color_tuple[i] - target_color_tuple[i])**2
    return math.sqrt(err_sq)
# (get_mixed_color と calculate_error は上記から流用)

def select_colors_sa(own_colors_list, target_color_vec, m_val,
                                     initial_num_samples=10, # 初期解生成用
                                     sa_initial_temp=1.0,   # 初期温度
                                     sa_final_temp=0.003,  # 最終温度
                                     sa_cooling_rate=0.95, # 冷却率
                                     sa_iterations_per_temp=10): # 各温度での試行回数
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0 or m_val == 0:
        mixed_c = (0,0,0) if m_val == 0 else own_colors_list[0] if num_own_colors > 0 else (0,0,0)
        return [0] * m_val if num_own_colors > 0 else [], calculate_error(mixed_c, target_color_vec)

    # 1. 初期解をランダムサンプリングで生成
    current_indices = [random.randint(0, num_own_colors - 1) for _ in range(m_val)]
    mixed_color = get_mixed_color(current_indices, own_colors_list, m_val)
    current_error = calculate_error(mixed_color, target_color_vec)

    # (オプション：初期解をもう少し良くする)
    for _ in range(initial_num_samples -1):
        temp_indices = [random.randint(0, num_own_colors - 1) for _ in range(m_val)]
        temp_mixed_color = get_mixed_color(temp_indices, own_colors_list, m_val)
        temp_error = calculate_error(temp_mixed_color, target_color_vec)
        if temp_error < current_error:
            current_error = temp_error
            current_indices = temp_indices
            
    best_indices = copy.deepcopy(current_indices)
    best_error = current_error
    
    # 2. 焼きなまし法
    current_temp = sa_initial_temp
    while current_temp > sa_final_temp:
        for _ in range(sa_iterations_per_temp): # 各温度で複数回近傍探索
            # 近傍を生成: 1要素変更
            neighbor_indices = copy.deepcopy(current_indices)
            if not neighbor_indices: continue

            idx_to_change = random.randint(0, m_val - 1)
            new_color_idx = random.randint(0, num_own_colors - 1)
            neighbor_indices[idx_to_change] = new_color_idx

            neighbor_mixed_color = get_mixed_color(neighbor_indices, own_colors_list, m_val)
            neighbor_error = calculate_error(neighbor_mixed_color, target_color_vec)

            delta_error = neighbor_error - current_error

            if delta_error < 0: # 改善した場合は常に移動
                current_indices = neighbor_indices
                current_error = neighbor_error
                if current_error < best_error: # 全体での最良解を更新
                    best_error = current_error
                    best_indices = copy.deepcopy(current_indices)
            else: # 悪化した場合でも確率で移動
                if current_temp > 1e-9: # 温度が0に近い場合はほぼ移動しない
                    acceptance_probability = math.exp(-delta_error / current_temp)
                    if random.random() < acceptance_probability:
                        current_indices = neighbor_indices
                        current_error = neighbor_error
        
        current_temp *= sa_cooling_rate # 温度を冷却
            
    return best_indices, best_error
def select_colors_rs(own_colors_list, target_color_vec, m_val, num_samples=500):
    # random samplingで色を選ぶ関数 (Pythonリストベース)

        
    best_err_val = float('inf')
    best_indices_to_drop = []
    num_own_colors = len(own_colors_list)

    if num_own_colors == 0:
        # ターゲットが(0,0,0)でない場合の誤差
        error_sum_sq = 0.0
        for tc_val in target_color_vec:
            error_sum_sq += tc_val**2
        return [], math.sqrt(error_sum_sq)
        
    if m_val == 0:
        # ターゲットが(0,0,0)でない場合の誤差
        error_sum_sq = 0.0
        for tc_val in target_color_vec:
            error_sum_sq += tc_val**2
        return [], math.sqrt(error_sum_sq)

    for _ in range(num_samples):
        # m_val_drops 個のランダムなインデックスを生成 (重複あり)
        current_indices = [random.randint(0, num_own_colors - 1) for _ in range(m_val)]
        
        # 選ばれた色を取得し混合 (Pythonリストで処理)
        sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
        for idx in current_indices:
            # own_colors_list の各要素は (r, g, b) のタプルまたはリストと仮定
            color = own_colors_list[idx] 
            sum_r += color[0]
            sum_g += color[1]
            sum_b += color[2]
        
        mixed_r = sum_r / m_val
        mixed_g = sum_g / m_val
        mixed_b = sum_b / m_val

        # 誤差計算 (Pythonリストとmathで処理)
        # target_color_vec も (r, g, b) のタプルまたはリストと仮定
        err_sq = (mixed_r - target_color_vec[0])**2 + \
                 (mixed_g - target_color_vec[1])**2 + \
                 (mixed_b - target_color_vec[2])**2
        current_err = math.sqrt(err_sq)

        if current_err < best_err_val:
            best_err_val = current_err
            best_indices_to_drop = current_indices
    
    if not best_indices_to_drop: # 万が一、一度も更新されなかった場合 (num_samples=0など)
        # デフォルトとして最初の色をM_val個使う
        best_indices_to_drop = [0] * m_val
        # この場合の誤差も計算しておく
        # (最初の色をm_val_drops滴使ったことになるので、混合結果は最初の色そのもの)
        default_color = own_colors_list[0]
        err_sq = (default_color[0] - target_color_vec[0])**2 + \
                 (default_color[1] - target_color_vec[1])**2 + \
                 (default_color[2] - target_color_vec[2])**2
        best_err_val = math.sqrt(err_sq)

    return (best_indices_to_drop, best_err_val)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    N = int(data[0])
    K = int(data[1])
    H = int(data[2])
    T = int(data[3])
    D = float(data[4])
    
    own_colors = []
    index = 5
    for _ in range(K):
        c = float(data[index])
        m = float(data[index+1])
        y = float(data[index+2])
        own_colors.append((c, m, y))
        index += 3

    # M_val=2
    targets = []
    for _ in range(H):
        c = float(data[index])
        m = float(data[index+1])
        y = float(data[index+2])
        targets.append((c, m, y))
        index += 3
        
    max_M=0
    for Mi in range(1,21):
        if T >= 2*Mi*1000:
            max_M = Mi
    
    # 初期仕切り出力: 縦仕切りは全て1、横仕切りは全て0
    for _ in range(N):
        print(" ".join(["1"] * (N-1)))
    for _ in range(N-1):
        print(" ".join(["0"] * N))
    
    remaining = [0] * N
    max_M = min(max_M, 8)  # M_valは最低でも3にする
    Ms=[2, max(2,max_M//2),max_M]
    # if max_M-3>3:
    #     Ms.append(max_M-3)
    if T>=6000 and Ms[1]==2:
        Ms[1]=3
    errs=[0]*len(Ms)

    N_trials =30
    for i in range(N_trials):
        for j in range(len(Ms)):
            if Ms[j] <= 3:
                colors, err = select_colors(own_colors, targets[i], Ms[j])
            else:
                colors, err = select_colors_sa(own_colors, targets[i], Ms[j])
            errs[j] += err
    estimated_scores = [0] * len(Ms)
    for i in range(len(Ms)):
        estimated_scores[i] =  np.round(errs[i]*1000/N_trials*1e4)+ D*Ms[i]*1000
    # estimated_scoreが一番小さいMsを選ぶ
    M_val = Ms[np.argmin(estimated_scores)]
    print(f"Estimated M value: {M_val}",file=sys.stderr)
    current_col = 0
    
    for i in range(H):
        col = current_col
        while remaining[col] > 0:
            print(f"3 0 {col}")
            remaining[col] -= 1
            
        target = targets[i]
        if M_val <=3:
            colors, r1 = select_colors(own_colors, target, M_val)
        else:
            colors, r1 = select_colors_sa(own_colors, target, M_val)
        for c in colors:
            print(f"1 0 {col} {c}")
        remaining[col] = M_val
        print(f"2 0 {col}")
        remaining[col] -= 1
        
        current_col = (current_col + 1) % N

if __name__ == "__main__":
    main()