import sys
import numpy as np
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
    M_val = 3 if T>=6000 else 2
    # M_val=2
    targets = []
    for _ in range(H):
        c = float(data[index])
        m = float(data[index+1])
        y = float(data[index+2])
        targets.append((c, m, y))
        index += 3
        
    
    
    # 初期仕切り出力: 縦仕切りは全て1、横仕切りは全て0
    for _ in range(N):
        print(" ".join(["1"] * (N-1)))
    for _ in range(N-1):
        print(" ".join(["0"] * N))
    
    remaining = [0] * N
    err2 = 0
    err3 = 0
    for i in range(H):
        colors, error = select_colors(own_colors, targets[i], 2)
        err2 += error
        colors, error = select_colors(own_colors, targets[i], 3)
        err3 += error
    estimated_score2= np.round(err2*10000)+ D*2*H
    estimated_score3 = np.round(err3*10000)+D*3*H
    if estimated_score2 > estimated_score3 and T >=6000:
        M_val = 3
    else:
        M_val = 2
    current_col = 0
    
    for i in range(H):
        col = current_col
        while remaining[col] > 0:
            print(f"3 0 {col}")
            remaining[col] -= 1
            
        target = targets[i]
        colors, r1 = select_colors(own_colors, target, M_val)
        for c in colors:
            print(f"1 0 {col} {c}")
        remaining[col] = M_val
        print(f"2 0 {col}")
        remaining[col] -= 1
        
        current_col = (current_col + 1) % N

if __name__ == "__main__":
    main()