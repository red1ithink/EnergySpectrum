import Function as F
import DefineFiles as D
import importlib
importlib.reload(F)
importlib.reload(D)
from Function import *
from DefineFiles import *
from scipy.signal import savgol_filter
from scipy import stats
import ruptures as rpt
from scipy.interpolate import interp1d

N=1023

def tracking22_data_merged(k, ek, pen, name,
                          slope_threshold=0.2,  # -5/3 혹은 -4에서 얼마나 가까운지 판정할 임계값
                          merge_plot=True       # 병합 전 세그먼트도 보고 싶다면 True
                         ):
    """
    2차원 난류 스펙트럼에서, PELT 알고리즘으로 먼저 세그먼트를 분할한 뒤
    기울기가 -5/3 or -4 근처인 세그먼트가 연속해서 나타나면 병합하여
    최종적으로 의미있는 구간(energy range, enstrophy range)를 찾는 함수.

    Parameters
    ----------
    k : 1D array
        파수 벡터
    ek : 1D array
        스펙트럼 값
    pen : float
        PELT 알고리즘의 Penalty 파라미터
    name : str
        출력 그래프 제목 등에 표시할 이름
    slope_threshold : float
        -5/3, -4 근처라고 판단할 오차 범위
        (절댓값 |m - (-5/3)| 또는 |m - (-4)| 가 이 값보다 작으면 "근처"로 봄)
    merge_plot : bool
        True 면, 병합 전 세그먼트도 표시
    """
    log_k = np.log(k)
    log_E = np.log(ek)

    # Savitzky-Golay filtering
    window_length = 15
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    # interpolation
    new_log_k = np.linspace(log_k.min(), log_k.max(), N)
    interp_func = interp1d(log_k, smoothed_log_E, kind='linear', fill_value='extrapolate')
    new_log_E = interp_func(new_log_k)
    new_k = np.exp(new_log_k)
    new_E = np.exp(new_log_E)

    signal = np.column_stack((new_log_k, new_log_E))

    class LogLogCost(rpt.base.BaseCost):
        model = "loglog"
        min_size = 2
        
        def fit(self, signal):
            self.signal = signal
            self.X = signal[:, 0]
            self.Y = signal[:, 1]
            return self
        
        def error(self, start, end):
            x_seg = self.X[start:end]
            y_seg = self.Y[start:end]
            if len(x_seg) < 2:
                return np.inf
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            residuals = y_seg - (m * x_seg + b)
            sse = np.sum(residuals**2)
            return sse

    cost = LogLogCost().fit(signal)
    algo = rpt.Pelt(custom_cost=cost).fit(signal)
    bkps = algo.predict(pen=pen) 

    segments_init = []
    start_ = 0
    for bp in bkps:
        segments_init.append((start_, bp))
        start_ = bp

    def get_segment_slope(start, end):
        """
        start~end 구간에서 (log_k, log_E) 데이터를 가지고
        최소제곱법으로 기울기(m)와 b를 구한 뒤
        표준오차(se_m)도 반환
        """
        x_seg = new_log_k[start:end]
        y_seg = new_log_E[start:end]
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - (m * x_seg + b)
        # 표준오차 계산
        mse = np.sum(residuals**2) / (len(x_seg) - 2)
        se_m = np.sqrt(mse / (len(x_seg) * np.var(x_seg)))
        return m, se_m, b

    def classify_slope_simple(m, threshold=slope_threshold):
        target_energy = -5/3 
        target_enst   = -4

        dist_energy = abs(m - target_energy)
        dist_enst   = abs(m - target_enst)

        if dist_energy < threshold and dist_energy < dist_enst:
            return "energy"
        elif dist_enst < threshold and dist_enst < dist_energy:
            return "enstrophy"
        else:
            return "none"

    def merge_segments(segments):
        merged = []
        i = 0
        while i < len(segments):
            s, e = segments[i]
            m, se_m, b_ = get_segment_slope(s, e)
            cls = classify_slope_simple(m, slope_threshold)

            if cls in ["energy", "enstrophy"]:
                j = i + 1
                while j < len(segments):
                    s2, e2 = segments[j]
                    m2, se_m2, b2 = get_segment_slope(s, e2)
                    cls2 = classify_slope_simple(m2, slope_threshold)
                    if cls2 == cls:
                        e = e2
                        j += 1
                    else:
                        break
                merged.append((s, e))
                i = j
            else:
                merged.append((s, e))
                i += 1
        return merged

    if merge_plot:
        plt.figure(figsize=(12, 5))
        plt.loglog(new_k, new_E, 'k-', alpha=0.3, label='Smoothed Data')
        colors = plt.cm.tab10(np.linspace(0, 1, len(segments_init)))
        for idx, (s, e) in enumerate(segments_init):
            k_seg = np.exp(new_log_k[s:e])
            E_seg = np.exp(new_log_E[s:e])
            m, se_m, b_ = get_segment_slope(s, e)
            plt.loglog(k_seg, E_seg, color=colors[idx], lw=2,
                       label=f"Seg{idx+1} (m={m:.2f})")
        plt.title(f"[Before Merging] {name}")
        plt.xlabel("k")
        plt.ylabel("E(k)")
        plt.legend()
        plt.grid(True, which='both', alpha=0.4)
        plt.tight_layout()
        plt.show()

    # 세그먼트 병합
    merged_segments = merge_segments(segments_init)

    plt.figure(figsize=(12, 6))
    plt.loglog(new_k, new_E, 'k-', alpha=0.3, label='Smoothed Data')

    colors = plt.cm.tab10(np.linspace(0, 1, len(merged_segments)))
    for idx, (s, e) in enumerate(merged_segments):
        k_seg = np.exp(new_log_k[s:e])
        E_seg = np.exp(new_log_E[s:e])
        m, se_m, b_ = get_segment_slope(s, e)
        plt.loglog(k_seg, E_seg, color=colors[idx], lw=2.5,
                   label=f"MergedSeg {idx+1}: slope={m:.3f}")

    print("[ Final Merged Segments ]")
    for idx, (s, e) in enumerate(merged_segments):
        m, se_m, b_ = get_segment_slope(s, e)
        k_start = new_k[s]
        k_end   = new_k[e-1] if e-1 < len(new_k) else new_k[-1]
        print(f"  MergedSeg {idx+1}: indices [{s}, {e}), "
              f"k from {k_start:.3e} to {k_end:.3e}, slope={m:.3f}")

    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title(f'Turbulence Spectrum Analysis (Merged) [{name}]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
    plt.show()

    return merged_segments



##########



## interpolation
def tracking2_data(k, ek, pen, name):

    log_k = np.log(k)
    log_E = np.log(ek)
    # Savitzky-Golay filter
    window_length = 15
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    # interpolation
    # E(k) ~ k^n => log(E) ~ n*log(k) + C
    new_log_k = np.linspace(log_k.min(), log_k.max(), N)
    interp_func = interp1d(log_k, smoothed_log_E, kind='linear', fill_value='extrapolate')
    new_log_E = interp_func(new_log_k)
    new_k = np.exp(new_log_k)
    new_E = np.exp(new_log_E)

    # making singal for ruptures(segmentation)
    signal = np.column_stack((new_log_k, new_log_E))

    # Custom Cost Class
    class LogLogCost(rpt.base.BaseCost):
        model = "loglog"
        min_size = 2          # Minimum size of segment
        
        def fit(self, signal):
            self.signal = signal
            self.X = signal[:, 0]
            self.Y = signal[:, 1]
            return self
        

        def error(self, start, end):
            x_seg = self.X[start:end]
            y_seg = self.Y[start:end]
            if len(x_seg) < 2:
                return np.inf
            # y = m*x + b
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            residuals = y_seg - (m * x_seg + b)
            sse = np.sum(residuals**2)
            return sse

    # cost
    cost = LogLogCost().fit(signal)

    # PELT Algorithm
    algo = rpt.Pelt(custom_cost=cost).fit(signal)
    pen = pen #Penalty
    bkps = algo.predict(pen=pen)
    rpt.display(signal, bkps, figsize=(10, 6))
    plt.show()

    segments = []
    start = 0
    for bp in bkps:
        segments.append((start, bp))
        start = bp

    # SSE
    def get_segment_slope(start, end):
        x_seg = new_log_k[start:end]
        y_seg = new_log_E[start:end]
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - (m * x_seg + b)
        mse = np.sum(residuals**2) / (len(x_seg) - 2)
        se_m = np.sqrt(mse / (len(x_seg) * np.var(x_seg)))
        return m, se_m, b

    # t-test
    def classify_slope(m, se_m, n):
        results = []
        for target in [-5/3, -4]:
            t_stat = (m - target) / se_m
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
            results.append((target, p_val))
        best_target, best_p = max(results, key=lambda x: x[1])
        if best_p > 0.05:
            return f"Slope {m:.3f}"
        else:
            return f"Slope {m:.3f}"

    # figure
    plt.figure(figsize=(12, 6))
    plt.loglog(new_k, new_E, 'k-', alpha=0.3, label='Original Data')
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for idx, (s, e) in enumerate(segments):
        k_seg = np.exp(new_log_k[s:e])  # Original 'k'
        E_seg = np.exp(new_log_E[s:e])
        m, se_m, b = get_segment_slope(s, e)
        label = classify_slope(m, se_m, e - s)
        plt.loglog(k_seg, E_seg, color=colors[idx], lw=2.5, 
                label=f"Segment {idx+1}: {label}")
    print("Found segments:")
    for idx, (s, e) in enumerate(segments):
        k_start = new_k[s]
        k_end = new_k[e-1] if e-1 < len(new_k) else new_k[-1]
        print(f"Segment {idx+1}: indices {s} to {e}, k from {k_start:.2e} to {k_end:.2e}")

    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title(f'Turbulence Spectrum Analysis [{name}]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()

def tracking2(file, pen, name):
    k, ek = get_ek(file) 
    k = k[1:]
    ek = ek[1:]

    log_k = np.log(k)
    log_E = np.log(ek)

    window_length = 15 
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    new_log_k = np.linspace(log_k.min(), log_k.max(), N)
    interp_func = interp1d(log_k, smoothed_log_E, kind='linear', fill_value='extrapolate')
    new_log_E = interp_func(new_log_k)
    new_k = np.exp(new_log_k)
    new_E = np.exp(new_log_E)

    signal = np.column_stack((new_log_k, new_log_E))

    class LogLogCost(rpt.base.BaseCost):
        model = "loglog"
        min_size = 2    
        
        def fit(self, signal):
            self.signal = signal
            self.X = signal[:, 0]
            self.Y = signal[:, 1]
            return self
            
        def error(self, start, end):
            x_seg = self.X[start:end]
            y_seg = self.Y[start:end]
            if len(x_seg) < 2:
                return np.inf
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            residuals = y_seg - (m * x_seg + b)
            sse = np.sum(residuals**2)
            return sse

    cost = LogLogCost().fit(signal)

    algo = rpt.Pelt(custom_cost=cost).fit(signal)
    pen = pen 
    bkps = algo.predict(pen=pen)

    segments = []
    start = 0
    for bp in bkps:
        segments.append((start, bp))
        start = bp

    def get_segment_slope(start, end):
        x_seg = new_log_k[start:end]
        y_seg = new_log_E[start:end]
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - (m * x_seg + b)
        mse = np.sum(residuals**2) / (len(x_seg) - 2)
        se_m = np.sqrt(mse / (len(x_seg) * np.var(x_seg)))
        return m, se_m, b

    def classify_slope(m, se_m, n):
        results = []
        for target in [-5/3, -4]:
            t_stat = (m - target) / se_m
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
            results.append((target, p_val))
        best_target, best_p = max(results, key=lambda x: x[1])
        if best_p > 0.05:
            return f"Slope {m:.3f}"
        else:
            return f"Slope {m:.3f}"

    plt.figure(figsize=(12, 6))
    plt.loglog(new_k, new_E, 'k-', alpha=0.3, label='Original Data')
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for idx, (s, e) in enumerate(segments):
        k_seg = np.exp(new_log_k[s:e])
        E_seg = np.exp(new_log_E[s:e])
        m, se_m, b = get_segment_slope(s, e)
        label = classify_slope(m, se_m, e - s)
        plt.loglog(k_seg, E_seg, color=colors[idx], lw=2.5, 
                label=f"Segment {idx+1}: {label}")
    print("Found segments:")
    for idx, (s, e) in enumerate(segments):
        k_start = new_k[s]
        k_end = new_k[e-1] if e-1 < len(new_k) else new_k[-1]
        print(f"Segment {idx+1}: indices {s} to {e}, k from {k_start:.2e} to {k_end:.2e}")

    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title(f'Turbulence Spectrum Analysis [{name}]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()

## Non interpolation
def tracking(file, pen, name):
    k, ek = get_ek(file)
    k = k[1:]
    ek = ek[1:]

    log_k = np.log(k)
    log_E = np.log(ek)
    window_length = 15
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    signal = np.column_stack((log_k, smoothed_log_E))

    class LogLogCost(rpt.base.BaseCost):
        model = "loglog"      
        min_size = 2          
        
        def fit(self, signal):
            self.signal = signal
            self.X = signal[:, 0]
            self.Y = signal[:, 1]
            return self
            
        def error(self, start, end):
            x_seg = self.X[start:end]
            y_seg = self.Y[start:end]
            if len(x_seg) < 2:
                return np.inf
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            residuals = y_seg - (m * x_seg + b)
            sse = np.sum(residuals**2)
            return sse

    cost = LogLogCost().fit(signal)

    algo = rpt.Pelt(custom_cost=cost).fit(signal)
    pen = pen
    bkps = algo.predict(pen=pen)

    segments = []
    start = 0
    for bp in bkps:
        segments.append((start, bp))
        start = bp

    def get_segment_slope(start, end):
        x_seg = log_k[start:end]
        y_seg = smoothed_log_E[start:end]
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - (m * x_seg + b)
        mse = np.sum(residuals**2) / (len(x_seg) - 2)
        se_m = np.sqrt(mse / (len(x_seg) * np.var(x_seg)))
        return m, se_m, b

    def classify_slope(m, se_m, n):
        results = []
        for target in [-5/3, -4]:
            t_stat = (m - target) / se_m
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
            results.append((target, p_val))
        best_target, best_p = max(results, key=lambda x: x[1])
        if best_p > 0.05:
            return f"Slope {m:.3f}"
        else:
            return f"Slope {m:.3f}"

    plt.figure(figsize=(12, 6))
    plt.loglog(k, ek, 'k-', alpha=0.3, label='Original Data')
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for idx, (s, e) in enumerate(segments):
        k_seg = np.exp(log_k[s:e])
        E_seg = np.exp(smoothed_log_E[s:e])
        m, se_m, b = get_segment_slope(s, e)
        label = classify_slope(m, se_m, e - s)
        plt.loglog(k_seg, E_seg, color=colors[idx], lw=2.5, 
                label=f"Segment {idx+1}: {label}")
    print("Found segments:")
    for idx, (s, e) in enumerate(segments):
        k_start = k[s]
        k_end = k[e-1] if e-1 < len(k) else k[-1]
        print(f"Segment {idx+1}: indices {s} to {e}, k from {k_start:.2e} to {k_end:.2e}")

    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title(f'Turbulence Spectrum Analysis [{name}]')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()

def tracking_data(k, ek, pen, name):

    log_k = np.log(k)
    log_E = np.log(ek)
    window_length = 15
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    signal = np.column_stack((log_k, smoothed_log_E))

    class LogLogCost(rpt.base.BaseCost):
        model = "loglog"
        min_size = 2
        
        def fit(self, signal):
            self.signal = signal
            self.X = signal[:, 0]
            self.Y = signal[:, 1]
            return self
            
        def error(self, start, end):
            x_seg = self.X[start:end]
            y_seg = self.Y[start:end]
            if len(x_seg) < 2:
                return np.inf
            A = np.vstack([x_seg, np.ones(len(x_seg))]).T
            m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
            residuals = y_seg - (m * x_seg + b)
            sse = np.sum(residuals**2)
            return sse

    cost = LogLogCost().fit(signal)

    algo = rpt.Pelt(custom_cost=cost).fit(signal)
    pen = pen 
    bkps = algo.predict(pen=pen)
    rpt.display(signal, bkps, figsize=(10, 6))
    plt.show()


    segments = []
    start = 0
    for bp in bkps:
        segments.append((start, bp))
        start = bp

    def get_segment_slope(start, end):
        x_seg = log_k[start:end]
        y_seg = smoothed_log_E[start:end]
        A = np.vstack([x_seg, np.ones(len(x_seg))]).T
        m, b = np.linalg.lstsq(A, y_seg, rcond=None)[0]
        residuals = y_seg - (m * x_seg + b)
        mse = np.sum(residuals**2) / (len(x_seg) - 2)
        se_m = np.sqrt(mse / (len(x_seg) * np.var(x_seg)))
        return m, se_m, b

    def classify_slope(m, se_m, n):
        results = []
        for target in [-5/3, -4]:
            t_stat = (m - target) / se_m
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
            results.append((target, p_val))
        best_target, best_p = max(results, key=lambda x: x[1])
        if best_p > 0.05:
            return f"Slope {m:.3f}"
        else:
            return f"Slope {m:.3f}"

    plt.figure(figsize=(12, 6))
    plt.loglog(k, ek, 'k-', alpha=0.3, label='Original Data')
    colors = plt.cm.tab10(np.linspace(0, 1, len(segments)))
    for idx, (s, e) in enumerate(segments):
        k_seg = np.exp(log_k[s:e])
        E_seg = np.exp(smoothed_log_E[s:e])
        m, se_m, b = get_segment_slope(s, e)
        label = classify_slope(m, se_m, e - s)
        plt.loglog(k_seg, E_seg, color=colors[idx], lw=2.5, 
                label=f"Segment {idx+1}: {label}")
    print("Found segments:")
    for idx, (s, e) in enumerate(segments):
        k_start = k[s]
        k_end = k[e-1] if e-1 < len(k) else k[-1]
        print(f"Segment {idx+1}: indices {s} to {e}, k from {k_start:.2e} to {k_end:.2e}")

    plt.xlabel('k')
    plt.ylabel('E(k)')
    plt.title(f'Turbulence Spectrum Analysis [{name}]')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', alpha=0.4)
    plt.tight_layout()
