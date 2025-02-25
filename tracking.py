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

def tracking2_data(k, ek, pen, name):

    log_k = np.log(k)
    log_E = np.log(ek)
    # Savitzky-Golay filter
    window_length = 15
    polyorder = 3
    smoothed_log_E = savgol_filter(log_E, window_length, polyorder)

    # interpolation
    # E(k) ~ k^n => log(E) ~ n*log(k) + C
    new_log_k = np.linspace(log_k.min(), log_k.max(), 1023)
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

    new_log_k = np.linspace(log_k.min(), log_k.max(), 1023)
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
