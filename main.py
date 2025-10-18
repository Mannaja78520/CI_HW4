# pso_mlp_airquality_days_numpy.py
# PSO + MLP (NumPy) for AirQualityUCI.xlsx — forecast in DAYS (5, 10).
# Reads .xlsx via zip+xml (no pandas/openpyxl). 10-fold time-blocked CV. Metric: MAE.
# Inputs: cols 3,6,8,10,11,12,13,14 ; Target: col 5 (Benzene)
# NOTE: No CSV outputs; print days only (no hours).

import os, time, zipfile
import xml.etree.ElementTree as ET
import numpy as np

# ========= CONFIG (แก้ตรงนี้ได้) =========
DATA_PATH   = "AirQualityUCI.xlsx"    # ชื่อไฟล์ .xlsx
DAYS_LIST   = [5, 10]                  # ระยะพยากรณ์เป็น "วัน"
ARCHS       = [[8], [16], [32], [16,8], [32,16]]   # โครงสร้าง hidden layers
PSO_CFG     = {"swarm": 20, "iters": 120, "w": 0.72, "c1": 1.5, "c2": 1.5, "v_max": 0.2}
SHOW_PLOTS  = True                     # โชว์กราฟตอนจบ
VERBOSE     = False                    # แสดง log ราย fold

INPUT_COLS  = [3,6,8,10,11,12,13,14]
TARGET_COL  = 5
# ========================================

# ---------- Utils ----------
def mean_std(a):
    a = np.asarray(a, dtype=float)
    return (float(a.mean()) if a.size else float('nan'),
            float(a.std(ddof=1)) if a.size>1 else 0.0)

def blocked_time_folds(n, k=10):
    base = n // k; rem = n % k
    sizes = [base + (1 if i < rem else 0) for i in range(k)]
    folds = []; s = 0
    for fs in sizes:
        folds.append((s, s+fs)); s += fs
    return folds

# ---------- Minimal XLSX Reader (first sheet) ----------
def _xlsx_colname_to_idx(col):
    s=0
    for ch in col:
        s = s*26 + (ord(ch)-ord('A')+1)
    return s-1

def _xlsx_parse_cell_ref(ref):
    i=0
    while i<len(ref) and ref[i].isalpha(): i+=1
    col = ref[:i]; row = ref[i:]
    return int(row)-1, _xlsx_colname_to_idx(col)

def load_xlsx_first_sheet_numeric(path):
    with zipfile.ZipFile(path, 'r') as z:
        sheets_files = sorted([n for n in z.namelist()
                               if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')])
        if not sheets_files:
            raise RuntimeError("No worksheet XML found in xlsx.")
        sheet_name = sheets_files[0]

        shared = []
        if 'xl/sharedStrings.xml' in z.namelist():
            sst = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in sst.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si'):
                texts=[]
                for t in si.findall('.//{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                    texts.append(t.text or "")
                shared.append("".join(texts))

        sh = ET.fromstring(z.read(sheet_name))
        nsu = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        data = {}
        max_r = -1; max_c = -1
        for c in sh.findall('.//'+nsu+'c'):
            ref = c.get('r')
            if ref is None: continue
            r_idx, c_idx = _xlsx_parse_cell_ref(ref)
            if r_idx>max_r: max_r=r_idx
            if c_idx>max_c: max_c=c_idx
            t = c.get('t'); v = c.find(nsu+'v')
            val = None
            if t == 'inlineStr':
                is_elem = c.find(nsu+'is')
                if is_elem is not None:
                    txts = [t.text or "" for t in is_elem.findall('.//'+nsu+'t')]
                    val = "".join(txts)
            elif t == 's' and v is not None:
                try:
                    idx = int(v.text.strip())
                    val = shared[idx] if 0<=idx<len(shared) else ""
                except:
                    val = ""
            elif v is not None:
                try:
                    val = float(v.text.strip())
                except:
                    val = v.text.strip()
            data[(r_idx, c_idx)] = val

        rows = []
        for r in range(max_r+1):
            row = []
            for c in range(15):  # 0..14
                v = data.get((r,c), None)
                if c in (0,1):
                    row.append(None)  # Date/Time not used
                else:
                    if isinstance(v,str):
                        s = v.strip().replace(',', '.')
                        try: row.append(float(s))
                        except: row.append(None)
                    else:
                        row.append(float(v) if isinstance(v,(int,float)) else None)
            rows.append(row)
        return rows

# ---------- Data prep ----------
def build_supervised(rows, input_cols, target_col, lag_hours):
    X, y = [], []
    n = len(rows)
    for t in range(n - lag_hours):
        ok = True; feats = []
        for c in input_cols:
            val = rows[t][c]
            if val is None or val == -200: ok=False; break
            feats.append(val)
        tgt = rows[t+lag_hours][target_col]
        if tgt is None or tgt == -200: ok=False
        if ok:
            X.append(feats); y.append(tgt)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float)

def fit_minmax(X):
    mn = X.min(axis=0); mx = X.max(axis=0)
    mx = np.where(np.abs(mx-mn) < 1e-12, mn+1.0, mx)
    return mn, mx

def apply_minmax(X, mn, mx):
    return (X - mn) / (mx - mn)

# ---------- MLP ----------
class MLP:
    def __init__(self, sizes):
        self.sizes = list(sizes)
        self.L = len(self.sizes)-1
        self.shapes = [(self.sizes[l+1], self.sizes[l]) for l in range(self.L)]
        self.total_params = sum(r*c + r for (r,c) in self.shapes)
    def _vec_to_params(self, vec):
        params = []; k = 0
        for (r,c) in self.shapes:
            W = np.array(vec[k:k+r*c], dtype=float).reshape(r,c); k += r*c
            B = np.array(vec[k:k+r],   dtype=float).reshape(r,1); k += r
            params.append((W,B))
        return params
    def forward_batch(self, X, vec):
        params = self._vec_to_params(vec)
        A = X.T
        for l,(W,B) in enumerate(params):
            Z = (W @ A) + B
            A = np.tanh(Z) if l < self.L-1 else Z
        return A.T.ravel()

# ---------- Metric ----------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat))) if y.size else float('inf')

# ---------- PSO ----------
class PSO:
    def __init__(self, dim, fitness_fn, swarm=20, iters=120, w=0.72, c1=1.5, c2=1.5, v_max=0.2, init_lo=-0.5, init_hi=0.5):
        self.dim=dim; self.fitness_fn=fitness_fn
        self.swarm=swarm; self.iters=iters
        self.w=w; self.c1=c1; self.c2=c2; self.v_max=v_max
        self.init_lo=init_lo; self.init_hi=init_hi
    def run(self, verbose=False):
        pos = np.random.uniform(self.init_lo, self.init_hi, size=(self.swarm, self.dim))
        vel = np.zeros((self.swarm, self.dim), dtype=float)
        pbest_pos = pos.copy()
        pbest_fit = np.array([self.fitness_fn(p) for p in pos], dtype=float)
        g_idx = int(np.argmin(pbest_fit))
        gbest_pos = pbest_pos[g_idx].copy()
        gbest_fit = float(pbest_fit[g_idx])
        if verbose:
            print(f"    PSO init gBest(train_MAE)={gbest_fit:.5f}")
        for it in range(self.iters):
            rp = np.random.rand(self.swarm, 1); rg = np.random.rand(self.swarm, 1)
            vel = self.w*vel + self.c1*rp*(pbest_pos-pos) + self.c2*rg*(gbest_pos-pos)
            np.clip(vel, -self.v_max, self.v_max, out=vel)
            pos += vel
            fits = np.array([self.fitness_fn(p) for p in pos], dtype=float)
            improved = fits < pbest_fit
            pbest_fit[improved] = fits[improved]
            pbest_pos[improved] = pos[improved]
            g_idx = int(np.argmin(pbest_fit))
            if pbest_fit[g_idx] < gbest_fit:
                gbest_fit = float(pbest_fit[g_idx]); gbest_pos = pbest_pos[g_idx].copy()
            if verbose and ((it+1) % max(1,self.iters//5) == 0):
                print(f"    iter {it+1}/{self.iters}: gBest(train_MAE)={gbest_fit:.5f}")
        return gbest_pos, gbest_fit

# ---------- Evaluation ----------
def evaluate_architecture(X, y, arch_hidden, lag_tag_days, pso_cfg, verbose=False):
    mlp = MLP([X.shape[1]] + arch_hidden + [1])
    folds = blocked_time_folds(len(X), 10)
    test_maes, train_maes = [], []
    t0 = time.time()
    for fi,(ts,te) in enumerate(folds):
        if ts == 0:  # no history
            continue
        Xtr_raw, ytr = X[:ts], y[:ts]
        Xte_raw, yte = X[ts:te], y[ts:te]
        mn, mx = fit_minmax(Xtr_raw)
        Xtr = apply_minmax(Xtr_raw, mn, mx)
        Xte = apply_minmax(Xte_raw, mn, mx)
        def fitness(vec): return mae(ytr, mlp.forward_batch(Xtr, vec))
        pso = PSO(dim=mlp.total_params, fitness_fn=fitness, **pso_cfg)
        if verbose:
            print(f"  Fold {fi:02d}: train={len(Xtr)} test={len(Xte)} lag={lag_tag_days} arch={arch_hidden}")
        gvec, tr_fit = pso.run(verbose=verbose)
        te_fit = mae(yte, mlp.forward_batch(Xte, gvec))
        train_maes.append(tr_fit); test_maes.append(te_fit)
        if verbose:
            print(f"    -> train_MAE={tr_fit:.5f}, test_MAE={te_fit:.5f}")
    elapsed = time.time() - t0
    m_train, s_train = mean_std(train_maes)
    m_test , s_test  = mean_std(test_maes)
    return {
        "arch": arch_hidden[:], "lag": lag_tag_days,
        "train_mean": m_train, "train_std": s_train,
        "test_mean": m_test, "test_std": s_test,
        "test_best": float(np.min(test_maes)) if test_maes else float('inf'),
        "test_worst": float(np.max(test_maes)) if test_maes else float('inf'),
        "folds_used": len(test_maes), "time_sec": elapsed,
    }

# ---------- Plot ----------
def make_plots(arch_list, s5, s10, show_plots=True):
    try:
        import matplotlib.pyplot as plt
        labels = ["-".join(map(str,a)) for a in arch_list]
        x = np.arange(len(labels)); width = 0.35
        plt.figure()
        plt.title("MAE(test) by Architecture (Days)")
        plt.bar(x - width/2, s5,  width, label="5 days")
        plt.bar(x + width/2, s10, width, label="10 days")
        plt.xticks(x, labels); plt.ylabel("MAE"); plt.legend(); plt.tight_layout()
        plt.savefig("mae_by_arch_days.png", dpi=150); print("Saved: mae_by_arch_days.png")
        plt.figure()
        plt.title("MAE(test) — 5 days")
        plt.plot(x, s5, marker='o'); plt.xticks(x, labels); plt.ylabel("MAE"); plt.tight_layout()
        plt.savefig("mae_5d_line.png", dpi=150); print("Saved: mae_5d_line.png")
        plt.figure()
        plt.title("MAE(test) — 10 days")
        plt.plot(x, s10, marker='o'); plt.xticks(x, labels); plt.ylabel("MAE"); plt.tight_layout()
        plt.savefig("mae_10d_line.png", dpi=150); print("Saved: mae_10d_line.png")
        if show_plots:
            plt.show()
    except Exception as e:
        print("Plot skipped (matplotlib unavailable):", e)

# ---------- Runner ----------
def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Data file not found: " + DATA_PATH)
    print("Loading XLSX...")
    rows = load_xlsx_first_sheet_numeric(DATA_PATH)
    print("Rows loaded:", len(rows))
    summaries = []
    for d in DAYS_LIST:
        lag_hours = int(d)*24
        print(f"\n=== Task: Predict Benzene {d} days ===")
        X, y = build_supervised(rows, INPUT_COLS, TARGET_COL, lag_hours)
        print(f"Prepared samples: {len(X)}")
        for arch in ARCHS:
            s = evaluate_architecture(X, y, arch, f"{d} days", PSO_CFG, verbose=VERBOSE)
            summaries.append(s)
            print(f"Arch {arch} -> MAE(test) mean={s['test_mean']:.4f} (std={s['test_std']:.4f}, "
                  f"best={s['test_best']:.4f}, worst={s['test_worst']:.4f}) | folds={s['folds_used']} | time={s['time_sec']:.1f}s")
    summaries.sort(key=lambda s: s["test_mean"])
    print("\n=== Summary (sorted by MAE test mean) ===")
    for s in summaries:
        print(f"Task {s['lag']:>4} | Arch {s['arch']!s:<8} | "
              f"MAE(test) {s['test_mean']:.4f} ± {s['test_std']:.4f} "
              f"(best {s['test_best']:.4f}, worst {s['test_worst']:.4f}) | "
              f"train_mean {s['train_mean']:.4f} | folds {s['folds_used']} | {s['time_sec']:.1f}s")
    def _get_mean(arch, lag_str):
        for s in summaries:
            if s["arch"] == arch and s["lag"] == lag_str:
                return s["test_mean"]
        return np.nan

    s5  = [_get_mean(a, "5 days")  for a in ARCHS]
    s10 = [_get_mean(a, "10 days") for a in ARCHS]
    make_plots(ARCHS, s5, s10, show_plots=SHOW_PLOTS)


if __name__ == "__main__":
    main()