"""Dữ liệu: QM9 (tải, đọc, lọc, chia tập), dataset HDF5, năng lượng nguyên tử hóa và đặc trưng cho ML."""
import json
import os
import re
import tarfile
import urllib.request

import h5py
import numpy as np
import pandas as pd

HA2KCAL = 627.5094740631
QM9_URL = "https://ndownloader.figshare.com/files/3195389"            # dsgdb9nsd.xyz.tar.bz2 (133885 phân tử)
UNCHAR_URL = "https://figshare.com/ndownloader/files/3195404"         # uncharacterized.txt (3054 phân tử lỗi kiểm tra)
ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
ELEMENTS = ["H", "C", "N", "O", "F"]


# ---------------------------------------------------------------- tải và đọc QM9
def download(url, dest, overwrite=False):
    """Tải file nếu chưa có; kiểm tra kích thước > 0."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not overwrite:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    assert os.path.getsize(dest) > 0, f"Tải về rỗng: {url}"
    return dest


def parse_qm9_xyz(text):
    """Đọc một file .xyz của QM9 (định dạng mở rộng). Hình học tính bằng Å, U0 bằng Hartree (B3LYP/6-31G(2df,p))."""
    lines = text.replace("*^", "E").split("\n")          # QM9 ghi số mũ dạng 1.5*^-6
    n = int(lines[0])
    props = lines[1].split()                              # gdb idx A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv
    atoms = [lines[2 + i].split() for i in range(n)]
    smiles = lines[n + 3].split()
    return {
        "gdb_id": int(props[1]), "n_atoms": n,
        "symbols": [a[0] for a in atoms],
        "coords_A": [[float(a[1]), float(a[2]), float(a[3])] for a in atoms],
        "smiles": smiles[0] if smiles else "", "smiles_relaxed": smiles[1] if len(smiles) > 1 else "",
        "U0_B3LYP_Ha": float(props[12]),
    }


def formula_of(symbols):
    cnt = {e: symbols.count(e) for e in set(symbols)}
    order = ["C", "H"] + sorted(e for e in cnt if e not in ("C", "H"))
    return "".join(f"{e}{cnt[e] if cnt[e] > 1 else ''}" for e in order if e in cnt)


def scan_qm9_tar(tar_path, max_heavy=5, out_jsonl=None, progress_every=20000):
    """Đọc toàn bộ tar một lần: thống kê số phân tử theo số nguyên tử nặng, giữ lại phân tử ≤ max_heavy.

    Trả về (list[dict] ứng viên, dict thống kê). Ghi ứng viên vào out_jsonl nếu được chỉ định.
    """
    keep, by_heavy, n_total = [], {}, 0
    with tarfile.open(tar_path, "r:bz2") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".xyz"):
                continue
            text = tf.extractfile(member).read().decode("utf-8", errors="replace")
            n_total += 1
            head = text.split("\n")
            n = int(head[0])
            heavy = sum(1 for ln in head[2:2 + n] if ln and ln[0] != "H")
            by_heavy[heavy] = by_heavy.get(heavy, 0) + 1
            if heavy <= max_heavy:
                rec = parse_qm9_xyz(text)
                rec.update(n_heavy=heavy, formula=formula_of(rec["symbols"]))
                keep.append(rec)
            if progress_every and n_total % progress_every == 0:
                print(f"  ... đã đọc {n_total} phân tử")
    stats = {"n_total": n_total, "by_n_heavy": dict(sorted(by_heavy.items())), "n_candidates": len(keep),
             "max_heavy": max_heavy}
    if out_jsonl:
        with open(out_jsonl, "w") as f:
            for rec in keep:
                f.write(json.dumps(rec) + "\n")
    return keep, stats


def read_uncharacterized(path):
    """Chỉ số GDB của các phân tử trong uncharacterized.txt (token đầu mỗi dòng dữ liệu là số nguyên)."""
    ids = set()
    for line in open(path, errors="replace"):
        tok = line.split()
        if tok and tok[0].isdigit():
            ids.add(int(tok[0]))
    return ids


def n_electrons(symbols):
    return sum(ATOMIC_NUMBER[s] for s in symbols)


def write_xyz(path, symbols, coords_A, comment=""):
    with open(path, "w") as f:
        f.write(f"{len(symbols)}\n{comment}\n")
        for s, c in zip(symbols, coords_A):
            f.write(f"{s:2s} {c[0]: .8f} {c[1]: .8f} {c[2]: .8f}\n")


def random_split(names, fractions=(0.6, 0.2, 0.2), seed=42):
    """Chia ngẫu nhiên theo phân tử (mỗi phân tử chỉ ở một tập). Trả về dict name -> 'train'|'val'|'test'."""
    rng = np.random.default_rng(seed)
    order = list(names)
    perm = rng.permutation(len(order))
    n = len(order)
    n_tr = int(round(fractions[0] * n))
    n_va = int(round(fractions[1] * n))
    out = {}
    for rank, i in enumerate(perm):
        out[order[i]] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")
    return out


# ---------------------------------------------------------------- năng lượng nguyên tử hóa
def composition(symbols):
    return {e: symbols.count(e) for e in sorted(set(symbols))}


def atomization_energy(table, name, e_col, dE=None):
    """AE = Σ E_nguyên_tử − E_phân_tử (Hartree) theo cột năng lượng e_col; dE: dict name -> hiệu chỉnh ML (Ha)."""
    row = table.loc[name]
    get = (lambda k: dE[k]) if dE is not None else (lambda k: 0.0)
    e_atoms = sum(n * (table.loc[f"atom_{el}", e_col] + get(f"atom_{el}")) for el, n in row["counts"].items())
    return float(e_atoms - (row[e_col] + get(name)))


def ae_error_kcal(table, names, e_col, dE=None, ref_col="e_ccsdt"):
    """Sai số năng lượng nguyên tử hóa so với CCSD(T) (kcal/mol) cho các phân tử trong `names`."""
    return np.array([(atomization_energy(table, n, e_col, dE) - atomization_energy(table, n, ref_col)) * HA2KCAL
                     for n in names])


def error_stats(err):
    err = np.asarray(err, dtype=float)
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max": float(np.max(np.abs(err))), "mean": float(np.mean(err)), "n": int(len(err))}


def bootstrap_mae_ci(err, n_boot=2000, seed=0):
    """Khoảng tin cậy 95% bootstrap của MAE."""
    rng = np.random.default_rng(seed)
    a = np.abs(np.asarray(err, dtype=float))
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n_boot)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------- HDF5
UNITS = {
    "energy": "Hartree (Ha); AE và sai số báo bằng kcal/mol (1 Ha = 627.5095 kcal/mol)",
    "length": "Bohr cho coords_bohr, weights (Bohr^3); hình học gốc (QM9) bằng Å",
    "rho": "mật độ electron tổng (e/Bohr^3) của PBE tự hợp trên lưới tích phân PySCF",
    "delta": "delta = E_CCSD(T) - E_PBE[n_PBE] cùng hình học và basis (Ha)",
}


def build_h5(h5_path, systems, pbe_dir, label_dir, meta):
    """Ghép mật độ PBE + nhãn CCSD(T) thành HDF5. `systems`: DataFrame (name, split, is_atom, symbol, ...).

    Trả về manifest (mỗi hệ một dòng: included, reason). Hệ không hội tụ/thiếu file bị loại và ghi lý do.
    """
    rows = []
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["units"] = json.dumps(UNITS, ensure_ascii=False)
        h5.attrs["meta"] = json.dumps(meta, ensure_ascii=False, default=str)
        grp = h5.create_group("systems")
        for r in systems.itertuples():
            row = {"name": r.name, "split": r.split, "included": False, "reason": ""}
            npz, pj, lj = (f"{pbe_dir}/{r.name}.npz", f"{pbe_dir}/{r.name}.json", f"{label_dir}/{r.name}.json")
            if not (os.path.exists(npz) and os.path.exists(pj) and os.path.exists(lj)):
                row["reason"] = "thiếu file PBE hoặc CCSD(T)"
            else:
                p, lab, d = json.load(open(pj)), json.load(open(lj)), np.load(npz)
                if not p["converged"]:
                    row["reason"] = "PBE không hội tụ"
                elif not (lab["ccsd_converged"] and lab["hf_converged"]):
                    row["reason"] = "CCSD(T)/HF không hội tụ"
                elif abs(p["int_rho"] - p["nelec"]) > 1e-3:
                    row["reason"] = f"∫ρ lệch {p['int_rho'] - p['nelec']:+.1e}"
                else:
                    g = grp.create_group(r.name)
                    g.create_dataset("rho", data=d["rho"], compression="gzip")
                    g.create_dataset("gnorm", data=d["gnorm"], compression="gzip")
                    g.create_dataset("weights", data=d["weights"], compression="gzip")
                    g.create_dataset("coords_bohr", data=d["coords"].astype(np.float32), compression="gzip")
                    g.create_dataset("symbols", data=np.array(json.loads(r.symbols_json), dtype="S2"))
                    g.create_dataset("coords_A", data=np.array(json.loads(r.coords_json)))
                    rho, w = d["rho"], d["weights"]
                    g.attrs.update(
                        split=r.split, is_atom=bool(r.is_atom), symbol=str(r.symbol), n_atoms=int(r.n_atoms),
                        counts=json.dumps(json.loads(r.counts_json)),
                        e_pbe=p["e_tot"], exc_pbe=p["exc"], e_ccsdt=lab["e_ccsdt"], e_hf=lab["e_hf"],
                        delta=lab["e_ccsdt"] - p["e_tot"], int_n43=float(np.sum(w * rho ** (4 / 3))),
                        int_rho=p["int_rho"], nelec=p["nelec"], nbf=p["nbf"], spin=p["spin"])
                    row["included"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def load_system(h5_path, name):
    with h5py.File(h5_path, "r") as h5:
        g = h5["systems"][name]
        out = {"name": name, "rho": g["rho"][()], "gnorm": g["gnorm"][()], "weights": g["weights"][()],
               "symbols": [s.decode() for s in g["symbols"][()]], "coords_A": g["coords_A"][()]}
        out.update({k: (json.loads(v) if k == "counts" else (v.item() if hasattr(v, "item") else v))
                    for k, v in g.attrs.items()})
    return out


def energy_table(h5_path):
    """Bảng vô hướng của mọi hệ (không đọc mật độ): index = name; cột counts là dict."""
    rows = []
    with h5py.File(h5_path, "r") as h5:
        for name, g in h5["systems"].items():
            a = dict(g.attrs)
            rows.append({"name": name, "split": a["split"], "is_atom": bool(a["is_atom"]), "symbol": a["symbol"],
                         "n_atoms": int(a["n_atoms"]), "counts": json.loads(a["counts"]),
                         "e_pbe": float(a["e_pbe"]), "e_ccsdt": float(a["e_ccsdt"]), "e_hf": float(a["e_hf"]),
                         "delta": float(a["delta"]), "exc_pbe": float(a["exc_pbe"]), "int_n43": float(a["int_n43"]),
                         "nelec": int(a["nelec"])})
    return pd.DataFrame(rows).set_index("name")


# ---------------------------------------------------------------- đặc trưng
def reduced_gradient(rho, gnorm, rho_floor=1e-10, s_max=20.0):
    """s = |∇ρ| / (2 (3π²)^(1/3) ρ^(4/3)), cắt về [0, s_max]; ρ được chặn dưới rho_floor."""
    r = np.maximum(rho, rho_floor)
    return np.clip(gnorm / (2 * (3 * np.pi ** 2) ** (1 / 3) * r ** (4 / 3)), 0.0, s_max), r
