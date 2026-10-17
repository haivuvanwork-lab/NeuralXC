"""Tiện ích DFT / CCSD(T) cho phân tử bằng PySCF (đơn vị nguyên tử: Hartree, Bohr; đầu vào hình học tính bằng Å)."""
import os
import platform
import subprocess
import time

import numpy as np

HA2KCAL = 627.5094740631
HA2EV = 27.211386245988
ATOM_SPIN = {"H": 1, "C": 2, "N": 3, "O": 2, "F": 1}    # 2S của nguyên tử ở trạng thái cơ bản (số electron độc thân)
DEFAULTS = dict(basis="cc-pvdz", grid_level=3, conv_tol=1e-9, max_cycle=100)


def resources():
    """Báo cáo tài nguyên: CPU/RAM cho PySCF, GPU cho JAX."""
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["ram_GB"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        info["ram_GB"] = None
    try:
        import pyscf
        info["pyscf"] = pyscf.__version__
    except Exception as e:  # pragma: no cover
        info["pyscf"] = f"lỗi: {e!r}"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        info["gpu"] = out.stdout.strip() or None
    except Exception:
        info["gpu"] = None
    return info


def build_mol(symbols, coords_A, basis=DEFAULTS["basis"], spin=0, charge=0, max_memory=None):
    """Dựng Mole PySCF. `spin` = số electron độc thân (2S)."""
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = [(s, tuple(float(x) for x in c)) for s, c in zip(symbols, coords_A)]
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = int(spin)
    mol.charge = int(charge)
    mol.verbose = 0
    if max_memory:
        mol.max_memory = max_memory
    mol.build()
    return mol


def build_atom(symbol, basis=DEFAULTS["basis"], max_memory=None):
    return build_mol([symbol], [[0.0, 0.0, 0.0]], basis=basis, spin=ATOM_SPIN[symbol], max_memory=max_memory)


def run_ks(mol, xc="pbe", grid_level=DEFAULTS["grid_level"], conv_tol=DEFAULTS["conv_tol"],
           max_cycle=DEFAULTS["max_cycle"]):
    """SCF Kohn-Sham (RKS nếu vỏ kín, UKS nếu có electron độc thân). Trả về (info, mf).

    Nếu không hội tụ: thử level shift, rồi SCF bậc hai (newton). `info['attempts']` ghi lại số lần thử.
    """
    from pyscf import dft
    mf = (dft.UKS if mol.spin != 0 else dft.RKS)(mol)
    mf.xc, mf.conv_tol, mf.max_cycle = xc, conv_tol, max_cycle
    mf.grids.level = grid_level
    t0 = time.time()
    mf.kernel()
    attempts = 1
    if not mf.converged:
        attempts = 2
        mf.level_shift, mf.max_cycle = 0.3, 3 * max_cycle
        mf.kernel(dm0=mf.make_rdm1())
    if not mf.converged:
        attempts = 3
        try:
            mf2 = mf.newton()
            mf2.kernel(dm0=mf.make_rdm1())
            if mf2.converged:
                mf = mf2
        except Exception:
            pass
    summary = getattr(mf, "scf_summary", {}) or {}
    info = {"xc": xc, "e_tot": float(mf.e_tot), "converged": bool(mf.converged), "attempts": attempts,
            "wall_s": round(time.time() - t0, 2), "nbf": int(mol.nao), "nelec": int(mol.nelectron),
            "spin": int(mol.spin), "grid_level": grid_level,
            "exc": float(summary["exc"]) if "exc" in summary else None}
    return info, mf


def density_on_grid(mol, mf):
    """Mật độ electron tổng ρ và |∇ρ| trên lưới tích phân của mf (tính theo khối để tiết kiệm RAM).

    Trả về dict: coords (N,3) Bohr, weights (N,), rho (N,), gnorm (N,), dm (ma trận mật độ tổng, và
    dm_spin nếu UKS), int_rho = ∫ρ.
    """
    from pyscf import dft
    dm = np.asarray(mf.make_rdm1())
    dm_tot = dm.sum(axis=0) if dm.ndim == 3 else dm
    ni = dft.numint.NumInt()
    coords, wts, rho, gn = [], [], [], []
    for ao, mask, weight, c in ni.block_loop(mol, mf.grids, mol.nao, 1, max_memory=2000):
        r4 = ni.eval_rho(mol, ao, dm_tot, mask, "GGA")
        rho.append(r4[0])
        gn.append(np.linalg.norm(r4[1:4], axis=0))
        wts.append(weight)
        coords.append(c)
    out = {"coords": np.vstack(coords), "weights": np.concatenate(wts), "rho": np.concatenate(rho),
           "gnorm": np.concatenate(gn), "dm": dm_tot}
    out["int_rho"] = float(np.sum(out["weights"] * out["rho"]))
    return out


def run_ccsdt(mol, conv_tol=1e-8, max_cycle=100):
    """CCSD(T) toàn electron trên tham chiếu HF (RHF cho vỏ kín, UHF/UCCSD(T) cho nguyên tử có spin)."""
    from pyscf import cc, scf
    t0 = time.time()
    unrestricted = mol.spin != 0
    hf = (scf.UHF if unrestricted else scf.RHF)(mol)
    hf.conv_tol = 1e-10
    hf.kernel()
    if not hf.converged:
        hf = hf.newton()
        hf.kernel()
    if mol.nelectron == 1:                 # nguyên tử H: không có tương quan electron
        e_corr, e_t, conv = 0.0, 0.0, True
    else:
        mycc = (cc.UCCSD if unrestricted else cc.CCSD)(hf)
        mycc.conv_tol, mycc.max_cycle = conv_tol, max_cycle
        mycc.kernel()
        e_corr, conv = float(mycc.e_corr), bool(mycc.converged)
        e_t = float(mycc.ccsd_t())
    return {"e_hf": float(hf.e_tot), "hf_converged": bool(hf.converged), "e_ccsd": float(hf.e_tot) + e_corr,
            "e_t": e_t, "e_ccsdt": float(hf.e_tot) + e_corr + e_t, "ccsd_converged": conv,
            "wall_s": round(time.time() - t0, 2), "nbf": int(mol.nao), "spin": int(mol.spin)}
