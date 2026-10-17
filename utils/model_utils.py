"""Mô hình ML-XC (JAX + Flax + Optax).

Hiệu chỉnh XC cục bộ phụ thuộc (ρ, s) lên mật độ PBE:
    ΔE_θ[n] = Σ_i w_i ρ_i f_θ(log10 ρ_i, asinh s_i)           [Ha]
    E_ML = E_PBE[n_PBE] + ΔE_θ[n_PBE]
huấn luyện để khớp nhãn  Δ = E_CCSD(T) − E_PBE[n_PBE].
Lớp cuối khởi tạo bằng 0 nên mô hình ban đầu chính là PBE. Mô hình chỉ thấy ρ và |∇ρ| (không thấy τ hay thông tin
không địa phương).
"""
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

HA2KCAL = 627.5094740631
C_LO = 2.273                      # hằng số Lieb–Oxford: E_xc >= -C_LO ∫ n^(4/3) dr
BUCKET = 16384                    # đệm số điểm lưới lên bội số này để giảm số lần biên dịch jit


class XCNet(nn.Module):
    hidden: tuple = (32, 32)

    @nn.compact
    def __call__(self, x):
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h)(x))
        return nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)(x)[..., 0]


def init_params(net, seed=0):
    return net.init(jax.random.PRNGKey(seed), jnp.zeros((1, 2)))["params"]


def point_features(rho, s):
    return jnp.stack([(jnp.log10(rho) + 4.0) / 3.0, jnp.arcsinh(s)], axis=-1)


def _pad(a, n, value):
    return np.concatenate([a, np.full(n - len(a), value, dtype=a.dtype)]) if n > len(a) else a


def make_dataset(systems, reduced_gradient_fn):
    """systems: list dict từ data_utils.load_system. Đệm lên bội của BUCKET với trọng số 0 (không đóng góp)."""
    ds = []
    for sy in systems:
        s, r = reduced_gradient_fn(sy["rho"], sy["gnorm"])
        n = int(np.ceil(len(r) / BUCKET) * BUCKET)
        w = sy["weights"]
        ds.append({"name": sy["name"], "n_atoms": float(sy["n_atoms"]),
                   "rho": jnp.asarray(_pad(r.astype(np.float32), n, 1e-6)),
                   "s": jnp.asarray(_pad(s.astype(np.float32), n, 0.0)),
                   "w": jnp.asarray(_pad(w.astype(np.float32), n, 0.0)),
                   "delta": float(sy["delta"]), "exc_pbe": float(sy["exc_pbe"]), "int_n43": float(sy["int_n43"])})
    return ds


def make_fns(net):
    """Tạo các hàm đã jit cho một kiến trúc: ΔE, và (loss, grad) cho một mẫu."""
    def delta_e(params, rho, s, w):
        f = net.apply({"params": params}, point_features(rho, s))
        return jnp.sum(w * rho * f)

    def sample_loss(params, rho, s, w, n_atoms, delta, exc_pbe, int_n43, lam_lo):
        de = delta_e(params, rho, s, w)
        mse = ((de - delta) * HA2KCAL / n_atoms) ** 2                       # (kcal/mol/nguyên tử)^2
        viol = jax.nn.relu(-(exc_pbe + de) - C_LO * int_n43) * HA2KCAL / n_atoms
        return mse + lam_lo * viol ** 2, viol > 0

    return jax.jit(delta_e), jax.jit(jax.value_and_grad(sample_loss, has_aux=True))


def predict_delta(de_fn, params, ds):
    """ΔE_θ (Ha) cho từng mẫu; trả về dict name -> float."""
    return {d["name"]: float(de_fn(params, d["rho"], d["s"], d["w"])) for d in ds}


def train(net, train_ds, eval_fn, lr=1e-3, weight_decay=0.0, epochs=300, batch_size=8, patience=40,
          lam_lo=1.0, seed=0, verbose=True, log_every=10):
    """Fit trên train_ds. `eval_fn(params) -> (số đo validation, dict phụ)` dùng để dừng sớm/chọn mô hình
    (số nhỏ hơn là tốt hơn). Không bao giờ nhìn tập test."""
    de_fn, vg = make_fns(net)
    params = init_params(net, seed)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, weight_decay=weight_decay))
    state = opt.init(params)
    rng = np.random.default_rng(seed)
    best, best_params, bad, history = np.inf, params, 0, []
    for ep in range(epochs):
        order = rng.permutation(len(train_ds))
        tot, n_lo = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            grads = None
            for i in idx:
                d = train_ds[i]
                (l, active), g = vg(params, d["rho"], d["s"], d["w"], d["n_atoms"], d["delta"], d["exc_pbe"],
                                    d["int_n43"], lam_lo)
                tot += float(l)
                n_lo += int(active)
                g = jax.tree_util.tree_map(lambda x: x / len(idx), g)
                grads = g if grads is None else jax.tree_util.tree_map(jnp.add, grads, g)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        val, extra = eval_fn(params, de_fn)
        history.append({"epoch": ep, "train_loss": tot / len(train_ds), "val": val, "lo_active": n_lo})
        if val < best - 1e-9:
            best, best_params, bad = val, params, 0
        else:
            bad += 1
        if verbose and (ep % log_every == 0 or bad == 0):
            print(f"epoch {ep:4d}  train loss {tot / len(train_ds):9.3f}  val {val:8.3f}  (LO vi phạm: {n_lo})")
        if bad >= patience:
            break
    return best_params, {"best_val": float(best), "history": history, "lr": lr, "weight_decay": weight_decay,
                         "hidden": list(net.hidden), "epochs_run": len(history)}, de_fn


# ---------------------------------------------------------------- baseline theo nguyên tố
def fit_element_baseline(table, names, elements=("H", "C", "N", "O", "F")):
    """Δ ≈ Σ_el N_el c_el (bình phương tối thiểu trên `names`). Trả về dict element -> c (Ha)."""
    A = np.array([[table.loc[n, "counts"].get(e, 0) for e in elements] for n in names], dtype=float)
    y = np.array([table.loc[n, "delta"] for n in names])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(zip(elements, c))


def element_baseline_predict(table, names, coef):
    return {n: float(sum(k * coef[e] for e, k in table.loc[n, "counts"].items())) for n in names}


# ---------------------------------------------------------------- lưu / nạp
def save_checkpoint(path, params, config):
    with open(path, "wb") as f:
        f.write(serialization.to_bytes(params))
    with open(path + ".json", "w") as f:
        json.dump(config, f, indent=2, default=float)


def load_checkpoint(path):
    config = json.load(open(path + ".json"))
    net = XCNet(hidden=tuple(config["hidden"]))
    template = init_params(net, 0)
    params = serialization.from_bytes(template, open(path, "rb").read())
    return net, params, config
