import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from RAHFL.Dataset.utils import (
    partition_data, get_dataloader, init_nets,
    generate_public_data_indexs, mkdirs, init_logs
)
from RAHFL.HHF.loss import DCLLoss, SupConLoss

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

ROOT_DIR = Path.cwd()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED                = 0
TRAIN_BATCH_SIZE    = 256
TEST_BATCH_SIZE     = 512
COMMUNICATION_EPOCH = 40
LEARNING_RATE       = 0.001
N_PARTICIPANTS      = 4
NETS_NAME_LIST      = ['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2']
PRIVATE_DATA_LEN    = 10000
PUBLIC_DATA_LEN     = 5000
N_CLASSES           = 10
DEVICE_IDS          = [0]

# Primary ablation corruption setting
ABLATION_XI_PRIV = 0.5
ABLATION_XI_PUB  = 0.5
ABLATION_XI_TEST = 0.5

# Dataset paths — adjust to your directory layout
PRIVATE_DIR    = ROOT_DIR / "Dataset" / "cifar_10_c"
PUBLIC_DIR     = ROOT_DIR / "Dataset" / "cifar_100_c"
PUBLIC_DIR_ORG = ROOT_DIR / "Dataset" / "cifar_100"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


# ═══════════════════════════════════════════════════════════════════════════════
# *** CORE FIX: SAFE MULTI-VIEW BATCH UNPACKING ***
# ═══════════════════════════════════════════════════════════════════════════════

def unpack_augmix_batch(images):
    if isinstance(images, (list, tuple)):
        # ── Format A: list of view tensors ───────────────────────────────────
        n = len(images)
        img0 = images[0]
        img1 = images[1] if n > 1 else images[0]
        img2 = images[2] if n > 2 else images[0]
        img3 = images[3] if n > 3 else images[0]
        return img0, img1, img2, img3

    elif isinstance(images, torch.Tensor):
        # ── Format B: stacked along channel dimension ─────────────────────
        # images.shape = (B, C_total, H, W)  where C_total = n_views * 3
        C_total = images.shape[1]

        if C_total % 3 != 0:
            raise ValueError(
                f"Stacked image tensor has {C_total} channels, "
                f"which is not divisible by 3. "
                f"Cannot determine number of views."
            )

        n_views = C_total // 3
        views   = list(torch.split(images, 3, dim=1))   # list of (B,3,H,W)

        img0 = views[0]
        img1 = views[1] if n_views > 1 else views[0]
        img2 = views[2] if n_views > 2 else views[0]
        img3 = views[3] if n_views > 3 else views[0]
        return img0, img1, img2, img3

    else:
        raise TypeError(
            f"unpack_augmix_batch: unexpected images type {type(images)}. "
            f"Expected list, tuple, or torch.Tensor."
        )


def diagnose_dataloader_format(dataloader, name="dataloader"):
    """
    Print the format of the first batch to assist debugging.
    Safe to call before the main training loop.
    """
    for images, labels in dataloader:
        print(f"\n[DIAG] {name}:")
        if isinstance(images, torch.Tensor):
            print(f"  Format B — Stacked tensor: shape={tuple(images.shape)}")
            n_views = images.shape[1] // 3
            print(f"  Inferred n_views = {n_views}")
        elif isinstance(images, (list, tuple)):
            print(f"  Format A — List of {len(images)} tensors, "
                  f"each shape={tuple(images[0].shape)}")
        print(f"  Labels shape: {tuple(labels.shape)}")
        break


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE HYPERPARAMETER COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_hyperparams(xi_priv, xi_pub, xi_test):
    """Compute all APEX-FL adaptive hyperparameters from corruption scalars."""
    return {
        'teacher_temp' : max(0.3,  1.0 - 0.7  * xi_pub),
        'mce_n_crops'  : round(1 + 2 * xi_pub),
        'jsd_l_target' : 2.0 * xi_priv,
        'label_smooth' : 0.1 * xi_priv,
        'pwkt_tau'     : max(5.0, 20.0 * (1.0 - xi_test)),
        'arc_scale'    : 1.0 + xi_test,
        'arc_cap'      : 2.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def smooth_ce(logits, labels, eps=0.0):
    """Label-smoothed cross-entropy. Reduces to standard NLL when eps=0."""
    log_p = F.log_softmax(logits, dim=1)
    nll   = F.nll_loss(log_p, labels)
    if eps < 1e-6:
        return nll
    uniform_entropy = -log_p.mean(dim=1).mean()
    return (1.0 - eps) * nll + eps * uniform_entropy


def ada_jsd_weight(ce_ema, l_target, base=12.0):
    """
    Adaptive JSD weight lambda_jsd^(t).
    Returns fixed base constant when l_target < 0.01 (clean data, xi=0).
    Output is clipped to [0.5*base, 1.5*base] = [6.0, 18.0].
    """
    if l_target < 0.01:
        return base
    ratio = 2.0 * (l_target - ce_ema) / max(l_target, 0.1)
    ratio = float(np.clip(ratio, -2.0, 2.0))
    scale = 0.5 + 0.5 * float(np.tanh(ratio))
    return float(np.clip(base * scale, base * 0.5, base * 1.5))


def compute_jsd(p_c, p_1, p_2):
    """Three-view Jensen-Shannon Divergence."""
    p_m = torch.clamp((p_c + p_1 + p_2) / 3.0, 1e-7, 1.0).log()
    return (
        F.kl_div(p_m, p_c,  reduction='batchmean') +
        F.kl_div(p_m, p_1,  reduction='batchmean') +
        F.kl_div(p_m, p_2,  reduction='batchmean')
    ) / 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# MCE TEACHER
# ═══════════════════════════════════════════════════════════════════════════════

def mce_teacher_target(network, images, n_crops, temp):
    """
    Multi-Crop Ensemble teacher soft labels.

    images must be a plain (B, 3, H, W) tensor — the public dataloader
    does NOT use AugMix so no unpacking is needed here.

    n_crops in {1, 2, 3}; temp = tau_teach.
    Returns detached probability tensor of shape (B, C).
    """
    network.eval()
    with torch.no_grad():
        # Ensure images are plain 3-channel regardless of any wrapping
        if isinstance(images, (list, tuple)):
            images = images[0]
        if images.shape[1] != 3:
            images = images[:, :3, :, :]   # fallback: take first 3 channels

        views = [images]

        if n_crops >= 2:
            # View 2: deterministic horizontal flip
            views.append(torch.flip(images, dims=[-1]))

        if n_crops >= 3:
            # View 3: stochastic affine crop + bilinear resize
            B, C, H, W = images.shape
            s    = 0.8 + 0.2 * torch.rand(1).item()
            h_c  = max(1, int(H * s))
            w_c  = max(1, int(W * s))
            top  = torch.randint(0, max(1, H - h_c + 1), (1,)).item()
            left = torch.randint(0, max(1, W - w_c + 1), (1,)).item()
            crop = images[:, :, top:top + h_c, left:left + w_c]
            crop = F.interpolate(crop, size=(H, W),
                                 mode='bilinear', align_corners=False)
            views.append(crop)

        prob_sum = None
        for view in views:
            logits, _ = network(view)
            p         = F.softmax(logits / max(temp, 1e-6), dim=1)
            prob_sum  = p if prob_sum is None else prob_sum + p

        return (prob_sum / len(views)).detach()


# ═══════════════════════════════════════════════════════════════════════════════
# PWKT WEIGHT STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

def weights_uniform(student_idx, K):
    """Uniform weighting — symmetric HFL."""
    w = [0.0 if j == student_idx else 1.0 / (K - 1) for j in range(K)]
    return w


def weights_binary_current(student_acc, all_acc, student_idx):
    """Binary Heaviside gate on instantaneous accuracy — AsymHFL."""
    raw   = [1.0 if (j != student_idx and all_acc[j] > student_acc)
             else 0.0 for j in range(len(all_acc))]
    total = sum(raw)
    if total < 1e-8:
        return [0.0] * len(all_acc)
    return [w / total for w in raw]


def weights_softmax_current(student_acc, all_acc, student_idx, tau):
    """Softmax over instantaneous accuracy — no Heaviside gate."""
    raw = []
    for j, acc_j in enumerate(all_acc):
        if j == student_idx:
            raw.append(0.0)
        else:
            raw.append(float(np.exp((acc_j - student_acc) / max(tau, 1e-6))))
    total = sum(raw)
    if total < 1e-8:
        return [0.0] * len(all_acc)
    return [w / total for w in raw]


def weights_softmax_peak(student_peak, all_peak, student_idx, tau):
    """Softmax over historical peak accuracy — no Heaviside gate."""
    raw = []
    for j, acc_j in enumerate(all_peak):
        if j == student_idx:
            raw.append(0.0)
        else:
            raw.append(float(np.exp((acc_j - student_peak) / max(tau, 1e-6))))
    total = sum(raw)
    if total < 1e-8:
        return [0.0] * len(all_peak)
    return [w / total for w in raw]


def weights_pwkt(student_peak, all_peak, student_idx, tau):
    """Full PWKT: Gibbs-Boltzmann over peak accuracy + Heaviside truncation."""
    raw = []
    for j, acc_j in enumerate(all_peak):
        if j == student_idx or acc_j <= student_peak:
            raw.append(0.0)   # Heaviside: exclude self and inferior peers
        else:
            raw.append(float(np.exp((acc_j - student_peak) / max(tau, 1e-6))))
    total = sum(raw)
    if total < 1e-8:
        return raw   # All zeros: no superior peers, skip distillation
    return [w / total for w in raw]


def arc_lambda(acc_now, acc_prev, scale, cap=2.0):
    """
    Adaptive Rescue Coefficient.
    acc_now and acc_prev are in PERCENTAGE scale [0, 100].
    Returns 0.0 when stable (non-negative delta).
    """
    delta = acc_now - acc_prev
    if delta >= 0:
        return 0.0
    return float(np.clip(abs(delta) * scale, 0.0, cap))


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_network(network, dataloader):
    """
    Full-dataset evaluation.
    correct/total maintained outside batch loop to avoid last-batch bias.
    Returns percentage accuracy in [0, 100].
    """
    network.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            # Public/test loaders return plain tensors — no unpacking needed
            if isinstance(images, (list, tuple)):
                images = images[0]
            if images.shape[1] != 3:
                images = images[:, :3, :, :]
            images = images.to(device)
            labels = labels.to(device)
            out, _ = network(images)
            _, pred = torch.max(out.data, 1)
            total   += labels.size(0)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / total


def evaluate_all_clients(net_list, xi_test):
    """Evaluate all K clients on the test set at severity xi_test."""
    accs = []
    for p in range(N_PARTICIPANTS):
        _, test_dl, _, _ = get_dataloader(
            dataset='cifar10',
            datadir=str(PRIVATE_DIR),
            train_bs=TRAIN_BATCH_SIZE,
            test_bs=TEST_BATCH_SIZE,
            corrupt_rate=ABLATION_XI_PUB,
            test_corrupt_rate=xi_test,
        )
        network = nn.DataParallel(net_list[p], device_ids=DEVICE_IDS).to(device)
        accs.append(evaluate_network(network, test_dl))
    return accs


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_pretrained_nets(xi_priv):
    """
    Load pre-trained client checkpoints for a given xi_priv.
    Checkpoints must exist at:
        ROOT_DIR/Network/Model_Storage_hetero_4client/
            cifar10_iid_0/random_corrupt_{xi_priv}/{arch}_{idx}.ckpt
    """
    load_dir = (
        ROOT_DIR / "Network"
        / "Model_Storage_hetero_4client"
        / "cifar10_iid_0"
        / f"random_corrupt_{xi_priv}"
    )
    net_list = init_nets(
        n_parties=N_PARTICIPANTS,
        nets_name_list=NETS_NAME_LIST,
        num_classes=N_CLASSES,
    )
    for i in range(N_PARTICIPANTS):
        ckpt_path = load_dir / f"{NETS_NAME_LIST[i]}_{i}.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Please run pre-training before ablation."
            )
        state = torch.load(str(ckpt_path), map_location=device)
        # Handle DataParallel-wrapped checkpoints
        if all(k.startswith('module.') for k in state.keys()):
            state = {k[len('module.'):]: v for k, v in state.items()}
        net_list[i].load_state_dict(state)
        print(f"  Loaded: {NETS_NAME_LIST[i]} (client {i})")
    return net_list


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING — FIXED WITH unpack_augmix_batch
# ═══════════════════════════════════════════════════════════════════════════════

def local_train_epoch(network, dataloader, optimizer,
                      ce_ema, hp, cl_module='dcl',
                      fixed_lambda=None):
    """
    One epoch of local training with safe multi-view unpacking.

    cl_module options:
        'none'   — no contrastive learning (CE + JSD only)
        'supcon' — standard 2-view SupCon on strong augmented pairs
        'dcl'    — 3-view DCL (APEX-FL default)

    fixed_lambda: if not None, use this value instead of ada_jsd_weight().

    Returns: (updated_ce_ema, mean_loss_float)
    """
    network.train()
    CE_EMA_MOM = 0.95
    losses     = []

    for batch in dataloader:
        images, labels = batch
        labels = labels.to(device).long()

        # ── SAFE UNPACK: handles both (B,9,H,W) tensor and list format ───────
        img0, img1, img2, img3 = unpack_augmix_batch(images)
        img0 = img0.to(device)
        img1 = img1.to(device)
        img2 = img2.to(device)
        img3 = img3.to(device)
        bs   = img0.size(0)

        # ── JSD: three views concatenated along BATCH dimension ──────────────
        # Each img has shape (B, 3, H, W) — correct for conv layers
        imgs_jsd       = torch.cat([img0, img1, img2], dim=0)  # (3B, 3, H, W)
        logits_all, _  = network(imgs_jsd)
        lc, la1, la2   = torch.split(logits_all, bs)

        # CE loss on original view only
        ce_loss = smooth_ce(lc, labels, eps=hp['label_smooth'])

        # Update EMA
        ce_ema = CE_EMA_MOM * ce_ema + (1 - CE_EMA_MOM) * ce_loss.item()

        # Adaptive or fixed JSD coefficient
        if fixed_lambda is not None:
            lam = fixed_lambda
        else:
            lam = ada_jsd_weight(ce_ema, hp['jsd_l_target'])

        # JSD consistency loss
        p_c  = F.softmax(lc,  dim=1)
        p_1  = F.softmax(la1, dim=1)
        p_2  = F.softmax(la2, dim=1)
        jsd  = compute_jsd(p_c, p_1, p_2)

        loss = ce_loss + lam * jsd

        # ── Contrastive branch ───────────────────────────────────────────────
        if cl_module == 'supcon':
            # Standard 2-view SupCon: original + strong augmented
            imgs_sc = torch.cat([img0, img1], dim=0)    # (2B, 3, H, W)
            feats   = network.module.backbone(imgs_sc)
            feats   = F.normalize(feats, dim=1)
            fc, f1  = torch.split(feats, bs)
            f_cat   = torch.cat([fc.unsqueeze(1), f1.unsqueeze(1)], dim=1)
            loss   += SupConLoss(temperature=0.2, device=device)(f_cat, labels)

        elif cl_module == 'dcl':
            # 3-view DCL: original + strong (img1) + weak (img3)
            imgs_dcl = torch.cat([img0, img1, img3], dim=0)  # (3B, 3, H, W)
            feats    = network.module.backbone(imgs_dcl)
            feats    = F.normalize(feats, dim=1)
            f_o, f_s, f_w = torch.split(feats, bs)
            loss += DCLLoss(
                temperature     = 0.2,
                device          = device,
                beta            = 1.0,
                ddm_temperature = 0.2,
            )(
                original_feature = f_o.unsqueeze(1),
                weak_feature     = f_w.unsqueeze(1),
                strong_feature   = f_s.unsqueeze(1),
                labels           = labels,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return ce_ema, float(np.mean(losses)) if losses else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# COLLABORATIVE DISTILLATION — FIXED
# ═══════════════════════════════════════════════════════════════════════════════

def collab_step(net_list, public_dl, collab_opts,
                acc_current, acc_prev, acc_history,
                hp, config):
    """
    One full pass over the public dataset for collaborative distillation.

    config keys:
        weight_strategy : 'uniform' | 'binary_current' | 'softmax_current'
                        | 'softmax_peak' | 'pwkt'
        use_mce         : bool   — whether to use MCE teacher
        use_arc         : bool   — whether to use ARC rescue
        mce_n_crops     : int    — override MCE view count
        mce_temp        : float  — override MCE teacher temperature
        pwkt_tau        : float  — override PWKT temperature
    """
    kl_fn = nn.KLDivLoss(reduction='batchmean').to(device)

    n_crops = config.get('mce_n_crops', hp['mce_n_crops'])
    temp    = config.get('mce_temp',    hp['teacher_temp'])
    tau     = config.get('pwkt_tau',    hp['pwkt_tau'])

    # ARC lambdas for all clients
    rescue_lams = []
    for p in range(N_PARTICIPANTS):
        lam = arc_lambda(acc_current[p], acc_prev[p],
                         hp['arc_scale'], hp['arc_cap'])
        rescue_lams.append(lam if config.get('use_arc', True) else 0.0)
    best_client = int(np.argmax(acc_current))

    for images, _ in public_dl:
        # Public dataloader returns plain (B, 3, H, W) tensors
        if isinstance(images, (list, tuple)):
            images = images[0]
        if images.shape[1] != 3:
            images = images[:, :3, :, :]
        images = images.to(device)

        # ── Compute teacher targets and student log-probs ─────────────────
        target_list  = []   # MCE soft labels, detached
        logsoft_list = []   # Student log-softmax, requires grad

        for p in range(N_PARTICIPANTS):
            network = nn.DataParallel(
                net_list[p], device_ids=DEVICE_IDS).to(device)

            # Teacher target
            if config.get('use_mce', True):
                target = mce_teacher_target(network, images, n_crops, temp)
            else:
                network.eval()
                with torch.no_grad():
                    logits, _ = network(images)
                    target    = F.softmax(logits, dim=1).detach()
            target_list.append(target)

            # Student log-prob (needs grad)
            network.train()
            logits, _ = network(images)
            logsoft_list.append(F.log_softmax(logits, dim=1))

        # ── Per-student PWKT + ARC update ─────────────────────────────────
        for p in range(N_PARTICIPANTS):
            strategy = config.get('weight_strategy', 'pwkt')

            if strategy == 'uniform':
                weights = weights_uniform(p, N_PARTICIPANTS)
            elif strategy == 'binary_current':
                weights = weights_binary_current(
                    acc_current[p], acc_current, p)
            elif strategy == 'softmax_current':
                weights = weights_softmax_current(
                    acc_current[p], acc_current, p, tau)
            elif strategy == 'softmax_peak':
                weights = weights_softmax_peak(
                    acc_history[p], acc_history, p, tau)
            else:  # 'pwkt'
                weights = weights_pwkt(
                    acc_history[p], acc_history, p, tau)

            loss    = torch.tensor(0.0, device=device, requires_grad=False)
            n_teach = 0.0

            for j in range(N_PARTICIPANTS):
                if j == p or weights[j] < 1e-8:
                    continue
                kl_val  = kl_fn(logsoft_list[p], target_list[j].detach())
                loss    = loss + weights[j] * kl_val
                n_teach += weights[j]

            # ARC emergency rescue from globally best client
            if rescue_lams[p] > 0 and best_client != p:
                rescue_kl = kl_fn(logsoft_list[p],
                                   target_list[best_client].detach())
                loss      = loss + rescue_lams[p] * rescue_kl
                n_teach   = max(n_teach, 1.0)

            if n_teach > 1e-8:
                collab_opts[p].zero_grad()
                (loss / n_teach).backward()
                collab_opts[p].step()


# ═══════════════════════════════════════════════════════════════════════════════
# FULL EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_experiment(config,
                   xi_priv=ABLATION_XI_PRIV,
                   xi_pub=ABLATION_XI_PUB,
                   xi_test=ABLATION_XI_TEST,
                   n_epochs=COMMUNICATION_EPOCH):
    """
    Run one complete federated training experiment with the given config dict.

    Supported config keys:
        do_collab        (bool)   — whether to run collaborative distillation
        cl_module        (str)    — 'none' | 'supcon' | 'dcl'
        weight_strategy  (str)    — 'uniform' | 'binary_current' |
                                    'softmax_current' | 'softmax_peak' | 'pwkt'
        use_mce          (bool)   — use MCE teacher
        use_arc          (bool)   — use ARC rescue
        mce_n_crops      (int)    — override view cardinality
        mce_temp         (float)  — override teacher temperature
        pwkt_tau         (float)  — override PWKT temperature
        jsd_lambda_fixed (float)  — use fixed JSD coefficient instead of adaptive

    Returns:
        clean_accs   : list of K clean test accuracies (%)
        corrupt_accs : list of K corrupted test accuracies (%)
    """
    set_seed(SEED)
    hp = compute_hyperparams(xi_priv, xi_pub, xi_test)

    # Build private data index map (iid partition)
    net_dataidx_map = {
        i: np.random.permutation(50000)[:PRIVATE_DATA_LEN]
        for i in range(N_PARTICIPANTS)
    }

    # Public data loader (no AugMix — plain CIFAR-100 samples)
    pub_idx = generate_public_data_indexs(
        'cifar100', str(PUBLIC_DIR_ORG), PUBLIC_DATA_LEN)
    public_dl, _, public_ds, _ = get_dataloader(
        dataset       = 'cifar100',
        datadir       = str(PUBLIC_DIR),
        train_bs      = TRAIN_BATCH_SIZE,
        test_bs       = TEST_BATCH_SIZE,
        dataidxs      = pub_idx,
        corrupt_rate  = xi_pub,
        test_corrupt_rate = xi_test,
        augmix_module = None,    # public loader MUST NOT use AugMix
    )

    # Load pre-trained networks
    net_list    = load_pretrained_nets(xi_priv)
    cl_module   = config.get('cl_module',        'dcl')
    do_collab   = config.get('do_collab',         True)
    fixed_lam   = config.get('jsd_lambda_fixed',  None)
    augmix_mod  = 'jsd' if cl_module in ('supcon', 'dcl') else None

    # Persistent collaborative optimizers (one per client)
    collab_opts = [
        optim.Adam(net_list[p].parameters(), lr=LEARNING_RATE)
        for p in range(N_PARTICIPANTS)
    ]

    acc_history = [0.0] * N_PARTICIPANTS
    acc_prev    = [0.0] * N_PARTICIPANTS

    for epoch in range(n_epochs):
        print(f"    Epoch {epoch+1}/{n_epochs}", end='\r', flush=True)

        # ── Phase 1: Evaluate ────────────────────────────────────────────────
        acc_current = evaluate_all_clients(net_list, xi_test)
        for p in range(N_PARTICIPANTS):
            acc_history[p] = max(acc_history[p], acc_current[p])

        # ── Phase 2: Collaborative distillation ─────────────────────────────
        if do_collab:
            collab_step(
                net_list, public_dl, collab_opts,
                acc_current, acc_prev, acc_history,
                hp, config,
            )

        acc_prev = acc_current[:]

        # ── Phase 3: Local training ──────────────────────────────────────────
        for p in range(N_PARTICIPANTS):
            network = nn.DataParallel(
                net_list[p], device_ids=DEVICE_IDS).to(device)
            network.train()

            train_dl, _, train_ds, _ = get_dataloader(
                dataset       = 'cifar10',
                datadir       = str(PRIVATE_DIR),
                train_bs      = TRAIN_BATCH_SIZE,
                test_bs       = TEST_BATCH_SIZE,
                dataidxs      = net_dataidx_map[p],
                corrupt_rate  = xi_priv,
                test_corrupt_rate=xi_test,
                augmix_module = augmix_mod,
            )
            local_epochs = max(int(len(public_ds) / max(len(train_ds), 1)), 1)
            local_opt    = optim.Adam(network.parameters(), lr=LEARNING_RATE)
            ce_ema_init  = hp['jsd_l_target'] if hp['jsd_l_target'] > 0.01 else 1.0

            ce_ema = ce_ema_init
            for _ in range(local_epochs):
                ce_ema, _ = local_train_epoch(
                    network    = network,
                    dataloader = train_dl,
                    optimizer  = local_opt,
                    ce_ema     = ce_ema,
                    hp         = hp,
                    cl_module  = cl_module,
                    fixed_lambda = fixed_lam,
                )
            # Write updated weights back to net_list
            net_list[p] = network.module

    print()  # newline after epoch counter

    # ── Final evaluation on both clean and corrupted test sets ──────────────
    print("    Final evaluation...")
    clean_accs   = evaluate_all_clients(net_list, xi_test=0)
    corrupt_accs = evaluate_all_clients(net_list, xi_test=1)

    return clean_accs, corrupt_accs


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE 1: COMPONENT-LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

def run_component_ablation():
    """
    Table tab:ablation_component
    Progressive component ablation at xi_priv=0.5.
    """
    print("\n" + "="*70)
    print("TABLE 1: Component-Level Ablation (xi=0.5)")
    print("="*70)

    configs = [
        #("Baseline (no FL)",
        # {'do_collab': False, 'cl_module': 'none',
        #  'weight_strategy': 'uniform',
        #  'use_mce': False, 'use_arc': False}),

        #("+ HFL (symmetric)",
        # {'do_collab': True,  'cl_module': 'none',
        #  'weight_strategy': 'uniform',
        #  'use_mce': False, 'use_arc': False}),

        #("+ AsymHFL (RAHFL base)",
        # {'do_collab': True,  'cl_module': 'none',
        #  'weight_strategy': 'binary_current',
        #  'use_mce': False, 'use_arc': False}),

        #("+ MCE",
        # {'do_collab': True,  'cl_module': 'none',
        #  'weight_strategy': 'binary_current',
        #  'use_mce': True,  'use_arc': False}),

        #("+ ADA-JSD",
        # {'do_collab': True,  'cl_module': 'none',
        #  'weight_strategy': 'binary_current',
        #  'use_mce': True,  'use_arc': False}),

        #("+ DCL",
        # {'do_collab': True,  'cl_module': 'dcl',
        #  'weight_strategy': 'binary_current',
        #  'use_mce': True,  'use_arc': False}),

        #("+ PWKT (Full APEX-FL)",
        # {'do_collab': True,  'cl_module': 'dcl',
        #  'weight_strategy': 'pwkt',
        #  'use_mce': True,  'use_arc': True}),
    ]

    results = {}
    prev_clean = prev_corrupt = None

    for label, cfg in configs:
        print(f"\n  Running: {label}")
        clean, corrupt = run_experiment(cfg)
        ca = float(np.mean(clean))
        xa = float(np.mean(corrupt))
        results[label] = {
            'clean'      : [float(x) for x in clean],
            'corrupt'    : [float(x) for x in corrupt],
            'clean_avg'  : ca,
            'corrupt_avg': xa,
        }
        dc = f"{ca - prev_clean:+.2f}"   if prev_clean   is not None else "---"
        dx = f"{xa - prev_corrupt:+.2f}" if prev_corrupt is not None else "---"
        print(f"  Clean   avg: {ca:.2f}% ({dc}) | "
              f"Per-client: {[f'{v:.2f}' for v in clean]}")
        print(f"  Corrupt avg: {xa:.2f}% ({dx}) | "
              f"Per-client: {[f'{v:.2f}' for v in corrupt]}")
        prev_clean   = ca
        prev_corrupt = xa

    _print_table_header("Component Ablation", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE 2: PWKT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pwkt_ablation():
    """
    Table tab:ablation_pwkt
    PWKT Transfer Distribution Comparison at xi=0.5.
    """
    print("\n" + "="*70)
    print("TABLE 2: PWKT Transfer Distribution Comparison (xi=0.5)")
    print("="*70)

    configs = [
        ("Uniform (HFL)",
         {'weight_strategy': 'uniform',
          'use_mce': True, 'cl_module': 'dcl', 'use_arc': False}),

        ("Binary gate, current ACC (AsymHFL)",
         {'weight_strategy': 'binary_current',
          'use_mce': True, 'cl_module': 'dcl', 'use_arc': False}),

        ("Softmax, current ACC, no gate",
         {'weight_strategy': 'softmax_current',
          'use_mce': True, 'cl_module': 'dcl', 'use_arc': False}),

        ("Softmax, peak ACC, no gate",
         {'weight_strategy': 'softmax_peak',
          'use_mce': True, 'cl_module': 'dcl', 'use_arc': False}),

        #("PWKT: peak ACC + Heaviside (APEX-FL)",
        # {'weight_strategy': 'pwkt',
        #  'use_mce': True, 'cl_module': 'dcl', 'use_arc': True}),
    ]

    results = {}
    base_clean = base_corrupt = None

    for label, cfg in configs:
        print(f"\n  Running: {label}")
        clean, corrupt = run_experiment(cfg)
        ca = float(np.mean(clean))
        xa = float(np.mean(corrupt))
        results[label] = {'clean_avg': ca, 'corrupt_avg': xa}

        if base_clean is None:
            base_clean, base_corrupt = ca, xa

        jc = ca - base_clean
        jx = xa - base_corrupt
        print(f"  Clean: {ca:.2f}%  Corrupt: {xa:.2f}%  "
              f"Joint Δ: {jc+jx:+.2f}")

    _print_table_header("PWKT Distribution", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE 3: ADA-JSD STATIC vs ADAPTIVE
# ═══════════════════════════════════════════════════════════════════════════════

def run_adajsd_ablation():
    """
    Table tab:ablation_adajsd
    ADA-JSD Static vs. Adaptive at xi in {0, 0.5, 1}.
    """
    print("\n" + "="*70)
    print("TABLE 3: ADA-JSD Static vs. Adaptive (all xi)")
    print("="*70)

    lambda_configs = [
        ("Static λ=6.0",                6.0),
        ("Static λ=12.0 (RAHFL base)", 12.0),
        ("Static λ=18.0",               18.0),
        #("ADA-JSD adaptive (APEX-FL)",  None),   # None = use adaptive
    ]

    xi_list = [0, 0.5, 1]
    results = {lbl: {} for lbl, _ in lambda_configs}

    base_cfg = {
        'weight_strategy': 'pwkt',
        'use_mce'        : True,
        'cl_module'      : 'dcl',
        'use_arc'        : True,
    }

    for lbl, lam in lambda_configs:
        for xi in xi_list:
            print(f"\n  Running: {lbl} | xi={xi}")
            cfg = dict(base_cfg)
            if lam is not None:
                cfg['jsd_lambda_fixed'] = lam

            clean, corrupt = run_experiment(
                cfg, xi_priv=xi, xi_pub=xi, xi_test=xi)
            ca = float(np.mean(clean))
            xa = float(np.mean(corrupt))
            results[lbl][xi] = {'clean_avg': ca, 'corrupt_avg': xa}
            print(f"  Clean: {ca:.2f}%  Corrupt: {xa:.2f}%")

    print("\n--- ADA-JSD Summary (Corrupt Avg %) ---")
    print(f"{'Method':<35} | xi=0.0  | xi=0.5  | xi=1.0")
    print("-"*65)
    for lbl in results:
        vals = " | ".join(
            f"{results[lbl][xi]['corrupt_avg']:>7.2f}"
            for xi in xi_list
        )
        print(f"{lbl:<35} | {vals}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE 4: MCE CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_mce_ablation():
    """
    Table tab:ablation_mce
    MCE Fixed vs. Adaptive at xi_pub=0.5.
    """
    print("\n" + "="*70)
    print("TABLE 4: MCE Fixed vs. Adaptive (xi_pub=0.5)")
    print("="*70)

    configs = [
        ("V=1, tau=1.0 (no MCE, RAHFL)",
         {'use_mce': False, 'mce_n_crops': 1, 'mce_temp': 1.0}),

        ("V=2, tau=1.0 (flip only, no temp sharpening)",
         {'use_mce': True,  'mce_n_crops': 2, 'mce_temp': 1.0}),

        ("V=3, tau=1.0 (all views, no temp sharpening)",
         {'use_mce': True,  'mce_n_crops': 3, 'mce_temp': 1.0}),

        ("V=2, tau=0.65 (flip + adaptive temp)",
         {'use_mce': True,  'mce_n_crops': 2, 'mce_temp': 0.65}),

        #("Adaptive V, adaptive tau (APEX-FL)",
        # {'use_mce': True}),   # uses hp defaults from xi_pub=0.5
    ]

    results = {}
    for label, mce_cfg in configs:
        print(f"\n  Running: {label}")
        cfg = {
            'weight_strategy': 'pwkt',
            'cl_module'      : 'dcl',
            'use_arc'        : True,
            **mce_cfg,
        }
        clean, corrupt = run_experiment(cfg)
        ca = float(np.mean(clean))
        xa = float(np.mean(corrupt))
        results[label] = {'clean_avg': ca, 'corrupt_avg': xa}
        print(f"  Clean: {ca:.2f}%  Corrupt: {xa:.2f}%")

    _print_table_header("MCE Configuration", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION TABLE 5: DCL STRATEGY
# ═══════════════════════════════════════════════════════════════════════════════

def run_dcl_ablation():
    """
    Table tab:ablation_dcl
    DCL Strategy Comparison at xi=0.5.
    """
    print("\n" + "="*70)
    print("TABLE 5: DCL Strategy Comparison (xi=0.5)")
    print("="*70)

    configs = [
        ("w/o SCL (no contrastive)",       {'cl_module': 'none'}),
        ("SupCon (2-view, strong pairs)",   {'cl_module': 'supcon'}),
        #("DCL 3-view (APEX-FL)",           {'cl_module': 'dcl'}),
    ]

    results   = {}
    base_ca   = None
    base_xa   = None

    for label, cl_cfg in configs:
        print(f"\n  Running: {label}")
        cfg = {
            'weight_strategy': 'pwkt',
            'use_mce'        : True,
            'use_arc'        : True,
            **cl_cfg,
        }
        clean, corrupt = run_experiment(cfg)
        ca = float(np.mean(clean))
        xa = float(np.mean(corrupt))

        if base_ca is None:
            base_ca, base_xa = ca, xa

        results[label] = {
            'clean_avg'    : ca,
            'corrupt_avg'  : xa,
            'delta_clean'  : ca - base_ca,
            'delta_corrupt': xa - base_xa,
        }
        print(f"  Clean: {ca:.2f}%  Corrupt: {xa:.2f}%  "
              f"Δ clean: {ca-base_ca:+.2f}  Δ corrupt: {xa-base_xa:+.2f}")

    _print_table_header("DCL Strategy", results)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PRINT HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _print_table_header(name, results):
    print(f"\n--- {name} Summary ---")
    print(f"{'Method':<48} | {'Clean Avg':>10} | {'Corrupt Avg':>11}")
    print("-"*76)
    for label, r in results.items():
        ca = r.get('clean_avg', float('nan'))
        xa = r.get('corrupt_avg', float('nan'))
        print(f"{label:<48} | {ca:>10.2f}% | {xa:>11.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger = init_logs()
    logger.info("APEX-FL Ablation Study — Full Runner (FIXED)")
    logger.info(f"xi: priv={ABLATION_XI_PRIV}, "
                f"pub={ABLATION_XI_PUB}, test={ABLATION_XI_TEST}")
    logger.info(f"Device: {device}")

    print("\n" + "█"*70)
    print("  APEX-FL ABLATION STUDY — ALL TABLES")
    print("█"*70)

    # ── Diagnose dataloader format before any experiment ─────────────────────
    print("\n[DIAGNOSTIC] Checking private dataloader format ...")
    dummy_dl, _, _, _ = get_dataloader(
        dataset='cifar10', datadir=str(PRIVATE_DIR),
        train_bs=4, test_bs=4,
        dataidxs=np.arange(100),
        corrupt_rate=ABLATION_XI_PRIV,
        test_corrupt_rate=ABLATION_XI_TEST,
        augmix_module='jsd',
    )
    diagnose_dataloader_format(dummy_dl, "private train loader (augmix=jsd)")

    print("\n[DIAGNOSTIC] Checking public dataloader format ...")
    pub_idx = generate_public_data_indexs(
        'cifar100', str(PUBLIC_DIR_ORG), 100)
    dummy_pub_dl, _, _, _ = get_dataloader(
        dataset='cifar100', datadir=str(PUBLIC_DIR),
        train_bs=4, test_bs=4,
        dataidxs=pub_idx,
        corrupt_rate=ABLATION_XI_PUB,
        test_corrupt_rate=ABLATION_XI_PUB,
        augmix_module=None,
    )
    diagnose_dataloader_format(dummy_pub_dl, "public loader (no augmix)")

    all_results = {}

    #print("\n[1/5] Component-Level Ablation")
    #all_results['component'] = run_component_ablation()

    print("\n[2/5] PWKT Transfer Distribution Comparison")
    all_results['pwkt'] = run_pwkt_ablation()

    print("\n[3/5] ADA-JSD Static vs. Adaptive")
    all_results['adajsd'] = run_adajsd_ablation()

    print("\n[4/5] MCE Fixed vs. Adaptive")
    all_results['mce'] = run_mce_ablation()

    print("\n[5/5] DCL Strategy Comparison")
    all_results['dcl'] = run_dcl_ablation()

    # ── Final consolidated summary ────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  FINAL CONSOLIDATED RESULTS SUMMARY")
    print("═"*70)

    #print("\n▶ TABLE 1 — Component Ablation (Corrupt Avg %)")
    #for lbl, r in all_results['component'].items():
    #    print(f"  {lbl:<35} | {r['corrupt_avg']:.2f}%")

    print("\n▶ TABLE 2 — PWKT Distribution (Clean / Corrupt %)")
    for lbl, r in all_results['pwkt'].items():
        print(f"  {lbl:<45} | {r['clean_avg']:.2f}% / {r['corrupt_avg']:.2f}%")

    print("\n▶ TABLE 3 — ADA-JSD (Corrupt Avg % at each xi)")
    for lbl, r in all_results['adajsd'].items():
        vals = "  ".join(
            f"xi={xi}: {r[xi]['corrupt_avg']:.2f}%"
            for xi in [0.0, 0.5, 1.0]
        )
        print(f"  {lbl:<35} | {vals}")

    print("\n▶ TABLE 4 — MCE Configuration (Clean / Corrupt %)")
    for lbl, r in all_results['mce'].items():
        print(f"  {lbl:<45} | {r['clean_avg']:.2f}% / {r['corrupt_avg']:.2f}%")

    print("\n▶ TABLE 5 — DCL Strategy (Clean / Corrupt %)")
    for lbl, r in all_results['dcl'].items():
        print(f"  {lbl:<40} | "
              f"{r['clean_avg']:.2f}% / {r['corrupt_avg']:.2f}%")

    # ── Save to JSON ──────────────────────────────────────────────────────────
    def _serialize(obj):
        """Recursively convert numpy types to plain Python for JSON."""
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, dict):
            return {str(k): _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(i) for i in obj]
        return obj

    save_path = ROOT_DIR / "ablation_results.json"
    with open(str(save_path), 'w') as f:
        json.dump(_serialize(all_results), f, indent=2)

    print(f"\n✓ All results saved to: {save_path}")
    print("✓ Ablation study complete.\n")


if __name__ == '__main__':
    main()