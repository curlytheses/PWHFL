import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from pathlib import Path

sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from RAHFL.Dataset.utils import (
    init_logs, partition_data, get_dataloader,
    init_nets, generate_public_data_indexs, mkdirs,
)
from RAHFL.HHF.loss import SupConLoss, DCLLoss

ROOT_DIR = Path.cwd()
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Parameters — identical to baseline ───────────────────────────────────────
Seed               = 0
TrainBatchSize     = 256
TestBatchSize      = 512
CommunicationEpoch = 40
Pariticpant_Params = {
    'loss_funnction': 'CE',
    'optimizer_name': 'Adam',
    'learning_rate' : 0.001,
}
AugMix_Module = 'jsd'
CL_Module     = 'dcl'
Col_Module    = 'apex'  # 'apex' | 'asymhfl' | 'hfl'

# ── Corruption rates — each drives a module ───────────────────────────────────
Private_Corrupt_Rate = 0.5   # → ADA-JSD L_target; label smooth ε
Test_Corrupt_Rate    = 0   # → PWKT τ; ARC scale
Public_Corrupt_Rate  = 0.5   # → MCE n_crops; teacher temperature

Model_setting          = 'hetero'
Nets_Name_List         = ['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2']
N_Participants         = len(Nets_Name_List)
Private_Dataset_Name   = 'cifar10'
Private_Dataset_Dir    = ROOT_DIR / "Dataset" / "cifar_10_c"
Private_Data_Len       = 10000
Private_Output_Channel = 10
Data_Partition         = 'iid'
Noniid_Dirichlet_Beta  = 0

Public_Dataset_Name    = 'cifar100'
Public_Dataset_Dir     = ROOT_DIR / "Dataset" / "cifar_100_c"
Public_Dataset_Dir_Org = ROOT_DIR / "Dataset" / "cifar_100"
Public_Dataset_Length  = 5000

Model_Load_Dir = (
    ROOT_DIR / "Network"
    / f"Model_Storage_{Model_setting}_4client"
    / f"{Private_Dataset_Name}_{Data_Partition}_{Noniid_Dirichlet_Beta}"
    / f"random_corrupt_{Private_Corrupt_Rate}"
)
Model_Save_Dir = (
    ROOT_DIR / f"APEX_Storage_{Model_setting}_4client"
    / f"{Private_Dataset_Name}_{Data_Partition}_{Noniid_Dirichlet_Beta}"
    / f"random_corrupt_{Private_Corrupt_Rate}"
)
Matrix_Update_Epoch = 1

# ── Derived hyperparameters ───────────────────────────────────────────────────
TEACHER_TEMP  = max(0.3, 1.0 - Public_Corrupt_Rate * 0.7)
LABEL_SMOOTH  = Private_Corrupt_Rate * 0.1
MCE_N_CROPS   = round(1 + 2 * Public_Corrupt_Rate)   # 1,2, or 3 crops
JSD_L_TARGET  = Private_Corrupt_Rate * 2.0            # expected CE at this ξ
PWKT_TAU      = max(5.0, 20.0 * (1.0 - Test_Corrupt_Rate))
ARC_SCALE     = 1.0 + Test_Corrupt_Rate


def mce_teacher_target(network: nn.Module, images: torch.Tensor,
                       n_crops: int, temperature: float) -> torch.Tensor:

    network.eval()
    with torch.no_grad():
        views = [images]   # original (always included)

        if n_crops >= 2:
            # Horizontal flip — deterministic, maximally different from original
            views.append(torch.flip(images, dims=[-1]))

        if n_crops >= 3:
            # Random crop + resize — stochastic, captures spatial robustness
            B, C, H, W = images.shape
            # 80-100% crop scale, same aspect ratio
            scale = 0.8 + 0.2 * torch.rand(1).item()
            crop_h = max(1, int(H * scale))
            crop_w = max(1, int(W * scale))
            top    = torch.randint(0, H - crop_h + 1, (1,)).item()
            left   = torch.randint(0, W - crop_w + 1, (1,)).item()
            crop   = images[:, :, top:top+crop_h, left:left+crop_w]
            crop   = F.interpolate(crop, size=(H, W), mode='bilinear',
                                   align_corners=False)
            views.append(crop)

        # Average softmax probabilities across all views
        probs_sum = None
        for view in views:
            logits, _ = network(view)
            p = F.softmax(logits / temperature, dim=1)
            if probs_sum is None:
                probs_sum = p
            else:
                probs_sum = probs_sum + p

        return (probs_sum / len(views)).detach()


def ada_jsd_weight(ce_loss_val: float,
                   l_target: float = JSD_L_TARGET,
                   base: float = 12.0) -> float:

    if l_target < 0.01:   # clean data — no adaptation needed
        return base
    ratio = 2.0 * (l_target - ce_loss_val) / max(l_target, 0.1)
    ratio = float(np.clip(ratio, -2.0, 2.0))   # prevent extreme tanh inputs
    scale = 0.5 + 0.5 * float(np.tanh(ratio))
    return float(np.clip(base * scale, base * 0.5, base * 1.5))


def pwkt_weights(student_acc: float, all_acc: list,
                 student_idx: int, tau: float = PWKT_TAU) -> list:
    """Exponential weighting by accuracy gap. τ from Test_Corrupt_Rate."""
    raw = []
    for j, acc_j in enumerate(all_acc):
        if j == student_idx or acc_j <= student_acc:
            raw.append(0.0)
        else:
            raw.append(float(np.exp((acc_j - student_acc) / tau)))
    total = sum(raw)
    if total < 1e-8:
        return raw
    return [w / total for w in raw]


def arc_lambda(acc_now: float, acc_prev: float,
               scale: float = ARC_SCALE, cap: float = 2.0) -> float:
    """Rescue coefficient: 0 if improving, |delta|*scale if declining."""
    delta = acc_now - acc_prev
    if delta >= 0:
        return 0.0
    return float(np.clip(abs(delta) * scale, 0.0, cap))


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_network(network, dataloader, logger):
    """B1 fixed: correct/total outside loop → full dataset."""
    network.eval()
    correct = 0
    total   = 0
    bar = tqdm(dataloader, desc="Eval", ncols=80, leave=True)
    with torch.no_grad():
        for images, labels in bar:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
            bar.set_postfix(acc=f"{100.*correct/total:.2f}%")
    acc = 100.0 * correct / total
    logger.info(f"Accuracy: {acc:.2f}%")
    return acc


def smooth_ce(logits, labels, eps=LABEL_SMOOTH):
    """Label-smoothed CE. Reduces to standard CE when eps=0."""
    log_p = F.log_softmax(logits, dim=1)
    nll   = F.nll_loss(log_p, labels)
    if eps < 1e-6:
        return nll
    uniform = -log_p.mean(dim=1).mean()
    return (1.0 - eps) * nll + eps * uniform


def update_model_via_private_data(
    device, network, localtrain_epoch, private_dataloader,
    loss_function, optimizer_method, learning_rate, logger
):
    if optimizer_method == 'Adam':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=1e-4)

    loss_history = []
    ce_ema       = JSD_L_TARGET if JSD_L_TARGET > 0.01 else 1.0  # running CE EMA
    CE_EMA_MOM   = 0.95

    train_bar = tqdm(private_dataloader, desc="Local Train",
                     unit="batch", ncols=80, leave=True)

    for _ in range(localtrain_epoch):
        for batch_idx, (images, labels) in enumerate(train_bar):
            if AugMix_Module == 'jsd':
                images_all = torch.cat([images[0], images[1], images[2]], 0).to(device)
                labels     = labels.to(device).long()
                logits_all, _ = network(images_all)
                lc, la1, la2  = torch.split(logits_all, images[0].size(0))

                # B4: label-smoothed CE
                ce_loss = smooth_ce(lc, labels)

                # ce_ema tracks recent CE history for stable adaptation
                ce_ema = CE_EMA_MOM * ce_ema + (1 - CE_EMA_MOM) * ce_loss.item()
                lam    = ada_jsd_weight(ce_ema)

                p_c  = F.softmax(lc,  dim=1)
                p_1  = F.softmax(la1, dim=1)
                p_2  = F.softmax(la2, dim=1)
                pm   = torch.clamp((p_c + p_1 + p_2) / 3.0, 1e-7, 1.0).log()
                jsd  = (F.kl_div(pm, p_c, reduction='batchmean') +
                        F.kl_div(pm, p_1, reduction='batchmean') +
                        F.kl_div(pm, p_2, reduction='batchmean')) / 3.0

                loss = ce_loss + lam * jsd   # ← adaptive λ 

                if CL_Module == 'supcon':
                    imgs_c = torch.cat([images[0], images[1]], 0).to(device)
                    feats  = network.module.backbone(imgs_c)
                    feats  = F.normalize(feats, dim=1)
                    fc, f1 = torch.split(feats, images[0].size(0))
                    feats  = torch.cat([fc.unsqueeze(1), f1.unsqueeze(1)], dim=1)
                    loss  += SupConLoss(temperature=0.2, device=device)(feats, labels)

                elif CL_Module == 'dcl':
                    # Full 4-view DCL — identical to RAHFL baseline
                    imgs_c = torch.cat([images[0], images[1], images[3]], 0).to(device)
                    feats  = network.module.backbone(imgs_c)
                    feats  = F.normalize(feats, dim=1)
                    fc1, f1, fc2 = torch.split(feats, images[0].size(0))
                    loss  += DCLLoss(temperature=0.2, device=device,
                                     beta=1.0, ddm_temperature=0.2)(
                        original_feature=fc1.unsqueeze(1),
                        weak_feature    =fc2.unsqueeze(1),
                        strong_feature  =f1.unsqueeze(1),
                        labels=labels
                    )
            else:
                images  = images.to(device)
                labels  = labels.to(device).long()
                outputs, _ = network(images)
                loss = smooth_ce(outputs, labels)
                ce_ema = CE_EMA_MOM * ce_ema + (1 - CE_EMA_MOM) * loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}",
                                  lam=f"{ada_jsd_weight(ce_ema):.1f}")
            logger.info(
                f"Train [{batch_idx*len(images[0] if AugMix_Module=='jsd' else images)}"
                f"/{len(private_dataloader.dataset)}"
                f" ({100.*batch_idx/len(private_dataloader):.0f}%)]"
                f" Loss:{loss.item():.4f} λ_jsd:{ada_jsd_weight(ce_ema):.2f}"
            )

    return network, loss_history


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger = init_logs()
    logger.info("APEX-FL — Adaptive Peer-Ensemble eXchange Federated Learning")
    logger.info(
        f"ξ_priv={Private_Corrupt_Rate} | ξ_test={Test_Corrupt_Rate} | "
        f"ξ_pub={Public_Corrupt_Rate}"
    )
    logger.info(
        f"MCE_N_CROPS={MCE_N_CROPS} | TEACHER_TEMP={TEACHER_TEMP:.2f} | "
        f"LABEL_SMOOTH={LABEL_SMOOTH:.3f} | PWKT_TAU={PWKT_TAU:.1f} | "
        f"ARC_SCALE={ARC_SCALE:.1f} | JSD_L_TARGET={JSD_L_TARGET:.2f}"
    )

    random.seed(Seed); np.random.seed(Seed)
    torch.manual_seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)

    device_ids = [0]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False

    # ── Data ──────────────────────────────────────────────────────────────────
    if Data_Partition == 'noniid':
        _, net_dataidx_map = partition_data(
            dataset=Private_Dataset_Name, datadir=str(Private_Dataset_Dir),
            partition=Data_Partition, num_classes=Private_Output_Channel,
            num_users=N_Participants, corrupt_rate=Private_Corrupt_Rate,
            test_corrupt_rate=Test_Corrupt_Rate, dirichlet_beta=Noniid_Dirichlet_Beta
        )
    else:
        net_dataidx_map = {
            i: np.random.permutation(50000)[:Private_Data_Len]
            for i in range(N_Participants)
        }

    # ── Networks ──────────────────────────────────────────────────────────────
    net_list = init_nets(
        n_parties=N_Participants,
        nets_name_list=Nets_Name_List,
        num_classes=Private_Output_Channel
    )
    for i in range(N_Participants):
        network   = net_list[i]
        network   = nn.DataParallel(network, device_ids=device_ids).to(device)
        ckpt_path = Model_Load_Dir / f"{Nets_Name_List[i]}_{i}.ckpt"
        network.load_state_dict(torch.load(str(ckpt_path)))
        logger.info(f"Client {i} ({Nets_Name_Lists[i] if False else Nets_Name_List[i]}): loaded")

    # ── Public data ───────────────────────────────────────────────────────────
    pub_idx = generate_public_data_indexs(
        Public_Dataset_Name, str(Public_Dataset_Dir_Org), Public_Dataset_Length
    )
    public_train_dl, _, public_train_ds, _ = get_dataloader(
        dataset=Public_Dataset_Name, datadir=str(Public_Dataset_Dir),
        train_bs=TrainBatchSize, test_bs=TestBatchSize,
        dataidxs=pub_idx, corrupt_rate=Public_Corrupt_Rate, test_corrupt_rate=Test_Corrupt_Rate, augmix_module=None
    )

    # B2: persistent collab optimizers
    collab_opts = [
        optim.Adam(net_list[i].parameters(), lr=Pariticpant_Params['learning_rate'])
        for i in range(N_Participants)
    ]

    # Tracking
    acc_history = [0.0] * N_Participants   # B5: historical max
    acc_prev    = [0.0] * N_Participants
    acc_current = [0.0] * N_Participants

    acc_list, col_loss_list, local_loss_list = [], [], []

    # ══════════════════════════════════════════════════════════════════════════
    # COMMUNICATION ROUNDS
    # ══════════════════════════════════════════════════════════════════════════
    for epoch_index in range(CommunicationEpoch):
        logger.info(f"===== Comm Epoch {epoch_index} =====")

        # ── 1. Evaluate (B1 fixed) ────────────────────────────────────────────
        acc_epoch_list         = []
        matrix_update_acc_list = []

        for p in range(N_Participants):
            _, test_dl, _, _ = get_dataloader(
                dataset=Private_Dataset_Name, datadir=str(Private_Dataset_Dir),
                train_bs=TrainBatchSize, test_bs=TestBatchSize,
                corrupt_rate=Private_Corrupt_Rate,
                test_corrupt_rate=Test_Corrupt_Rate
            )
            network  = net_list[p]
            network  = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network, test_dl, logger)
            acc_epoch_list.append(accuracy)

            if epoch_index % Matrix_Update_Epoch == 0:
                acc_history[p]  = max(acc_history[p], accuracy)
                acc_current[p]  = accuracy
                matrix_update_acc_list.append(acc_history[p])

        acc_list.append(acc_epoch_list)
        avg = sum(acc_epoch_list) / N_Participants
        logger.info(f"Avg: {avg:.2f}% | Clients: {[f'{a:.1f}' for a in acc_epoch_list]}")

        # ARC: detect struggling clients
        rescue_lambdas = []
        for p in range(N_Participants):
            lam = arc_lambda(acc_current[p], acc_prev[p])
            rescue_lambdas.append(lam)
            if lam > 0:
                logger.info(f"  ARC: Client {p} declining "
                            f"Δ={acc_current[p]-acc_prev[p]:.2f}% → λ={lam:.3f}")
        best_client = int(np.argmax(acc_current))
        logger.info(f"  Best client for rescue: {best_client} "
                    f"({Nets_Name_List[best_client]} @ {acc_current[best_client]:.1f}%)")
        acc_prev = acc_current[:]

        # ── 2. Collaborative phase — MCE + PWKT + ARC ────────────────────────
        kl_loss_batch_list = []

        for batch_idx, (images, _) in enumerate(public_train_dl):
            images = images.to(device)

            # Each teacher's target is averaged over MCE_N_CROPS augmented views
            target_list  = []   # MCE-ensembled, temperature-sharpened, detached
            logsoft_list = []   # student log-probs for KL loss (needs grad)

            for p in range(N_Participants):
                network = net_list[p]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)

                # Teacher target: MCE ensemble (no grad, multiple crops)
                target = mce_teacher_target(
                    network, images, MCE_N_CROPS, TEACHER_TEMP
                )
                target_list.append(target)   # [B, C], detached

                # Student log-probs (need grad for backward)
                network.train()
                logits, _ = network(images)
                logsoft_list.append(F.log_softmax(logits, dim=1))

            # Update each client
            for p in range(N_Participants):
                network = net_list[p]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()

                kl  = nn.KLDivLoss(reduction='batchmean').to(device)
                opt = collab_opts[p]   # B2: persistent Adam
                opt.zero_grad()

                loss    = torch.tensor(0.0, device=device)
                n_teach = 0.0

                if Col_Module in ('hfl', 'asymhfl', 'apex'):

                    if Col_Module == 'hfl':
                        weights = [
                            (1.0/(N_Participants-1)) if j != p else 0.0
                            for j in range(N_Participants)
                        ]
                    elif Col_Module == 'asymhfl':
                        # Standard binary gate (for ablation comparison)
                        raw = [
                            1.0 if (j != p and matrix_update_acc_list[p]
                                    <= matrix_update_acc_list[j]) else 0.0
                            for j in range(N_Participants)
                        ]
                        s = sum(raw)
                        weights = [w/s if s > 0 else 0.0 for w in raw]
                    else:
                        # PWKT: soft exponential weights
                        weights = pwkt_weights(
                            matrix_update_acc_list[p],
                            matrix_update_acc_list, p
                        )

                    # Main KL loss with PWKT weights
                    for j in range(N_Participants):
                        if j == p or weights[j] < 1e-8:
                            continue
                        # MCE-ensembled teacher target (Innovation 1)
                        loss   += weights[j] * kl(logsoft_list[p], target_list[j])
                        n_teach += weights[j]

                    # ARC: emergency rescue from best client for struggling students
                    if (Col_Module == 'apex'
                            and rescue_lambdas[p] > 0
                            and best_client != p):
                        # Use MCE-ensembled best client target (more reliable rescue)
                        rescue = rescue_lambdas[p] * kl(
                            logsoft_list[p], target_list[best_client]
                        )
                        loss   += rescue
                        n_teach = max(n_teach, 1.0)
                        if batch_idx == 0:   # log only first batch per epoch
                            logger.info(
                                f"  ARC rescue: client {p} ← client {best_client}, "
                                f"λ={rescue_lambdas[p]:.3f}, "
                                f"rescue_loss={rescue.item():.4f}"
                            )

                    if n_teach > 1e-8:
                        loss = loss / n_teach
                        kl_loss_batch_list.append(loss.item())
                        loss.backward()
                        opt.step()

        col_loss_list.append(kl_loss_batch_list)
        if kl_loss_batch_list:
            logger.info(
                f"Collab avg KL: {sum(kl_loss_batch_list)/len(kl_loss_batch_list):.4f}"
            )

        # ── 3. Local update — DCL ─────────────────
        local_loss_batch_list = []
        for p in range(N_Participants):
            network = net_list[p]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()

            train_dl, _, train_ds, _ = get_dataloader(
                dataset=Private_Dataset_Name, datadir=str(Private_Dataset_Dir),
                train_bs=TrainBatchSize, test_bs=TestBatchSize,
                dataidxs=net_dataidx_map[p],
                corrupt_rate=Private_Corrupt_Rate, test_corrupt_rate=Test_Corrupt_Rate, augmix_module=AugMix_Module
            )
            localtrain_epoch = max(int(len(public_train_ds) / len(train_ds)), 1)

            network, loss_hist = update_model_via_private_data(
                device=device, network=network,
                localtrain_epoch=localtrain_epoch,
                private_dataloader=train_dl,
                loss_function=Pariticpant_Params['loss_funnction'],
                optimizer_method=Pariticpant_Params['optimizer_name'],
                learning_rate=Pariticpant_Params['learning_rate'],
                logger=logger
            )
            local_loss_batch_list.append(np.mean(loss_hist))

        local_loss_list.append(local_loss_batch_list)

        # ── 4. Final save ─────────────────────────────────────────────────────
        if epoch_index == CommunicationEpoch - 1:
            logger.info("Final Evaluation")
            final_accs = []
            for p in range(N_Participants):
                _, test_dl, _, _ = get_dataloader(
                    dataset=Private_Dataset_Name, datadir=str(Private_Dataset_Dir),
                    train_bs=TrainBatchSize, test_bs=TestBatchSize,
                    corrupt_rate=Test_Corrupt_Rate,
                    test_corrupt_rate=Test_Corrupt_Rate
                )
                network  = net_list[p]
                network  = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network, test_dl, logger)
                final_accs.append(accuracy)

            avg_final = sum(final_accs) / N_Participants
            logger.info(f"APEX-FL Final avg: {avg_final:.2f}%")
            logger.info(f"Per client: {[f'{a:.2f}' for a in final_accs]}")

            mkdirs(str(Model_Save_Dir))
            for p in range(N_Participants):
                network   = net_list[p]
                network   = nn.DataParallel(network, device_ids=device_ids).to(device)
                ckpt_path = Model_Save_Dir / f"{Nets_Name_List[p]}_{p}.ckpt"
                torch.save(network.state_dict(), str(ckpt_path))
                logger.info(f"Saved: {ckpt_path}")

            logger.info("PWHFL training complete.")