import os
import sys
import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from tqdm import tqdm
from pathlib import Path

# Ensure we can import from two levels up (adjust if needed)
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from RAHFL.Dataset.utils import (
    init_logs,
    partition_data,
    get_dataloader,
    init_nets,
    generate_public_data_indexs,
    mkdirs,
)
from RAHFL.HHF.loss import SupConLoss, DCLLoss

# ──────────────────────────────────────────────────────────────────────────────
# Since we've already done:
#   from google.colab import drive
#   drive.mount('/content/drive')
#   os.chdir('/content/drive/MyDrive/Colab Notebooks/RAHFL')
#
# our working directory is:
#   /content/drive/MyDrive/Colab Notebooks/RAHFL
#
# So any subfolder under this can be referenced with forward slashes (Linux style).
ROOT_DIR = Path.cwd()  # /content/drive/MyDrive/Colab Notebooks/RAHFL
# ──────────────────────────────────────────────────────────────────────────────

# GPU setup
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ──────────────────────────────────────────────────────────────────────────────
# Global Parameters
Seed               = 0
TrainBatchSize     = 256
TestBatchSize      = 512
CommunicationEpoch = 40

Pariticpant_Params = {
    'loss_funnction': 'CE',
    'optimizer_name': 'Adam',
    'learning_rate' : 0.001
}
AugMix_Module      = 'jsd'      # ['jsd', 'nojsd', None]
CL_Module          = 'dcl'      # ['supcon', 'dcl', None]
Col_Module         = 'asymhfl'  # ['hfl', 'asymhfl']

# Corruption Setting
Private_Corrupt_Rate = 0   # [0, 1, 0.5]
Test_Corrupt_Rate    = 0   # [0, 1, 0.5]
Public_Corrupt_Rate  = 0   # [0, 1, 0.5]

# Model Setting
Model_setting   = 'hetero'  # ['hetero', 'homo']
Nets_Name_List  = ['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2']
N_Participants  = len(Nets_Name_List)

# Private Dataset Setting
Private_Dataset_Name = 'cifar10'
# "cifar10" → folder "cifar_10_c" under ROOT_DIR/Dataset
dataset_folder = f"cifar_10_c"                # "cifar_10_c"
Private_Dataset_Dir = ROOT_DIR / "Dataset" / dataset_folder

Private_Data_Len      = 10000
Private_Output_Channel = 10
Data_Partition         = 'iid'   # ['iid', 'noniid']
Noniid_Dirichlet_Beta  = 0       # 0 for iid

# Public Dataset Setting
Public_Dataset_Name     = 'cifar100'     # "100"
public_folder_c       = f"cifar_100_c"      # "cifar_100_c"
public_folder_org     = f"cifar_100"        # "cifar_100"

Public_Dataset_Dir     = ROOT_DIR / "Dataset" / public_folder_c
Public_Dataset_Dir_Org = ROOT_DIR / "Dataset" / public_folder_org
Public_Dataset_Length  = 5000

# Model Load / Save Setting
load_folder = (
    ROOT_DIR
    / "Network"
    / f"Model_Storage_{Model_setting}_4client"
    / f"{Private_Dataset_Name}_{Data_Partition}_{Noniid_Dirichlet_Beta}"
    / f"random_corrupt_{Private_Corrupt_Rate}"
)
Model_Load_Dir = load_folder

save_folder_name = "Final_Model_Storage_" + f"{Model_setting}_4client"
Model_Save_Dir = ROOT_DIR / save_folder_name / f"{Private_Dataset_Name}_{Data_Partition}_{Noniid_Dirichlet_Beta}" / f"random_corrupt_{Private_Corrupt_Rate}"

# Matrix Update Times
Matrix_Update_Epoch = 1
# ──────────────────────────────────────────────────────────────────────────────


def evaluate_network(network, dataloader, logger):
    network.eval()
    eval_bar = tqdm(dataloader, desc="Evaluating", unit="batch", ncols=80, leave=True)
    with torch.no_grad():
      for images, labels in eval_bar:
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100.0 * correct / total
            eval_bar.set_postfix(acc=f"{acc:.2f}%")

        final_acc = 100.0 * correct / total
        logger.info(f"Test Accuracy of the model on the test images: {final_acc} %")
    print(f"Evaluation complete: Accuracy = {final_acc:.2f}%")
    return final_acc




def update_model_via_private_data(
    device,
    network,
    localtrain_epoch,
    private_dataloader,
    loss_function,
    optimizer_method,
    learning_rate,
    logger
):
    if loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if optimizer_method == 'Adam':
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_method == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    participant_local_loss_batch_list = []
    
    train_bar = tqdm(private_dataloader, desc="Private Training", unit="batch", ncols=80, leave=True)

    for _ in range(localtrain_epoch):
        for batch_idx, (images, labels) in enumerate(train_bar):
            if AugMix_Module == 'jsd':
                # AugMix + JSD
                images_all = torch.cat([images[0], images[1], images[2]], 0).to(device)
                labels = labels.to(device)
                logits_all, _ = network(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                labels = labels.long()

                # Cross-entropy on clean images
                loss = criterion(logits_clean, labels)

                p_clean = F.softmax(logits_clean, dim=1)
                p_aug1  = F.softmax(logits_aug1, dim=1)
                p_aug2  = F.softmax(logits_aug2, dim=1)
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, 1e-7, 1.0).log()
                jsd_loss = (
                    F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug1,  reduction='batchmean') +
                    F.kl_div(p_mixture, p_aug2,  reduction='batchmean')
                ) / 3.0
                loss += 12 * jsd_loss

                if CL_Module == 'supcon':
                    # SupCon feature loss
                    images_cont = torch.cat([images[0], images[1]], 0).to(device)
                    features = network.module.backbone(images_cont)
                    features = F.normalize(features, dim=1)
                    fclean, f1 = torch.split(features, images[0].size(0))
                    features = torch.cat([fclean.unsqueeze(1), f1.unsqueeze(1)], dim=1)
                    cont_loss_fn = SupConLoss(temperature=0.2, device=device)
                    cont_loss = cont_loss_fn(features, labels)
                    loss += cont_loss

                elif CL_Module == 'dcl':
                    # DCL feature loss
                    images_cont = torch.cat([images[0], images[1], images[3]], 0).to(device)
                    features = network.module.backbone(images_cont)
                    features = F.normalize(features, dim=1)
                    fclean1, f1, fclean2 = torch.split(features, images[0].size(0))
                    cont_supddm_loss_fn = DCLLoss(
                        temperature=0.2,
                        device=device,
                        beta=1.0,
                        ddm_temperature=0.2
                    )
                    cont_loss = cont_supddm_loss_fn(
                        original_feature=fclean1.unsqueeze(1),
                        weak_feature    =fclean2.unsqueeze(1),
                        strong_feature  =f1.unsqueeze(1),
                        labels=labels
                    )
                    loss += cont_loss
            else:
                # Original code path
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = network(images)
                labels = labels.long()
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()

            train_bar.set_postfix(loss=f"{loss.item():.6f}")

            logger.info(
                f"Private Train : [{batch_idx * len(images[0])}/{len(private_dataloader.dataset)} "
                f"({100. * batch_idx / len(private_dataloader):.0f}%)] "
                f"Loss: {loss.item():.6f}"
            )
    print("Private training finished")
    return network, participant_local_loss_batch_list


if __name__ == '__main__':
    # Initialize logger
    logger = init_logs()
    logger.info("Random Seed and Server Config")

    # Set seeds
    random.seed(Seed)
    np.random.seed(Seed)
    torch.manual_seed(Seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Seed)

    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    # ───────── Initialize Participants' Data splits ─────────────────────────────
    logger.info("Initialize Participants' Data idxs")
    if Data_Partition == 'noniid':
        _, net_dataidx_map = partition_data(
            dataset=Private_Dataset_Name,
            datadir=str(Private_Dataset_Dir),
            partition=Data_Partition,
            num_classes=Private_Output_Channel,
            num_users=N_Participants,
            corrupt_rate=Private_Corrupt_Rate,
            test_corrupt_rate=Test_Corrupt_Rate,
            dirichlet_beta=Noniid_Dirichlet_Beta
        )
    else:  # iid
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)[:Private_Data_Len]
            net_dataidx_map[index] = idxes

    logger.info(f"Data idx map: {net_dataidx_map}")
    print("Data indices partitioned")

    # ───────── Initialize Networks ─────────────────────────────────────────────
    net_list = init_nets(
        n_parties=N_Participants,
        nets_name_list=Nets_Name_List,
        num_classes=Private_Output_Channel
    )

    print("Networks initialized")
    # ───────── Load Pretrained Private Models ─────────────────────────────────
    logger.info("Load Participants' Models")
    print("Loading pretrained private models")
    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[i]
        ckpt_path = Model_Load_Dir / f"{netname}_{i}.ckpt"
        network.load_state_dict(torch.load(str(ckpt_path)))

    print("Private models loaded")
    # ───────── Initialize Public Data ──────────────────────────────────────────
    logger.info("Initialize Public Data Parameters")
    print("Generating public data indices and dataloader")
    public_data_indexs = generate_public_data_indexs(
        dataset=Public_Dataset_Name,
        datadir=str(Public_Dataset_Dir_Org),
        size=Public_Dataset_Length
    )
    public_train_dl, _, public_train_ds, _ = get_dataloader(
        dataset=Public_Dataset_Name,
        datadir=str(Public_Dataset_Dir),
        train_bs=TrainBatchSize,
        test_bs=TestBatchSize,
        dataidxs=public_data_indexs,
        corrupt_rate=Public_Corrupt_Rate,
        augmix_module=None
    )

    print(f"Public training dataset size: {len(public_train_ds)}")
    col_loss_list    = []
    local_loss_list  = []
    local_jsd_loss_list = []
    acc_list         = []

    for epoch_index in range(CommunicationEpoch):
        print(f"===== Comm Epoch {epoch_index} =====")
        logger.info(f"The {epoch_index}th Communication Epoch")

        # ───────────────── Evaluate Models on Private Test ────────────────────
        print("Starting private evaluation phase")
        logger.info('Evaluate Models')
        acc_epoch_list = []
        if epoch_index % Matrix_Update_Epoch == 0:
            matrix_update_acc_list = []

        for participant_index in range(N_Participants):
            netname = Nets_Name_List[participant_index]
            _, test_dl, _, _ = get_dataloader(
                dataset=Private_Dataset_Name,
                datadir=str(Private_Dataset_Dir),
                train_bs=TrainBatchSize,
                test_bs=TestBatchSize,
                test_corrupt_rate=Test_Corrupt_Rate
            )
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)

            if epoch_index % Matrix_Update_Epoch == 0:
                matrix_update_acc_list.append(accuracy)

        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants
        logger.info(f"Average Test Accuracy across participants: {accuracy_avg} %")
        print(f"Private evaluation complete, average accuracy: {accuracy_avg:.2f}%")

        # ───────── HHF (Public Data Fusion) ───────────────────────────────────
        print("Starting public fusion (HHF) phase")
        for batch_idx, (images, _) in enumerate(public_train_dl):
            linear_output_list         = []
            linear_output_target_list  = []
            kl_loss_batch_list         = []
            participant_kl_list        = []

            # 1) Calculate each participant’s logits and targets
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                images = images.to(device)
                logits, _ = network(images)
                p_logits  = F.softmax(logits, dim=1)
                linear_output_target_list.append(p_logits.clone().detach())
                plog_logits = F.log_softmax(logits, dim=1)
                linear_output_list.append(plog_logits)

            # 2) Update via KL divergence among participants
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()

                criterion = nn.KLDivLoss(reduction='batchmean').to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()

                loss = torch.tensor(0.0, device=device)
                learn_from_client_num = 0

                if Col_Module == 'hfl':
                    for i in range(N_Participants):
                        if i != participant_index:
                            weight_index = 1.0 / (N_Participants - 1)
                            loss_batch_sample = criterion(
                                linear_output_list[participant_index],
                                linear_output_target_list[i]
                            )
                            loss += weight_index * loss_batch_sample
                    kl_loss_batch_list.append(loss.item())
                    loss.backward()
                    optimizer.step()

                elif Col_Module == 'asymhfl':
                    for i in range(N_Participants):
                        if i != participant_index:
                            if matrix_update_acc_list[participant_index] <= matrix_update_acc_list[i]:
                                learn_from_client_num += 1
                                loss_batch_sample = criterion(
                                    linear_output_list[participant_index],
                                    linear_output_target_list[i]
                                )
                                loss += loss_batch_sample
                    if learn_from_client_num > 0:
                        loss = loss / learn_from_client_num
                        kl_loss_batch_list.append(loss.item())
                        loss.backward()
                        optimizer.step()

            col_loss_list.append(kl_loss_batch_list)

        print("Public fusion phase complete")
        # ───────── Update Participants via Private Data ───────────────────────
        print("Starting private update phase")
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()

            train_dl, _, train_ds, _ = get_dataloader(
                dataset=Private_Dataset_Name,
                datadir=str(Private_Dataset_Dir),
                train_bs=TrainBatchSize,
                test_bs=TestBatchSize,
                dataidxs=net_dataidx_map[participant_index],
                corrupt_rate=Private_Corrupt_Rate,
                augmix_module=AugMix_Module
            )

            # Determine local training epochs based on public vs. private sizes
            localtrain_epoch = max(int(len(public_train_ds) / len(train_ds)), 1)

            network, private_loss_batch_list = update_model_via_private_data(
                device=device,
                network=network,
                localtrain_epoch=localtrain_epoch,
                private_dataloader=train_dl,
                loss_function=Pariticpant_Params['loss_funnction'],
                optimizer_method=Pariticpant_Params['optimizer_name'],
                learning_rate=Pariticpant_Params['learning_rate'],
                logger=logger
            )

            mean_private_loss = np.mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss)

        local_loss_list.append(local_loss_batch_list)
        print("Private update phase complete")

        # ───────── Final Evaluation & Save on last epoch ─────────────────────
        if epoch_index == CommunicationEpoch - 1:
            print("Final evaluation round")
            logger.info('Final Evaluate Models')
            final_accs = []
            for participant_index in range(N_Participants):
                _, test_dl, _, _ = get_dataloader(
                    dataset=Private_Dataset_Name,
                    datadir=str(Private_Dataset_Dir),
                    train_bs=TrainBatchSize,
                    test_bs=TestBatchSize,
                    test_corrupt_rate=Test_Corrupt_Rate
                )
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                final_accs.append(accuracy)

            avg_final_acc = sum(final_accs) / N_Participants
            logger.info(f"Average Test Accuracy in final round: {avg_final_acc} %")
            print(f"Final average accuracy: {avg_final_acc:.2f}%")

            # Ensure save directory exists
            mkdirs(str(Model_Save_Dir))
            print(f"Saving final models to {Model_Save_Dir}")
            # Save each participant’s final model
            for participant_index in range(N_Participants):
                netname = Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                ckpt_path = Model_Save_Dir / f"{netname}_{participant_index}.ckpt"
                torch.save(network.state_dict(), str(ckpt_path))
                logger.info(f"Saved model: {ckpt_path}")
            print("Final models saved")