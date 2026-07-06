import os
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.optim as optim
import torch.nn as nn

from pathlib import Path
from numpy import *

# ──────────────────────────────────────────────────────────────────────────────
# Ensure Python can import from the parent directories
sys.path.append("..")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from RAHFL.Dataset.utils import init_logs, get_dataloader, init_nets, mkdirs, partition_data
# ──────────────────────────────────────────────────────────────────────────────

# Set ROOT_DIR to the current working directory (should be /content/drive/MyDrive/Colab Notebooks/RAHFL)
ROOT_DIR = Path.cwd()

# ──────────────────────────────────────────────────────────────────────────────
# Global Parameters
Seed             = 7
TrainBatchSize   = 256
TestBatchSize    = 512
Pretrain_Epoch   = 40

Pariticpant_Params = {
    "loss_funnction": "CE",
    "optimizer_name": "Adam",
    "learning_rate": 0.001
}

AugMix_Module = None  # ['jsd', 'nojsd', None]

# Corruption Setting
Private_Corrupt_Rate = 0.5  # [0, 1, 0.5]
Test_Corrupt_Rate    = 0  # [0, 1]

# Model Setting
Model_setting   = "hetero"  # ['hetero', 'homo']
Nets_Name_List  = ["ResNet10", "ResNet12", "ShuffleNet", "Mobilenetv2"]
N_Participants  = len(Nets_Name_List)

# Dataset Setting
Private_Dataset_Name = "cifar10"
Private_Dataset_Dir  = ROOT_DIR / "RAHFL" / "Dataset/cifar_10_c" 
Private_Data_Len     = 10000
Private_Output_Channel = 10

Data_Partition      = "iid"  # ['iid', 'noniid']
Noniid_Dirichlet_Beta = 0    # iid:0 ; noniid:1.0

# Model Save Setting
# Build MODEL_SAVE_DIR as a pathlib.Path under ROOT_DIR
Model_Save_Dir = (
    ROOT_DIR / "RAHFL"
    / "Network"
    / f"Model_Storage_{Model_setting}_4client"
    / f"{Private_Dataset_Name}_{Data_Partition}_{Noniid_Dirichlet_Beta}"
    / f"random_corrupt_{Private_Corrupt_Rate}"
)
# ──────────────────────────────────────────────────────────────────────────────


def pretrain_network(network, localtrain_epoch, private_dataloader, loss_function, optimizer_method, learning_rate, logger):
    if loss_function == "CE":
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    if optimizer_method == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_method == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    network.train()
    num_batches = len(private_dataloader)

    for epoch in range(localtrain_epoch):
        progress_bar = tqdm(
            private_dataloader,
            total=num_batches,
            desc=f"Epoch {epoch + 1}/{localtrain_epoch}",
            unit="batch",
            ncols=80,
            leave=True
        )
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = network(images)
            labels = labels.long()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.6f}")
            logger.info(
                "Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    batch_idx * len(images),
                    len(private_dataloader.dataset),
                    100.0 * batch_idx / len(private_dataloader),
                    loss.item(),
                )
            )
    return network


def evaluate_network(network, dataloader, logger):
    network.eval()
    eval_bar = tqdm(
        dataloader,
        desc="Evaluating",
        unit="batch",
        ncols=80,
        leave=True
    )
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in eval_bar:
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
    return acc


if __name__ == "__main__":
    # Ensure the model save directory exists
    mkdirs(str(Model_Save_Dir))

    logger = init_logs()
    logger.info("Random Seed and Server Config")
    print("Random Seed and Server Config")

    # Set seeds for reproducibility
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_ids = [0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs")

    if Data_Partition == "noniid":
        net_datanum_map = {}
        _, net_dataidx_map = partition_data(
            dataset=Private_Dataset_Name,
            datadir=str(Private_Dataset_Dir),
            partition=Data_Partition,
            num_classes=Private_Output_Channel,
            num_users=N_Participants,
            corrupt_rate=Private_Corrupt_Rate,
            test_corrupt_rate=Test_Corrupt_Rate,
            dirichlet_beta=Noniid_Dirichlet_Beta,
        )
    elif Data_Partition == "iid":
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)
            idxes = idxes[:Private_Data_Len]
            net_dataidx_map[index] = idxes

    logger.info(net_dataidx_map)

    net_list = init_nets(
        n_parties=N_Participants,
        nets_name_list=Nets_Name_List,
        num_classes=Private_Output_Channel
    )

    logger.info("Pretrain Participants Models")
    print("Pretrain Participants Models")

    for index in range(N_Participants):
        train_dl, _, train_ds, _ = get_dataloader(
            dataset=Private_Dataset_Name,
            datadir=str(Private_Dataset_Dir),
            train_bs=TrainBatchSize,
            test_bs=TestBatchSize,
            dataidxs=net_dataidx_map[index],
            corrupt_rate=Private_Corrupt_Rate,
            test_corrupt_rate = Test_Corrupt_Rate,
            augmix_module=AugMix_Module,
        )

        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]

        logger.info(
            f"Pretrain the {index}th Participant Model with N_training: {len(train_ds)}"
        )
        print(f"Pretrain the {index}th Participant Model with N_training: {len(train_ds)}")

        network = pretrain_network(
            network=network,
            localtrain_epoch=Pretrain_Epoch,
            private_dataloader=train_dl,
            loss_function=Pariticpant_Params["loss_funnction"],
            optimizer_method=Pariticpant_Params["optimizer_name"],
            learning_rate=Pariticpant_Params["learning_rate"],
            logger=logger,
        )

        logger.info(f"Save the {index}th Participant Model")
        print(f"Save the {index}th Participant Model")

        ckpt_path = Model_Save_Dir / f"{netname}_{index}.ckpt"
        torch.save(network.state_dict(), str(ckpt_path))

    logger.info("Evaluate Models")
    print("Evaluate Models")

    test_accuracy_list = []
    for index in range(N_Participants):
        _, test_dl, _, _ = get_dataloader(
            dataset=Private_Dataset_Name,
            datadir=str(Private_Dataset_Dir),
            train_bs=TrainBatchSize,
            test_bs=TestBatchSize,
            corrupt_rate=Private_Corrupt_Rate,
            test_corrupt_rate=Test_Corrupt_Rate,
        )

        network = net_list[index]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[index]

        ckpt_path = Model_Save_Dir / f"{netname}_{index}.ckpt"
        network.load_state_dict(torch.load(str(ckpt_path)))

        output = evaluate_network(network=network, dataloader=test_dl, logger=logger)
        test_accuracy_list.append(output)

    print(f"The average Accuracy of models on the test images: {mean(test_accuracy_list)}")