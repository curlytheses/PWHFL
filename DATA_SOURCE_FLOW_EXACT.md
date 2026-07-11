# EXACT DATA SOURCE FLOW - PWHFL

## ═══════════════════════════════════════════════════════════════════════════════
## DATA SOURCE 1: CIFAR-100 (PUBLIC DATA) 
## ═══════════════════════════════════════════════════════════════════════════════

### Physical Location on Disk
```
./Dataset/cifar_100/
├── train  (pickle file - CIFAR-100 training split)
│   ├── data: numpy array [50000, 32, 32, 3] uint8
│   ├── labels: array [50000] (100 classes: 0-99)
│   └── batch_label: "training batch 1"
└── test   (pickle file - CIFAR-100 test split)
    ├── data: numpy array [10000, 32, 32, 3] uint8
    └── labels: array [10000]

./ Dataset/cifar_100_c/  ← SAME DATA but with corruption applied
├── train/
│   ├── clean.npy [50000, 32, 32, 3]
│   └── random_corrupt_0.5.npy [50000, 32, 32, 3]  ← corrupted version
└── test/
    └── ...
```

### Exact Picking Code (pwhfl.py Lines 332-339)

```python
# LINE 56-57: Configuration
Public_Dataset_Name    = 'cifar100'
Public_Dataset_Dir_Org = ROOT_DIR / "Dataset" / "cifar_100"      # ← ORIGINAL
Public_Dataset_Dir     = ROOT_DIR / "Dataset" / "cifar_100_c"    # ← CORRUPTED
Public_Dataset_Length  = 5000

# LINE 332-334: Random Index Selection
pub_idx = generate_public_data_indexs(
    Public_Dataset_Name, str(Public_Dataset_Dir_Org), Public_Dataset_Length
)
```

### What generate_public_data_indexs() Does (utils.py Lines 190-199)

```python
def generate_public_data_indexs(dataset: str, datadir: str, size: int):
    """Randomly select `size` indices from the training set."""
    data_path = Path(datadir)
    if dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(data_path)
        # X_train shape: [50000, 32, 32, 3]
        # y_train shape: [50000]
        
        n_train = y_train.shape[0]  # = 50000
        all_idxs = list(range(n_train))  # = [0, 1, 2, ..., 49999]
        
        return np.random.choice(all_idxs, size, replace=False)
        # Returns: random 5000 unique indices from [0..49999]
        # Example: [12847, 3456, 49821, 128, 31245, ...]
```

### Data Flow: Disk → Memory → GPU

```
STEP 1: LOAD FROM DISK
─────────────────────────────────────────────────────────────
┌─ load_cifar100_data(Path("./Dataset/cifar_100"))
│  └─ Opens pickle files from disk
│     ├─ Read "train" pickle → numpy array [50000, 32, 32, 3]
│     └─ Read "test" pickle → numpy array [10000, 32, 32, 3]
│
└─ Returns: X_train, y_train, X_test, y_test

STEP 2: RANDOM INDEX SELECTION
─────────────────────────────────────────────────────────────
┌─ generate_public_data_indexs("cifar100", "./Dataset/cifar_100", 5000)
│  └─ np.random.choice([0..49999], 5000, replace=False)
│
└─ pub_idx = [12847, 3456, 49821, 128, 31245, ..., 48302]  # 5000 indices

STEP 3: CREATE DATALOADER
─────────────────────────────────────────────────────────────
┌─ get_dataloader(
│    dataset="cifar100",
│    datadir="./Dataset/cifar_100_c",     # ← CORRUPTED version used!
│    dataidxs=pub_idx,                    # ← Our 5000 indices
│    train_bs=256,
│    corrupt_rate=Public_Corrupt_Rate     # = 0.5
│  )
│
├─ Load from ./Dataset/cifar_100_c/train/random_corrupt_0.5.npy
│  └─ Shape: [50000, 32, 32, 3]
│
├─ Filter by pub_idx
│  └─ data = corrupted_array[pub_idx]
│     → Shape becomes [5000, 32, 32, 3]
│
├─ Apply Transforms (lines 309-318):
│  ├─ transforms.ToPILImage()
│  ├─ transforms.RandomCrop(32, padding=4)
│  ├─ transforms.RandomHorizontalFlip()
│  ├─ transforms.RandomRotation(15)
│  ├─ transforms.ToTensor()  
│  └─ transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
│
└─ Create DataLoader(batch_size=256, shuffle=False)
   └─ Output: 20 batches of [256, 3, 32, 32] float32

STEP 4: FEED TO MODEL DURING MCE
─────────────────────────────────────────────────────────────
For round = 0 to 39:
    For batch_idx = 0 to 19:  # 20 batches total
        batch_images, batch_labels = public_train_dl[batch_idx]
        # Shape: images [256, 3, 32, 32], labels [256]
        
        For each teacher model in [ResNet10, ResNet12, ShuffleNet, MobileNetV2]:
            teacher_targets = mce_teacher_target(
                teacher_model, 
                batch_images,           # [256, 3, 32, 32]
                n_crops=MCE_N_CROPS,    # 2 (original + horizontal flip)
                temperature=TEACHER_TEMP # 0.65
            )
            # Output shape: [256, 100]  ← soft targets for 100 CIFAR-100 classes
```

---

## ═══════════════════════════════════════════════════════════════════════════════
## DATA SOURCE 2: CIFAR-10 PRIVATE TEST SET
## ═══════════════════════════════════════════════════════════════════════════════

### Physical Location on Disk
```
./Dataset/cifar_10_c/
├── train/
│   ├── clean.npy [50000, 32, 32, 3]
│   └── random_corrupt_0.5.npy [50000, 32, 32, 3]
└── test/
    ├── clean.npy [10000, 32, 32, 3]  ← USED FOR EVALUATION!
    └── random_corrupt_0.5.npy [10000, 32, 32, 3]
```

### Exact Picking Code (pwhfl.py Lines 365-373)

```python
# LINE 49-50: Configuration
Private_Dataset_Name   = 'cifar10'
Private_Dataset_Dir    = ROOT_DIR / "Dataset" / "cifar_10_c"
Test_Corrupt_Rate      = 0   # ← Uses CLEAN test set

# LINE 365-370: Load Test DataLoader (runs EVERY epoch)
for p in range(N_Participants):  # for each client 0,1,2,3
    _, test_dl, _, _ = get_dataloader(
        dataset=Private_Dataset_Name,           # = 'cifar10'
        datadir=str(Private_Dataset_Dir),       # = './Dataset/cifar_10_c'
        train_bs=TrainBatchSize,                # = 256
        test_bs=TestBatchSize,                  # = 512
        corrupt_rate=Private_Corrupt_Rate,      # = 0.5 (NOT USED FOR TEST)
        test_corrupt_rate=Test_Corrupt_Rate     # = 0    ← CLEAN TEST
    )
    # Note: NO dataidxs parameter → uses ALL test samples
    
    # LINE 373: Run Evaluation
    accuracy = evaluate_network(network, test_dl, logger)
```

### What get_dataloader() Does for CIFAR-10 (utils.py Lines 221-304)

```python
def get_dataloader(
    dataset="cifar10",
    datadir="./Dataset/cifar_10_c",
    ...
    test_corrupt_rate=0,      # ← This determines which test file!
    ...
):
    # ─ CIFAR-10 Branch (lines 221-305)
    if dataset == "cifar10":
        
        # TEST DATASET CREATION (lines 292-304)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                std=[x/255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        
        test_ds = CIFAR_C(
            str(data_path),                    # = "./Dataset/cifar_10_c"
            train=False,                       # ← Load test set (not train)
            transform=transform_test,
            corrupt_rate=test_corrupt_rate     # = 0
        )
```

### What CIFAR_C Dataset Does (init_dataset.py Lines 138-174)

```python
class CIFAR_C(Dataset):
    def __init__(self, root: str, dataidxs=None, train=True, 
                 transform=None, corrupt_rate=0):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.corrupt_rate = corrupt_rate
        self.data, self.target = self.Construct_Participant_Dataset()

    def Construct_Participant_Dataset(self):
        root_path = Path(self.root)
        
        # PICK FILE BASED ON train & corrupt_rate
        if self.train:
            data_path = root_path / "train"
        else:
            data_path = root_path / "test"    # ← for evaluation
        
        # PICK CORRUPTION LEVEL
        if self.corrupt_rate == 0:
            data = np.load(data_path / "clean.npy")
            # Loads: ./Dataset/cifar_10_c/test/clean.npy
            # Shape: [10000, 32, 32, 3]
        else:
            data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
            # Would load if corrupt_rate > 0
        
        target = np.load(data_path / "labels.npy")
        # Loads: ./Dataset/cifar_10_c/test/labels.npy
        # Shape: [10000]
        
        # KEY DIFFERENCE FROM PUBLIC DATA:
        # dataidxs is None here (not provided in get_dataloader call)
        if self.dataidxs is not None:
            data = data[self.dataidxs]      # ← Would filter here, but dataidxs=None
            target = target[self.dataidxs]
        
        # NO FILTERING → Use all 10000 samples!
        return data, target
        # Final shapes: [10000, 32, 32, 3], [10000]
```

### Data Flow: Disk → Memory → Forward Pass

```
STEP 1: LOAD FROM DISK
─────────────────────────────────────────────────────────────
┌─ CIFAR_C(root="./Dataset/cifar_10_c", train=False, corrupt_rate=0)
│  └─ Construct_Participant_Dataset()
│     ├─ data_path = "./Dataset/cifar_10_c/test"
│     ├─ Load data = np.load("./Dataset/cifar_10_c/test/clean.npy")
│     │  └─ Shape: [10000, 32, 32, 3]  ALL 10000 TEST SAMPLES
│     └─ Load target = np.load("./Dataset/cifar_10_c/test/labels.npy")
│        └─ Shape: [10000]
│
└─ No index filtering (dataidxs=None)

STEP 2: CREATE DATALOADER
─────────────────────────────────────────────────────────────
┌─ DataLoader(dataset=test_ds, batch_size=512, shuffle=False)
│
└─ Output: 20 batches of [512, 32, 32, 3] float32
   ├─ Batch 0: samples [0:512]
   ├─ Batch 1: samples [512:1024]
   ├─ ...
   └─ Batch 19: samples [9488:10000]

STEP 3: FORWARD PASS (EVALUATION) - pwhfl.py Lines 161-178
─────────────────────────────────────────────────────────────
def evaluate_network(network, dataloader, logger):
    network.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():  # ← NO GRADIENTS, INFERENCE ONLY
        for batch_idx, (images, labels) in enumerate(test_dl):
            # Batch shape: images [512, 3, 32, 32]
            #             labels [512]
            
            images = images.to(device)    # → GPU
            labels = labels.to(device)    # → GPU
            
            outputs, _ = network(images)
            # outputs shape: [512, 10]  ← 10 CIFAR-10 classes
            
            _, predicted = torch.max(outputs.data, 1)
            # predicted shape: [512]  ← argmax class for each sample
            
            total   += labels.size(0)       # += 512
            correct += (predicted == labels).sum().item()
            # Count how many predictions match ground truth
    
    acc = 100.0 * correct / total
    # After 20 batches: acc = (# correct out of 10000) * 100
    
    logger.info(f"Accuracy: {acc:.2f}%")
    return acc
```

---

## ═══════════════════════════════════════════════════════════════════════════════
## COMPARISON: PUBLIC vs PRIVATE DATA
## ═══════════════════════════════════════════════════════════════════════════════

| Aspect | CIFAR-100 (Public) | CIFAR-10 (Test) |
|--------|-------------------|-----------------|
| **Source** | `./Dataset/cifar_100` | `./Dataset/cifar_10_c/test/` |
| **Classes** | 100 | 10 |
| **Size** | 50,000 (training) | 10,000 (test) |
| **Picking Method** | Random 5,000 indices via `np.random.choice()` | NO picking - use ALL 10,000 samples |
| **File Loaded** | `cifar_100_c/train/random_corrupt_0.5.npy` | `cifar_10_c/test/clean.npy` |
| **Array Shape** | [50000, 32, 32, 3] then filtered to [5000, 32, 32, 3] | [10000, 32, 32, 3] (no filtering) |
| **Purpose** | MCE teacher target generation (collabo phase) | Accuracy evaluation (every epoch) |
| **Used In Round** | 20 batches × 40 rounds = 800 batches | 20 batches × 40 rounds = 800 forward passes |
| **Transformations** | RandomCrop, RandomRotation, RandomHFlip, Normalize | Only Normalize |
| **No. of Batches** | 20 (256 samples/batch) | 20 (512 samples/batch) |

---

## ═══════════════════════════════════════════════════════════════════════════════
## EXECUTION SEQUENCE WHEN pwhfl.py RUNS
## ═══════════════════════════════════════════════════════════════════════════════

```
LINE 281-298: Setup & Seeds
├─ Set CUDA device to GPU 0
├─ Set random seeds (deterministic)
└─ device = cuda:0

LINE 313-316: Load Private Data Indices
├─ net_dataidx_map = {
│    0: [random 10000 indices],
│    1: [random 10000 indices],
│    2: [random 10000 indices],
│    3: [random 10000 indices]
│  }
└─ (4 clients, each gets 10K random samples from CIFAR-10 train)

LINE 319-329: Load & Initialize Networks
├─ net_list = [ResNet10, ResNet12, ShuffleNet, MobileNetV2]
├─ Load pretrained weights:
│  ├─ ResNet10: ./Network/Model_Storage_.../ResNet10_0.ckpt
│  ├─ ResNet12: ./Network/Model_Storage_.../ResNet12_1.ckpt
│  ├─ ShuffleNet: ./Network/Model_Storage_.../ShuffleNet_2.ckpt
│  └─ MobileNetV2: ./Network/Model_Storage_.../Mobilenetv2_3.ckpt
└─ All loaded to GPU

LINE 332-339: Load PUBLIC DATA (CIFAR-100)
├─ pub_idx = generate_public_data_indexs("cifar100", "./Dataset/cifar_100", 5000)
│  └─ Returns: [12847, 3456, ..., 48302]  (5000 random unique indices)
│
└─ public_train_dl = get_dataloader(
     dataset="cifar100",
     datadir="./Dataset/cifar_100_c",
     dataidxs=pub_idx,
     train_bs=256,
     test_bs=512
   )
   └─ Creates 20 batches of [256, 3, 32, 32] CIFAR-100 data


═════════════════════════════════════════════════════════════════════════════════
FOR EPOCH = 0 TO 39:  (40 communication rounds)
═════════════════════════════════════════════════════════════════════════════════

    LINE 360-374: STEP 1 - EVALUATION (FORWARD PASS ON CIFAR-10 TEST)
    ┌─ For each client p in [0, 1, 2, 3]:
    │  ├─ _, test_dl, _, _ = get_dataloader(
    │  │    dataset="cifar10",
    │  │    datadir="./Dataset/cifar_10_c",
    │  │    test_bs=512,
    │  │    test_corrupt_rate=0              ← CLEAN test set
    │  │  )
    │  │
    │  ├─ test_ds = CIFAR_C(
    │  │    "./Dataset/cifar_10_c",
    │  │    train=False,
    │  │    corrupt_rate=0
    │  │  )
    │  │  └─ Loads ALL 10000 samples from cifar_10_c/test/clean.npy
    │  │
    │  ├─ test_dl creates 20 batches [512, 3, 32, 32]
    │  │
    │  └─ accuracy = evaluate_network(net_list[p], test_dl, logger)
    │     ├─ Forward pass: for batch in test_dl (20 iterations)
    │     ├─ Compute: outputs = model(images_batch)  [512, 10]
    │     ├─ Compare: predicted == labels → accuracy
    │     └─ Returns: acc_p (accuracy for client p)
    │
    └─ acc_epoch_list = [acc_0, acc_1, acc_2, acc_3]
       ├─ Example: [85.3%, 87.1%, 83.5%, 84.8%]
       └─ Logged to logger

    LINE 380-430: STEP 2 - COLLABORATIVE PHASE (MCE + PWKT + KD)
    ┌─ For each batch_idx in range(len(public_train_dl)):  # 20 batches
    │
    │  ├─ pub_images, pub_labels = public_train_dl[batch_idx]
    │  │  └─ Shape: [256, 3, 32, 32] from CIFAR-100
    │  │
    │  ├─ For each teacher_idx in [0, 1, 2, 3]:
    │  │  ├─ teacher_targets = mce_teacher_target(
    │  │  │    net_list[teacher_idx],
    │  │  │    pub_images,              [256, 3, 32, 32]
    │  │  │    n_crops=MCE_N_CROPS,     2
    │  │  │    temperature=0.65
    │  │  │  )
    │  │  │  ├─ View 1: original images [256, 3, 32, 32]
    │  │  │  ├─ View 2: horizontal flip [256, 3, 32, 32]
    │  │  │  ├─ Average softmax outputs
    │  │  │  └─ teacher_targets: [256, 100]  ← soft probabilities
    │  │  │
    │  │  └─ Store teacher_targets[teacher_idx] = [256, 100]
    │  │
    │  └─ For each student_idx in [0, 1, 2, 3]:
    │     ├─ Compute PWKT weights:
    │     │  weight_j = exp((acc_j - acc_student) / PWKT_TAU)
    │     │  PWKT_TAU = max(5.0, 20.0*(1-Test_Corrupt_Rate)) = 20.0
    │     │
    │     ├─ Compute weighted KD loss:
    │     │  loss_kl = sum(weight_j * KL(student_logits, teacher_targets_j))
    │     │
    │     ├─ ARC check:
    │     │  if accuracy declining → add rescue loss
    │     │
    │     └─ Backward & update student weights via Adam

    LINE 435-475: STEP 3 - LOCAL TRAINING (CIFAR-10 PRIVATE DATA)
    ┌─ For each client p in [0, 1, 2, 3]:
    │
    │  ├─ private_dl = get_dataloader(
    │  │    dataset="cifar10",
    │  │    dataidxs=net_dataidx_map[p],     ← Client p's 10K samples!
    │  │    train_bs=256,
    │  │    augmix_module='jsd'
    │  │  )
    │  │
    │  ├─ Train for 1 epoch:
    │  │  └─ For each batch in private_dl:
    │  │     ├─ images_aug = [original, weak_aug, strong_aug]  3 views
    │  │     ├─ Forward pass → logits
    │  │     ├─ Loss = CE + λ*JSD + DCL
    │  │     ├─ Backward → gradients
    │  │     └─ Adam.step() → update weights
    │  │
    │  └─ Store updated model weights

═════════════════════════════════════════════════════════════════════════════════
END FOR LOOP (after 40 communication rounds)
═════════════════════════════════════════════════════════════════════════════════

Final Results:
├─ acc_history = [max acc for each of 4 clients across 40 rounds]
├─ Logged to ./Logs/pwhfl.log
└─ Models saved to ./APEX_Storage_hetero_4client/.../
```

---

## ═══════════════════════════════════════════════════════════════════════════════
## KEY FACTS
## ═══════════════════════════════════════════════════════════════════════════════

1. **CIFAR-100 picking is RANDOM:**
   - From 50,000 training samples → randomly select 5,000
   - Done ONCE at startup via `np.random.choice()`
   - Same 5,000 indices used for all 40 rounds

2. **CIFAR-10 test picking is DETERMINISTIC:**
   - ALL 10,000 test samples → NO random selection
   - Used as-is every epoch for evaluation
   - Deterministic: always the same 10K samples, same order

3. **Different CIFAR datasets:**
   - CIFAR-100: 100 classes, used for PUBLIC collaborative phase
   - CIFAR-10: 10 classes, used for PRIVATE local training & evaluation

4. **Data corruption:**
   - Public data (CIFAR-100): loaded from `cifar_100_c/` with corruption_rate=0.5
   - Private test data: loaded from `cifar_10_c/test/clean.npy` with corruption_rate=0
   - This means test evaluation uses CLEAN data, but MCE uses CORRUPTED data

5. **No data leakage between phases:**
   - Public data (CIFAR-100) never touches local training
   - Private test data never used for training
   - MCE uses different dataset than evaluation
