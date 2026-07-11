# EXACT DATA SOURCES IN PWHFL - Where Data Comes From

---

## **1. CIFAR-100 (PUBLIC DATA SOURCE)**

### **File Hierarchy:**
```
Dataset/
├── cifar_100/                          ← Directory on disk
│   ├── train_batch_1 ... train_batch_5
│   └── test_batch
├── cifar.py                            ← CIFAR-100 class definition
├── utils.py                            ← Loading & indexing functions
└── init_dataset.py                     ← Alternative loading module
```

---

### **STEP 1: Physical Data Loading (Dataset/cifar.py, Lines 208-330)**

**Class Definition:**
```python
class CIFAR100(data.Dataset):
    """CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html> Dataset"""
    
    def __init__(self, root, train=True, transform=None, ...):
        self.root = root
        self.train = train
        self.transform = transform
```

**For TRAIN SET (train=True):**
- **Source:** `./Dataset/cifar_100/` directory
- **Binary Files:** `train_batch_1` through `train_batch_5` (pickle files)
- **Each file:** Contains dict with keys `'data'` (image bytes) and `'fine_labels'` (100 classes)
- **Loading (Lines 87-118):**
  ```python
  for batch_name in ['train_batch_1', 'train_batch_2', ..., 'train_batch_5']:
      fo = open(batch_path, 'rb')
      entry = pickle.load(fo, encoding='latin1')
      self.train_data.append(entry['data'])        # numpy array
      self.train_labels.extend(entry['fine_labels'])
  
  self.train_data = np.concatenate(self.train_data, axis=0)  # [50000, 3072]
  self.train_data = self.train_data.reshape((50000, 3, 32, 32))
  self.train_data = self.train_data.transpose((0, 2, 3, 1))  # [50000, 32, 32, 3]
  self.train_labels = np.array(self.train_labels)  # [50000]
  ```
- **Output:** 
  - `train_data`: [50,000, 32, 32, 3] uint8 numpy array
  - `train_labels`: [50,000] numpy array with values 0-99

**For TEST SET (train=False):**
- **Source:** `./Dataset/cifar_100/test_batch` (single pickle file)
- **Loading (Lines 110-130):**
  ```python
  fo = open(test_batch_path, 'rb')
  entry = pickle.load(fo, encoding='latin1')
  self.test_data = entry['data']                   # [10000, 3072]
  self.test_labels = entry['fine_labels']          # [10000] with 0-99
  
  self.test_data = self.test_data.reshape((10000, 3, 32, 32))
  self.test_data = self.test_data.transpose((0, 2, 3, 1))  # [10000, 32, 32, 3]
  ```
- **Output:**
  - `test_data`: [10,000, 32, 32, 3] uint8 numpy array
  - `test_labels`: [10,000] numpy array with values 0-99

---

### **STEP 2: Public Data Index Generation (Dataset/utils.py, Lines 190-199)**

**Function Call in pwhfl.py (Line 332-334):**
```python
pub_idx = generate_public_data_indexs(
    Public_Dataset_Name='cifar100',           # ← Selects CIFAR-100
    datadir=str(Public_Dataset_Dir_Org),     # ← Points to ./Dataset/cifar_100/
    size=Public_Dataset_Length                # ← 5000 samples
)
```

**Exact Function (Dataset/utils.py, Lines 190-199):**
```python
def generate_public_data_indexs(dataset: str, datadir: str, size: int):
    """
    Randomly select `size` indices from the training set of the given dataset.
    """
    data_path = Path(datadir)
    
    # Load CIFAR-100 training set
    if dataset == "cifar100":
        X_train, y_train, X_test, y_test = load_cifar100_data(data_path)
        # X_train: [50000, 32, 32, 3]
        # y_train: [50000]
    
    n_train = y_train.shape[0]              # = 50000
    all_idxs = list(range(n_train))         # = [0, 1, 2, ..., 49999]
    
    # RANDOM SELECTION WITHOUT REPLACEMENT
    return np.random.choice(all_idxs, size, replace=False)
    # Returns: [idx1, idx2, ..., idx_5000] (random order, no duplicates)
```

**Output of `pub_idx`:**
```
[12847, 3456, 49821, 523, ..., 18743]  # 5000 unique indices
```

---

### **STEP 3: Public Dataloader Creation (Dataset/utils.py, Lines 335-339)**

**Call in pwhfl.py:**
```python
public_train_dl, _, public_train_ds, _ = get_dataloader(
    dataset='cifar100',
    datadir=str(Public_Dataset_Dir),        # ← ./Dataset/cifar_100/
    train_bs=TrainBatchSize,                # ← 256
    test_bs=TestBatchSize,
    dataidxs=pub_idx,                       # ← [5000 random indices]
    corrupt_rate=Public_Corrupt_Rate,       # ← 0.0 (no corruption)
    augmix_module=None                      # ← No augmentation (plain CIFAR-100)
)
```

**Inside get_dataloader (Dataset/utils.py, Lines 307-380):**

```python
elif dataset == "cifar100":
    if augmix_module is None:
        # NO AUGMENTATION PATH (public data uses this)
        transform_train = transforms.Compose([
            transforms.ToTensor(),                    # [0, 1]
            # NO COLOR JITTER, NO RANDOM CROP, NO FLIP
        ])
        
        train_ds = Cifar100FL(                        # ← Custom CIFAR-100 class
            str(data_path),
            train=True,
            transform=transform_train,
            dataidxs=pub_idx,                         # ← Apply index filtering HERE
            corrupt_rate=0.0
        )
    
    # Transform for inference
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_ds = Cifar100FL(
        str(data_path),
        train=False,                                  # ← Use test set
        transform=transform_test,
        corrupt_rate=0.0
    )
    
    # Create DataLoaders
    train_dl = data.DataLoader(
        dataset=train_ds,
        batch_size=256,
        drop_last=True,
        shuffle=True,                                 # ← Random batch order
        num_workers=0
    )
```

**DataLoader Characteristics:**
- Batch size: 256
- Total batches: 5000 / 256 ≈ 20 batches (drop_last=True)
- Each batch yields: `(images, labels)` where labels are **ignored** (using `_` in loop)

---

### **STEP 4: Usage in Collaborative Phase (pwhfl.py, Lines 401-402)**

```python
for batch_idx, (images, _) in enumerate(public_train_dl):
    images = images.to(device)              # [256, 3, 32, 32] float32, GPU
    # Loop through 20 batches total
    
    # BATCH 0:
    #   images.shape = [256, 3, 32, 32]
    #   indices used: pub_idx[0:256]
    #   actual CIFAR-100 samples: rows [pub_idx[0], pub_idx[1], ..., pub_idx[255]]
    
    # BATCH 1:
    #   indices used: pub_idx[256:512]
    #   ...
    
    # BATCH 19:
    #   indices used: pub_idx[4744:5000]
```

**SUMMARY: CIFAR-100 Data Flow**
```
DISK: ./Dataset/cifar_100/
     ├── train_batch_1..5 (50K samples, 100 classes)
     └── test_batch (10K samples)
          ↓
Dataset/cifar.py (CIFAR100 class)
     ↓
load_cifar100_data() → [X_train, y_train, X_test, y_test]
     ↓
generate_public_data_indexs() → select 5000 random indices
     ↓
get_dataloader(dataidxs=pub_idx)
     ↓
Cifar100FL dataset (filtered by pub_idx)
     ↓
DataLoader (batch_size=256, shuffle=True)
     ↓
pwhfl.py collaborative phase
     ↓
public_train_dl provides batches [256, 3, 32, 32]
```

---

---

## **2. PRIVATE TEST SET (FORWARD PASS DATA)**

### **File Hierarchy:**
```
Dataset/
├── cifar_10/                             ← CIFAR-10 (different from public!)
│   ├── data_batch_1 ... data_batch_5
│   └── test_batch
├── cifar.py                              ← CIFAR-10 class definition
└── utils.py
```

---

### **STEP 1: Data Partition During Pretraining (Dataset/sampling.py, Lines 1-30)**

**Each client gets its OWN private training set:**

```python
def iid_sampling(y_train, num_users):
    """
    Randomly partition dataset into `num_users` IID chunks.
    """
    num_items = len(y_train)               # = 50000 (CIFAR-10 training set size)
    num_items_per_user = num_items // num_users  # = 50000 / 4 = 12500
    
    user_dict = {i: np.array([], dtype=np.int64) for i in range(num_users)}
    idxs = np.arange(num_items)           # = [0, 1, 2, ..., 49999]
    
    # Random permutation WITHOUT replacement
    idxs_permuted = np.random.permutation(idxs)
    
    # Split into 4 equal parts
    for i in range(num_users):
        user_dict[i] = idxs_permuted[i*num_items_per_user:(i+1)*num_items_per_user]
    
    return user_dict
    # Output:
    # {
    #   0: [49821, 5234, 18765, ..., 3421],  # 12500 indices
    #   1: [12456, 39876, 4521, ..., 28974],  # 12500 indices
    #   2: [26543, 1234, 45678, ..., 9876],  # 12500 indices
    #   3: [48765, 23456, 7654, ..., 38901]  # 12500 indices
    # }
```

**PRIVATE DATA ASSIGNMENT:**
- **Client 0:** 12,500 private CIFAR-10 training samples
- **Client 1:** 12,500 private CIFAR-10 training samples
- **Client 2:** 12,500 private CIFAR-10 training samples
- **Client 3:** 12,500 private CIFAR-10 training samples

---

### **STEP 2: Test Set Loading (During Pretraining)**

**In Network/pretrain.py (around lines 60-80):**
```python
# Load private test set (SHARED across all clients)
_, _, test_dl = get_data_loader_cifar10(
    root='./Dataset/cifar_10',             # ← CIFAR-10 test directory
    batch_size=64,
    train=False                             # ← Load test set
)
# test_dl provides batches from CIFAR-10 test set
```

**CIFAR-10 Test Set Details:**
- **Source:** `./Dataset/cifar_10/test_batch` (binary pickle)
- **Size:** 10,000 samples
- **Classes:** 10 (0-9)
- **Shape:** [10,000, 32, 32, 3]
- **Format:** uint8 numpy arrays

---

### **STEP 3: Test Set in Collaborative Phase (pwhfl.py)**

**In pwhfl.py (Lines 360-375):**

```python
# Load test set ONCE at beginning
_, _, test_dl = get_dataloader(
    dataset='cifar10',
    datadir=str(Test_Dataset_Dir),         # ← ./Dataset/cifar_10/
    train_bs=TrainBatchSize,
    test_bs=TestBatchSize,                 # ← 512
    dataidxs=None,                         # ← Use ALL test samples
    corrupt_rate=0.0,
    test_corrupt_rate=Test_Corrupt_Rate,   # ← 0.0
    augmix_module=None
)

# test_dl contains:
# - Total samples: 10,000
# - Batch size: 512
# - Total batches: 10000 / 512 ≈ 20 batches
# - Each batch: [512, 3, 32, 32] float32
```

---

### **STEP 4: Evaluation on Test Set (pwhfl.py, Lines 365-380)**

**During Each Communication Round:**

```python
for epoch_index in range(num_epochs):
    # ... training ...
    
    # EVALUATION PHASE
    acc_epoch_list = []
    
    for p in range(N_Participants):  # For each of 4 clients
        network = net_list[p]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        
        # FORWARD PASS on ENTIRE test set
        accuracy = evaluate_network(network, test_dl, logger)
        #                                        ↑
        #                        This test_dl has 10K CIFAR-10 samples
        
        acc_epoch_list.append(accuracy)
```

---

### **STEP 5: evaluate_network() Function (Detailed)**

**Inside evaluate_network (pwhfl.py or utils.py):**

```python
def evaluate_network(net, test_dl, logger):
    net.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        # Iterate through ALL test batches
        for batch_idx, (x, y) in enumerate(test_dl):
            # Each iteration:
            x = x.to(device)    # [512, 3, 32, 32] → GPU
            y = y.to(device)    # [512] → GPU (labels 0-9)
            
            # FORWARD PASS (no gradients)
            logits, _ = net(x)  # [512, 10] output logits
            
            # Get predictions
            _, predicted = torch.max(logits.data, 1)  # [512]
            
            # Count correct
            total += y.size(0)                 # += 512
            correct += (predicted == y).sum().item()
        
        accuracy = 100.0 * correct / total     # % correct
    
    return accuracy
```

**Example Execution:**
```
Batch 0: x=[512, 3, 32, 32], y=[512]
         → logits=[512, 10]
         → correct += (predicted == y).sum()
         → total = 512

Batch 1: x=[512, 3, 32, 32], y=[512]
         → logits=[512, 10]
         → correct += (predicted == y).sum()
         → total = 1024

...

Batch 19: x=[512, 3, 32, 32], y=[512]
          → logits=[512, 10]
          → correct += (predicted == y).sum()
          → total = 10240 (exceeds 10K due to drop_last)

Final: accuracy = 100.0 * correct / 10000 = XX.XX%
```

---

### **STEP 6: Where Test Data Comes From - Exact Files**

**Physical Disk Location:**
```
./Dataset/cifar_10/
├── readme.html
├── data_batch_1          ← pickle, contains training samples
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
└── test_batch            ← pickle, contains 10,000 test samples
                              Binary format: {'data': [10000, 3072], 'labels': [10000]}
```

**Loading Process (Dataset/cifar.py, Lines 87-131):**

```python
class CIFAR10(data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        
        if self.train:
            # Load 5 training batches
            self.train_data = []
            self.train_labels = []
            for batch_name in self.train_list:  # data_batch_1..5
                path = os.path.join(self.root, batch_name)
                with open(path, 'rb') as f:
                    entry = pickle.load(f, encoding='latin1')
                self.train_data.append(entry['data'])
                self.train_labels.extend(entry['labels'])
            
            self.train_data = np.concatenate(self.train_data)  # [50000, 3072]
            self.train_data = self.train_data.reshape((50000, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))
        else:
            # Load test batch
            path = os.path.join(self.root, 'test_batch')
            with open(path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
            self.test_data = entry['data']      # [10000, 3072]
            self.test_labels = entry['labels']  # [10000]
            
            self.test_data = self.test_data.reshape((10000, 3, 32, 32))
            self.test_data = self.test_data.transpose((0, 2, 3, 1))
```

---

## **SUMMARY TABLE: EXACT DATA SOURCES**

| Aspect | CIFAR-100 (Public) | CIFAR-10 (Private Test) |
|--------|-------------------|------------------------|
| **Disk Location** | `./Dataset/cifar_100/` | `./Dataset/cifar_10/` |
| **Dataset Type** | 100-class classification | 10-class classification |
| **Training Samples** | 50,000 | (Private, 12,500 per client) |
| **Test Samples** | 10,000 | 10,000 |
| **How Selected** | Random 5,000 indices via `np.random.choice()` | ALL 10,000 test samples |
| **Loading Function** | `load_cifar100_data(datadir)` | `get_dataloader(train=False)` |
| **Index Filtering** | `Cifar100FL(dataidxs=pub_idx)` | No filtering (use all) |
| **Batch Size** | 256 | 512 |
| **Augmentation** | None (plain ToTensor) | None (plain ToTensor) |
| **Usage in PWHFL** | MCE teacher targets (20 batches) | Evaluation/accuracy (20 batches) |
| **Numpy Array Shape** | [5000, 32, 32, 3] per epoch | [10000, 32, 32, 3] per epoch |
| **Data Type** | uint8 (disk) → float32 (GPU) | uint8 (disk) → float32 (GPU) |

---

## **VISUAL FLOW**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PWHFL Data Sources                           │
└─────────────────────────────────────────────────────────────────┘

DISK:
├── ./Dataset/cifar_100/ (50K + 10K samples, 100 classes)
│   ├── train_batch_1..5
│   └── test_batch
│        └─→ generate_public_data_indexs() → 5000 random indices
│             └─→ get_dataloader(dataidxs=pub_idx)
│                  └─→ public_train_dl (20 batches of 256)
│                       └─→ MCE Teacher Targets + PWKT Update
│
└── ./Dataset/cifar_10/ (50K + 10K samples, 10 classes)
    ├── data_batch_1..5 (used for pretraining private data)
    └── test_batch (10K samples)
         └─→ get_dataloader(train=False)
              └─→ test_dl (20 batches of 512)
                  └─→ evaluate_network() → Accuracy per client per round
```

---

## **KEY TAKEAWAYS**

1. **CIFAR-100 public data:** 
   - Loaded from disk every training run
   - Random 5,000 indices selected via `np.random.choice(range(50000), 5000, replace=False)`
   - Used for MCE targets and PWKT calculations
   - **Different from private data** (different dataset, different classes)

2. **CIFAR-10 private test set:**
   - All 10,000 samples used (no sampling)
   - Loaded once at initialization
   - Used for evaluation/accuracy computation
   - **Same test set** evaluated for all 4 clients each round
   - Forward pass: no gradient computation, only inference

3. **Data Picking Mechanism:**
   - **Public:** `np.random.choice()` with fixed random seed
   - **Private test:** No picking—use all 10,000 samples
