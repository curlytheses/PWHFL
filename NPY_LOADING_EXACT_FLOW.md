# .NPY FILE LOADING - EXACT FLOW WITH CODE

## COMPLETE EXECUTION PATH

### Entry Point: `pwhfl.py` Line 332-333 (PUBLIC DATA)
```python
pub_dl = get_dataloader(
    dataset='cifar100',
    datadir='./Dataset/cifar_100_c/',
    dataidxs=pub_idx,           # [12847, 3456, 49821, ...] 5000 indices
    train=True,
    batch_size=256
)
```

---

## STEP 1: `get_dataloader()` FUNCTION - Dataset/utils.py Lines 210-400

**File:** `/vercel/share/v0-project/Dataset/utils.py`

```python
def get_dataloader(
    dataset='cifar10',
    datadir='./Dataset/',
    dataidxs=None,
    train=True,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    corrupt_rate=0,
    test_corrupt_rate=0,
    augmix_module=None
):
    """Create PyTorch DataLoaders"""
    data_path = Path(datadir)  # Line 218: Path('./Dataset/cifar_100_c/')

    # ─── CIFAR-100 BRANCH ─────────────────────────────────────
    elif dataset == "cifar100":  # Line 307
        if augmix_module is None:
            # Create transform pipeline
            transform_train = transforms.Compose([...])  # Line 309+
            
            # ✓ THIS IS WHERE THE .NPY FILES GET PICKED UP!
            # Calls CIFAR_C class which loads the actual .npy files
            train_ds = CIFAR_C(
                str(data_path),                # './Dataset/cifar_100_c/'
                dataidxs=dataidxs,             # [12847, 3456, ...] 5000 indices
                train=True,                    # Use train/clean.npy
                transform=transform_train,
                corrupt_rate=corrupt_rate      # 0 = use clean.npy
            )
        
        transform_test = transforms.Compose([...])
        test_ds = CIFAR_C(
            str(data_path),                    # './Dataset/cifar_100_c/'
            dataidxs=None,                     # NO FILTERING for test
            train=False,                       # Use test/clean.npy
            transform=transform_test,
            corrupt_rate=test_corrupt_rate
        )

    # Create DataLoader from dataset
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,                 # 256
        shuffle=shuffle,                       # True
        num_workers=num_workers                # 0
    )
    
    return train_dl, test_dl, ...
```

---

## STEP 2: CIFAR_C CLASS - Dataset/init_dataset.py Lines 138-174

**File:** `/vercel/share/v0-project/Dataset/init_dataset.py`

```python
class CIFAR_C(Dataset):
    """Load CIFAR-10/100 data from .npy files"""
    
    def __init__(self, root: str, dataidxs=None, train=True, transform=None, corrupt_rate=0):
        """
        Args:
            root: './Dataset/cifar_100_c/' or './Dataset/cifar_10_c/'
            dataidxs: [12847, 3456, ...] indices to pick, or None
            train: True for train/clean.npy, False for test/clean.npy
            corrupt_rate: 0 for clean.npy, or 0.5 for random_corrupt_0.5.npy
        """
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.corrupt_rate = corrupt_rate
        
        # ✓ CALL THE MAIN LOADING FUNCTION
        self.data, self.target = self.Construct_Participant_Dataset()  # Line 145

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # THIS IS WHERE .NPY FILES ARE ACTUALLY PICKED FROM DISK!
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    def Construct_Participant_Dataset(self):  # Line 147
        """
        Load .npy files from disk
        """
        root_path = Path(self.root)  # Line 148
        
        # Determine train or test split
        if self.train:                                    # Line 149
            data_path = root_path / "train"              # Line 150
            # Becomes: './Dataset/cifar_100_c/train'
        else:
            data_path = root_path / "test"               # Line 152
            # Becomes: './Dataset/cifar_100_c/test'

        # ✓ LOAD .NPY FILE FROM DISK - THIS IS THE KEY LINE!
        if self.corrupt_rate == 0:                       # Line 154
            data = np.load(data_path / "clean.npy")      # Line 155
            # ACTUAL DISK PATH: './Dataset/cifar_100_c/train/clean.npy'
            # OUTPUT: numpy array shape [50000, 32, 32, 3] or [10000, 32, 32, 3]
        else:
            data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
            # EXAMPLE: './Dataset/cifar_100_c/train/random_corrupt_0.5.npy'
            # OUTPUT: numpy array shape [50000, 32, 32, 3]

        # ✓ LOAD LABELS .NPY FILE FROM DISK
        target = np.load(data_path / "labels.npy")       # Line 158
        # ACTUAL DISK PATH: './Dataset/cifar_100_c/train/labels.npy'
        # OUTPUT: numpy array shape [50000,] or [10000,]

        # ✓ APPLY DATAIDXS FILTERING IF PROVIDED
        if self.dataidxs is not None:                    # Line 160
            data = data[self.dataidxs]                   # Line 161
            # BEFORE: [50000, 32, 32, 3]
            # AFTER:  [5000, 32, 32, 3]  (only selected indices)
            
            target = target[self.dataidxs]               # Line 162
            # BEFORE: [50000,]
            # AFTER:  [5000,]

        return data, target                              # Line 164
        # Returns: (numpy array, numpy array)

    def __len__(self):                                   # Line 166
        return self.data.shape[0]
        # Returns: 5000 or 10000

    def __getitem__(self, index):                        # Line 169
        """Get single sample with transform"""
        img = self.data[index, ...]                      # Line 170
        # Shape: [32, 32, 3]
        
        label = self.target[index]                       # Line 171
        # Shape: scalar (int)
        
        if self.transform:
            img = self.transform(img)                    # Line 172-173
            # Applies: ToTensor, Normalize, etc.
            # OUTPUT: tensor [3, 32, 32]
        
        return img, label                                # Line 174
        # Returns: (tensor [3, 32, 32], int)
```

---

## STEP 3: PYTORCH DATALOADER BATCHING

Once `CIFAR_C` loads all data into `self.data` and `self.target`:

```python
train_dl = DataLoader(
    train_ds,              # CIFAR_C instance with [5000, 32, 32, 3] data
    batch_size=256,
    shuffle=True,
    num_workers=0
)

# DataLoader will:
# 1. Call __getitem__ 256 times per batch
# 2. Stack outputs into batches
# 3. Return:
#    - images: [256, 3, 32, 32] tensor
#    - labels: [256] tensor
```

---

## COMPLETE FLOW DIAGRAM

```
pwhfl.py Line 332
    ↓
get_dataloader(dataset='cifar100', datadir='./Dataset/cifar_100_c/', dataidxs=[5000 indices])
    ↓
Dataset/utils.py Line 307-400
    ├─ Create transform_train pipeline
    ├─ Create CIFAR_C instance (Line 314)
    │   ↓
    │   Dataset/init_dataset.py Line 138-174
    │       ├─ CIFAR_C.__init__()
    │       │   ↓
    │       │   Construct_Participant_Dataset()  (Line 147)
    │       │       ├─ root_path = Path('./Dataset/cifar_100_c/')  (Line 148)
    │       │       ├─ data_path = root_path / "train"  (Line 150)
    │       │       │   # Result: './Dataset/cifar_100_c/train'
    │       │       │
    │       │       ├─ data = np.load(data_path / "clean.npy")  (Line 155)
    │       │       │   # DISK FILE: './Dataset/cifar_100_c/train/clean.npy'
    │       │       │   # LOADED: [50000, 32, 32, 3]
    │       │       │
    │       │       ├─ target = np.load(data_path / "labels.npy")  (Line 158)
    │       │       │   # DISK FILE: './Dataset/cifar_100_c/train/labels.npy'
    │       │       │   # LOADED: [50000,]
    │       │       │
    │       │       └─ Apply filtering: data = data[dataidxs]  (Line 161)
    │       │           # BEFORE: [50000, 32, 32, 3]
    │       │           # AFTER:  [5000, 32, 32, 3]
    │       │
    │       └─ Store in self.data and self.target
    │
    ├─ Create test_ds = CIFAR_C(..., train=False)  (Line 330)
    │   └─ SAME AS ABOVE BUT FOR TEST SET
    │
    └─ Create DataLoader
        ├─ for each batch:
        │   ├─ Select 256 indices
        │   ├─ Call CIFAR_C.__getitem__(idx) for each
        │   │   ├─ img = self.data[idx]  # [32, 32, 3]
        │   │   ├─ label = self.target[idx]  # scalar
        │   │   ├─ img = transform(img)  # [3, 32, 32]
        │   │   └─ return img, label
        │   └─ Stack into batch [256, 3, 32, 32] + [256]
        └─ Yield batch

```

---

## EXACT FILE PICKING MECHANISM

### For PUBLIC DATA (CIFAR-100):

```
DISK: ./Dataset/cifar_100_c/train/clean.npy
      ↓ np.load() [Line 155 in init_dataset.py]
MEMORY: [50000, 32, 32, 3] uint8 numpy array
      ↓ Filter with dataidxs=[12847, 3456, ...] [Line 161 in init_dataset.py]
MEMORY: [5000, 32, 32, 3] uint8 numpy array
      ↓ Store in CIFAR_C.data
CIFAR_C instance ready for DataLoader
```

### For TEST DATA (CIFAR-10):

```
DISK: ./Dataset/cifar_10_c/test/clean.npy
      ↓ np.load() [Line 155 in init_dataset.py]
MEMORY: [10000, 32, 32, 3] uint8 numpy array
      ↓ NO FILTERING (dataidxs=None) [Line 160 checks if None]
MEMORY: [10000, 32, 32, 3] uint8 numpy array (unchanged)
      ↓ Store in CIFAR_C.data
CIFAR_C instance ready for DataLoader
```

---

## KEY LINES SUMMARY

| Action | File | Line | Code |
|--------|------|------|------|
| **Entry** | pwhfl.py | 332 | `pub_dl = get_dataloader(...)` |
| **Function call** | utils.py | 307 | `elif dataset == "cifar100":` |
| **Create dataset** | utils.py | 314 | `train_ds = CIFAR_C(...)` |
| **Load data.npy** | init_dataset.py | 155 | `data = np.load(data_path / "clean.npy")` |
| **Load labels.npy** | init_dataset.py | 158 | `target = np.load(data_path / "labels.npy")` |
| **Apply filtering** | init_dataset.py | 161 | `data = data[self.dataidxs]` |
| **Return data** | init_dataset.py | 164 | `return data, target` |
| **Batch samples** | utils.py | 400+ | `DataLoader creation` |

