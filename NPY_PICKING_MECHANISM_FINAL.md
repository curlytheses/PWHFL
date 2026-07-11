# .NPY FILE PICKING MECHANISM - COMPLETE WITH ACTUAL CODE

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 1: PUBLIC DATA (CIFAR-100) - MCE Teacher Target Generation
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### ENTRY POINT: pwhfl.py (Lines 328-333)

```python
pub_idx = generate_public_data_indexs('cifar100', './Dataset/cifar_100_c/', 5000)
# Returns: [12847, 3456, 49821, 10285, ..., 38947] — 5000 random unique indices

pub_dl = get_dataloader(
    dataset='cifar100',                              # ← Dataset type
    datadir='./Dataset/cifar_100_c/',                # ← Physical disk location
    dataidxs=pub_idx,                                # ← [5000 random indices]
    train=True,                                      # ← Use train/ folder
    batch_size=256
)
```

---

### FUNCTION 1: get_dataloader() → Dataset/utils.py (Lines 210-400)

**Entry to this function:**
- `dataset='cifar100'`
- `datadir='./Dataset/cifar_100_c/'`
- `dataidxs=[12847, 3456, ..., 5000 items]`
- `train=True`
- `batch_size=256`

**Execution (Lines 307-326):**
```python
elif dataset == "cifar100":                 # Line 307 — Check dataset type
    if augmix_module is None:               # Line 308
        # ✓ CREATE TRANSFORM PIPELINE
        transform_train = transforms.Compose([  # Line 309-319
            transforms.ToPILImage(),            # PIL Image input
            transforms.RandomCrop(32, padding=4),  # Random crop
            transforms.RandomHorizontalFlip(),   # 50% horizontal flip
            transforms.RandomRotation(15),       # Rotate ±15°
            transforms.ToTensor(),               # [H,W,C] → [C,H,W]
            transforms.Normalize(                # Normalize with CIFAR-100 stats
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            )
        ])
        
        # ✓ CALL CIFAR_C CLASS TO LOAD .NPY FILES
        train_ds = CIFAR_C(                 # Line 320 ← THIS IS WHERE NPY IS LOADED
            str(data_path),                 # './Dataset/cifar_100_c/'
            dataidxs=dataidxs,              # [12847, 3456, ..., 5000 items]
            train=True,                     # Use train/clean.npy
            transform=transform_train,      # Apply transforms
            corrupt_rate=corrupt_rate       # 0 (use clean.npy)
        )
```

**Flow continues (Line 357-362):**
```python
        # Also create test dataset for later
        test_ds = CIFAR_C(
            str(data_path),                 # './Dataset/cifar_100_c/'
            train=False,                    # Use test/clean.npy
            transform=transform_test,
            corrupt_rate=test_corrupt_rate  # 0 (use clean.npy)
        )
```

---

### FUNCTION 2: CIFAR_C.__init__() → Dataset/init_dataset.py (Lines 138-145)

**When called:**
```python
train_ds = CIFAR_C(
    root='./Dataset/cifar_100_c/',
    dataidxs=[12847, 3456, ..., 5000 items],
    train=True,
    transform=<transform pipeline>,
    corrupt_rate=0
)
```

**Execution:**
```python
class CIFAR_C(Dataset):
    def __init__(self, root: str, dataidxs=None, train=True, transform=None, corrupt_rate=0):
        # Line 140-144: Store parameters
        self.root = root                                    # './Dataset/cifar_100_c/'
        self.dataidxs = dataidxs                            # [12847, 3456, ..., 5000 items]
        self.train = train                                  # True
        self.transform = transform                          # <transform pipeline>
        self.corrupt_rate = corrupt_rate                    # 0
        
        # Line 145: ✓ CALL THE NPY LOADING FUNCTION
        self.data, self.target = self.Construct_Participant_Dataset()
```

---

### FUNCTION 3: Construct_Participant_Dataset() → Dataset/init_dataset.py (Lines 147-164)

**This is WHERE .NPY FILES ARE PHYSICALLY LOADED FROM DISK:**

```python
def Construct_Participant_Dataset(self):
    # Line 148: Create Path object
    root_path = Path(self.root)
    # root_path = PosixPath('./Dataset/cifar_100_c/')

    # Line 149-152: Determine if train or test
    if self.train:
        data_path = root_path / "train"
        # data_path = PosixPath('./Dataset/cifar_100_c/train')
    else:
        data_path = root_path / "test"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LINE 154-157: ✓ LOAD DATA FROM .NPY FILE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if self.corrupt_rate == 0:                              # Line 154
        # data_path = PosixPath('./Dataset/cifar_100_c/train')
        # Construct full path: './Dataset/cifar_100_c/train/clean.npy'
        data = np.load(data_path / "clean.npy")             # Line 155
        
        # ✓ np.load() ACTUALLY READS FROM DISK HERE!
        # Returns: numpy array with dtype=uint8, shape=[50000, 32, 32, 3]
        # Each pixel value: 0-255
        # Total memory: 50000 × 32 × 32 × 3 = 153,600,000 bytes ≈ 146 MB
        
    else:
        # Alternative path (not used in this case, but shown for completeness)
        data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
        # Example: './Dataset/cifar_100_c/train/random_corrupt_0.5.npy'
        # Would return [50000, 32, 32, 3] with corrupted images

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LINE 158: ✓ LOAD LABELS FROM .NPY FILE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    target = np.load(data_path / "labels.npy")              # Line 158
    
    # ✓ np.load() ACTUALLY READS FROM DISK HERE!
    # Full path: './Dataset/cifar_100_c/train/labels.npy'
    # Returns: numpy array with dtype=int64, shape=[50000,]
    # Each label: 0-99 (100 classes for CIFAR-100)
    # Total memory: 50000 × 8 bytes = 400 KB

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LINE 160-162: ✓ FILTER INDICES (DATA PICKING)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if self.dataidxs is not None:                           # Line 160
        # self.dataidxs = [12847, 3456, 49821, ..., 38947] (5000 unique indices)
        
        data = data[self.dataidxs]                          # Line 161
        # BEFORE: [50000, 32, 32, 3]
        # np.load returns full 50K samples
        # NOW: Select only the 5000 samples at those indices
        # AFTER: [5000, 32, 32, 3]
        # Example: data[[12847, 3456, 49821, ...]] selects specific samples
        
        target = target[self.dataidxs]                      # Line 162
        # BEFORE: [50000,]
        # AFTER: [5000,]
        # Example: labels[[12847, 3456, 49821, ...]]

    return data, target                                     # Line 164
    # Returns two numpy arrays:
    # data:   [5000, 32, 32, 3] uint8 - The actual image data
    # target: [5000,] int64 - The class labels (0-99)
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## PART 2: TEST DATA (CIFAR-10) - Evaluation
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### ENTRY POINT: pwhfl.py (Lines 360-373)

```python
_, test_dl, _, _ = get_dataloader(
    dataset='cifar10',                              # ← Different dataset!
    datadir='./Dataset/cifar_10_c/',                # ← Different location!
    dataidxs=None,                                  # ← NO FILTERING (important!)
    train=False,                                    # ← Use test/ folder
    test_corrupt_rate=0,
    batch_size=512
)
```

---

### SAME FLOW BUT FOR TEST:

**get_dataloader() → Dataset/utils.py (Lines 221-305):**
```python
if dataset == "cifar10":                    # Line 221 — Different branch!
    if augmix_module is None:
        transform_train = transforms.Compose([...])  # Not used for test
        
        train_ds = CIFAR_C(
            str(data_path),                 # './Dataset/cifar_10_c/'
            dataidxs=dataidxs,              # Some indices (for private training)
            train=True,
            transform=transform_train,
            corrupt_rate=corrupt_rate
        )

    # ✓ THIS IS CALLED FOR EVALUATION
    transform_test = transforms.Compose([   # Line 292-298
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        )
    ])
    
    test_ds = CIFAR_C(                       # Line 299 ← LOAD TEST NPY
        str(data_path),                      # './Dataset/cifar_10_c/'
        train=False,                         # ← Use test/ folder
        transform=transform_test,
        corrupt_rate=test_corrupt_rate       # 0
    )
```

**CIFAR_C class execution (SAME as above but with different parameters):**
```python
class CIFAR_C(Dataset):
    def __init__(self, root='./Dataset/cifar_10_c/', dataidxs=None, train=False, ...):
        self.root = root                                # './Dataset/cifar_10_c/'
        self.dataidxs = dataidxs                        # None (KEY DIFFERENCE!)
        self.train = train                              # False
        self.data, self.target = self.Construct_Participant_Dataset()

def Construct_Participant_Dataset(self):
    root_path = Path(self.root)                         # './Dataset/cifar_10_c/'
    
    if self.train:                                      # False
        data_path = root_path / "train"
    else:
        data_path = root_path / "test"                  # ← USE TEST FOLDER
        # data_path = './Dataset/cifar_10_c/test'

    if self.corrupt_rate == 0:                          # 0 = True
        data = np.load(data_path / "clean.npy")         # ← LOAD TEST DATA
        # Full path: './Dataset/cifar_10_c/test/clean.npy'
        # Returns: [10000, 32, 32, 3] uint8
        
    target = np.load(data_path / "labels.npy")          # ← LOAD TEST LABELS
        # Full path: './Dataset/cifar_10_c/test/labels.npy'
        # Returns: [10000,] int64 (10 classes: 0-9)

    if self.dataidxs is not None:                       # None ← SKIP FILTERING!
        # This condition is False, so NO filtering applied
        data = data[self.dataidxs]  # NOT executed
        target = target[self.dataidxs]  # NOT executed

    return data, target
    # Returns:
    # data:   [10000, 32, 32, 3] uint8 - ALL test samples
    # target: [10000,] int64 - ALL test labels
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## COMPARISON: .NPY PICKING DIFFERENCE
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### PUBLIC DATA (CIFAR-100):
```
Disk File:           ./Dataset/cifar_100_c/train/clean.npy
np.load() OUTPUT:    [50000, 32, 32, 3]
dataidxs:            [12847, 3456, 49821, ..., 38947]  ← 5000 indices
After filtering:     [5000, 32, 32, 3]  ← SELECTED
Memory to GPU:       5000 samples per batch
Used for:            MCE teacher target generation
```

### TEST DATA (CIFAR-10):
```
Disk File:           ./Dataset/cifar_10_c/test/clean.npy
np.load() OUTPUT:    [10000, 32, 32, 3]
dataidxs:            None  ← NO FILTERING
After filtering:     [10000, 32, 32, 3]  ← ALL USED
Memory to GPU:       All 10000 samples (in batches of 512)
Used for:            Evaluation / accuracy forward pass
```

---

## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## KEY INSIGHT: THE PICKING HAPPENS IN 2 PLACES
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Place 1: generate_public_data_indexs() - RANDOM INDEX SELECTION**
- File: Dataset/utils.py Line 190
- Selects WHICH 5000 indices from 50K CIFAR-100 samples
- Uses: `np.random.choice([0..49999], 5000, replace=False)`

**Place 2: Construct_Participant_Dataset() - INDEX FILTERING**
- File: Dataset/init_dataset.py Line 161
- Filters the loaded array by those indices
- Uses: `data = data[self.dataidxs]`

Together: `[50K] → [select 5K indices] → [filter array] → [5K array]`

