# GET_DATALOADER - COMPLETE INTERNAL EXECUTION TRACE

## File Location
**File:** `/vercel/share/v0-project/Dataset/utils.py` (Lines 210-365)

---

## FUNCTION SIGNATURE (Line 200-211)

```python
def get_dataloader(
    dataset='cifar10',
    datadir='./Dataset/cifar_10_c/',
    dataidxs=None,
    train=True,
    batch_size=512,
    transform_train=None,
    transform_test=None,
    corrupt_rate=0,
    test_corrupt_rate=0,
    augmix_module=None
):
```

---

## CASE 1: PUBLIC DATA (CIFAR-100) - For MCE Teacher Targets

**Called from:** `pwhfl.py` Line 332-333

```python
pub_dl = get_dataloader(
    dataset='cifar100',
    datadir='./Dataset/cifar_100_c/',
    dataidxs=pub_idx,                    # [5000 random indices]
    train=True,
    batch_size=256,
    corrupt_rate=0
)
```

### STEP 1: Path Initialization (Line 218)
```python
data_path = Path(datadir)  # → Path('./Dataset/cifar_100_c/')
```

### STEP 2: CIFAR-100 Branch (Line 307-365)

```python
elif dataset == "cifar100":
    # Branch A: augmix_module=None (our case)
    if augmix_module is None:
        # TRANSFORM CREATION (Lines 309-319)
        transform_train = transforms.Compose([
            transforms.ToPILImage(),           # [H,W,C] numpy → PIL
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),              # PIL → [C,H,W] tensor
            transforms.Normalize(
                mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            )
        ])
        
        # DATASET CREATION (Lines 320-326)
        train_ds = CIFAR_C(
            str(data_path),           # './Dataset/cifar_100_c/'
            dataidxs=dataidxs,        # [indices 0-4999 randomly selected from 50K]
            train=True,
            transform=transform_train,
            corrupt_rate=corrupt_rate # 0 (clean)
        )
```

### Inside CIFAR_C.__init__() (Dataset/init_dataset.py Lines 138-145)

```python
class CIFAR_C(Dataset):
    def __init__(self, root: str, dataidxs=None, train=True, 
                 transform=None, corrupt_rate=0):
        self.root = root
        self.dataidxs = dataidxs        # [5000 indices]
        self.train = train              # True
        self.transform = transform      # Compose transformations
        self.corrupt_rate = corrupt_rate # 0
        
        # KEY STEP: Load data from disk
        self.data, self.target = self.Construct_Participant_Dataset()
```

### Inside Construct_Participant_Dataset() (Lines 147-164)

```python
def Construct_Participant_Dataset(self):
    root_path = Path(self.root)  # Path('./Dataset/cifar_100_c/')
    
    if self.train:
        data_path = root_path / "train"  # './Dataset/cifar_100_c/train'
    else:
        data_path = root_path / "test"
    
    # PICK FILE BASED ON CORRUPTION
    if self.corrupt_rate == 0:
        # LOAD CLEAN DATA FROM DISK
        data = np.load(data_path / "clean.npy")
        # File: ./Dataset/cifar_100_c/train/clean.npy
        # Shape: [50000, 32, 32, 3]  ← ALL 50,000 training samples
        # dtype: uint8 (0-255 pixel values)
    else:
        data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
    
    # LOAD LABELS
    target = np.load(data_path / "labels.npy")
    # File: ./Dataset/cifar_100_c/train/labels.npy
    # Shape: [50000]
    # Content: class indices 0-99 (CIFAR-100 has 100 classes)
    
    # CRITICAL: Filter by dataidxs
    if self.dataidxs is not None:
        data = data[self.dataidxs]       # [5000, 32, 32, 3] ← FILTERED
        target = target[self.dataidxs]   # [5000] ← FILTERED
    
    # dataidxs was provided, so we HAVE filtered
    return data, target
```

### STEP 3: Transform Creation (Lines 350-356)

```python
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    )
])
```

### STEP 4: Test Dataset Creation (Lines 357-362)

```python
test_ds = CIFAR_C(
    str(data_path),
    train=False,
    transform=transform_test,
    corrupt_rate=test_corrupt_rate
)
# Not used for public data in pwhfl.py (only train_ds is used)
```

### STEP 5: DataLoader Creation (Lines 400+)

```python
# Exact line will be after the if-elif block
train_dl = DataLoader(
    dataset=train_ds,        # CIFAR_C with 5000 samples
    batch_size=256,
    shuffle=True,
    drop_last=False,
    num_workers=0
)
# Output: 20 batches of [256, 3, 32, 32] normalized tensors
# Batch 0: samples [0:256]
# Batch 1: samples [256:512]
# ...
# Batch 19: samples [4744:5000]

test_dl = DataLoader(dataset=test_ds, batch_size=512, shuffle=False)
# Not used

val_dl = None
return train_dl, test_dl, val_dl, None
```

---

## CASE 2: PRIVATE TEST SET (CIFAR-10) - For Evaluation

**Called from:** `pwhfl.py` Line 360-373

```python
_, test_dl, _, _ = get_dataloader(
    dataset='cifar10',
    datadir='./Dataset/cifar_10_c/',
    dataidxs=None,                  # ← NO FILTERING!
    train=False,                    # ← TEST SET
    test_corrupt_rate=0             # ← CLEAN DATA
)
```

### STEP 1: Path Initialization (Line 218)
```python
data_path = Path(datadir)  # → Path('./Dataset/cifar_10_c/')
```

### STEP 2: CIFAR-10 Branch (Line 221-305)

```python
if dataset == "cifar10":
    if augmix_module is None:
        # TRANSFORM CREATION FOR TRAIN (Lines 223-239)
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.pad(
                Variable(x.unsqueeze(0), requires_grad=False),
                (4, 4, 4, 4),
                mode="reflect"
            ).data.squeeze()),
            # ... other transforms ...
        ])
        # Not used for test_dl (we pass train=False)
        
        # TRANSFORM CREATION FOR TEST (Lines 292-298)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )
        ])
        
        # TEST DATASET CREATION (Lines 299-304)
        test_ds = CIFAR_C(
            str(data_path),           # './Dataset/cifar_10_c/'
            dataidxs=None,            # ← NO FILTERING!
            train=False,              # ← TEST SET
            transform=transform_test,
            corrupt_rate=test_corrupt_rate  # 0 (clean)
        )
```

### Inside CIFAR_C.__init__() for Test Data (Lines 138-145)

```python
class CIFAR_C(Dataset):
    def __init__(self, root: str, dataidxs=None, train=True, 
                 transform=None, corrupt_rate=0):
        self.root = root
        self.dataidxs = dataidxs        # None ← NO FILTERING
        self.train = train              # False ← TEST SET
        self.transform = transform      # test transforms
        self.corrupt_rate = corrupt_rate # 0 ← CLEAN
        
        # KEY STEP: Load data from disk
        self.data, self.target = self.Construct_Participant_Dataset()
```

### Inside Construct_Participant_Dataset() for Test (Lines 147-164)

```python
def Construct_Participant_Dataset(self):
    root_path = Path(self.root)  # Path('./Dataset/cifar_10_c/')
    
    if self.train:
        data_path = root_path / "train"
    else:
        data_path = root_path / "test"  # ← THIS BRANCH (test set)
    
    # PICK FILE BASED ON CORRUPTION
    if self.corrupt_rate == 0:
        # LOAD CLEAN TEST DATA FROM DISK
        data = np.load(data_path / "clean.npy")
        # File: ./Dataset/cifar_10_c/test/clean.npy
        # Shape: [10000, 32, 32, 3]  ← ALL 10,000 test samples
        # dtype: uint8 (0-255 pixel values)
    else:
        data = np.load(data_path / f"random_corrupt_{self.corrupt_rate}.npy")
    
    # LOAD LABELS
    target = np.load(data_path / "labels.npy")
    # File: ./Dataset/cifar_10_c/test/labels.npy
    # Shape: [10000]
    # Content: class indices 0-9 (CIFAR-10 has 10 classes)
    
    # CRITICAL: Filter by dataidxs
    if self.dataidxs is not None:
        data = data[self.dataidxs]       # Would filter, but dataidxs=None
        target = target[self.dataidxs]
    
    # dataidxs is None, so NO FILTERING
    # Return ALL 10,000 samples
    return data, target
```

### STEP 3: DataLoader Creation (Lines 400+)

```python
train_dl = DataLoader(...)  # Not returned (train=False)

test_dl = DataLoader(
    dataset=test_ds,        # CIFAR_C with ALL 10000 samples
    batch_size=512,
    shuffle=False,
    drop_last=False,
    num_workers=0
)
# Output: 20 batches of [512, 3, 32, 32] normalized tensors
# Batch 0: samples [0:512]
# Batch 1: samples [512:1024]
# ...
# Batch 19: samples [9488:10000]  ← 512 samples (10000 % 512 = 488, but batch_size=512)
# Actually: Last batch has fewer samples

val_dl = None
return _, test_dl, _, _
```

---

## STEP-BY-STEP EXECUTION DIAGRAM

### PUBLIC DATA FLOW (CIFAR-100)
```
get_dataloader(dataset='cifar100', dataidxs=[5000 random indices], train=True)
    ↓
    data_path = Path('./Dataset/cifar_100_c/')
    ↓
    CIFAR_C('cifar_100_c/', dataidxs=[5000 indices], train=True, corrupt_rate=0)
        ↓
        Construct_Participant_Dataset()
            ↓
            root_path = Path('./Dataset/cifar_100_c/')
            data_path = root_path / "train"
            ↓
            data = np.load('./Dataset/cifar_100_c/train/clean.npy')
                └─ Shape: [50000, 32, 32, 3]  ← ALL 50K
            ↓
            target = np.load('./Dataset/cifar_100_c/train/labels.npy')
                └─ Shape: [50000]
            ↓
            data = data[[5000 indices]]  ← FILTER HERE
            target = target[[5000 indices]]
                └─ Shape: [5000, 32, 32, 3] and [5000]
            ↓
            return data, target
        ↓
        self.data = [5000, 32, 32, 3]
        self.target = [5000]
    ↓
    transform_train applied per sample during DataLoader iteration
    ↓
    DataLoader(train_ds, batch_size=256, shuffle=True)
        └─ 20 batches × [256, 3, 32, 32] + [256] labels
```

### TEST DATA FLOW (CIFAR-10)
```
get_dataloader(dataset='cifar10', dataidxs=None, train=False, test_corrupt_rate=0)
    ↓
    data_path = Path('./Dataset/cifar_10_c/')
    ↓
    CIFAR_C('cifar_10_c/', dataidxs=None, train=False, corrupt_rate=0)
        ↓
        Construct_Participant_Dataset()
            ↓
            root_path = Path('./Dataset/cifar_10_c/')
            data_path = root_path / "test"  ← TEST SET
            ↓
            data = np.load('./Dataset/cifar_10_c/test/clean.npy')
                └─ Shape: [10000, 32, 32, 3]  ← ALL 10K
            ↓
            target = np.load('./Dataset/cifar_10_c/test/labels.npy')
                └─ Shape: [10000]
            ↓
            # dataidxs=None, NO FILTERING
                └─ Shape: [10000, 32, 32, 3] and [10000]
            ↓
            return data, target
        ↓
        self.data = [10000, 32, 32, 3]
        self.target = [10000]
    ↓
    transform_test applied per sample during DataLoader iteration
    ↓
    DataLoader(test_ds, batch_size=512, shuffle=False)
        └─ 20 batches × [512, 3, 32, 32] + [512] labels
```

---

## KEY CLASSES/FUNCTIONS CALLED

| What | Where | Lines | What It Does |
|------|-------|-------|-------------|
| `Path()` | pathlib | - | Convert string to Path object |
| `CIFAR_C` | Dataset/init_dataset.py | 138-174 | Loads numpy arrays from disk |
| `Construct_Participant_Dataset()` | Dataset/init_dataset.py | 147-164 | Loads .npy files, applies index filtering |
| `np.load()` | numpy | - | Loads .npy file from disk to numpy array |
| `data[indices]` | numpy | - | Fancy indexing to filter samples |
| `transforms.Compose()` | torchvision | - | Chains multiple transformations |
| `transforms.ToTensor()` | torchvision | - | [H,W,C] numpy → [C,H,W] tensor [0,1] |
| `transforms.Normalize()` | torchvision | - | (x - mean) / std per channel |
| `transforms.ToPILImage()` | torchvision | - | [C,H,W] tensor → PIL Image |
| `transforms.RandomCrop()` | torchvision | - | Randomly crop and pad |
| `transforms.RandomHorizontalFlip()` | torchvision | - | 50% probability flip |
| `transforms.RandomRotation()` | torchvision | - | Random rotation 0-15° |
| `DataLoader` | torch.utils.data | - | Batches dataset, creates Iterator |

---

## DATA SHAPES AT EACH STAGE

### PUBLIC (CIFAR-100) Public Indexing

| Stage | Shape | dtype | Device | Notes |
|-------|-------|-------|--------|-------|
| Disk (clean.npy) | [50000, 32, 32, 3] | uint8 | CPU/Disk | Raw pixels 0-255 |
| After load | [50000, 32, 32, 3] | uint8 | CPU RAM | Loaded into memory |
| After filtering | [5000, 32, 32, 3] | uint8 | CPU RAM | 5000 indices selected |
| After ToPILImage | PIL Image (H=32, W=32) | - | CPU | Per-sample in DataLoader |
| After RandomCrop | PIL Image (H=32, W=32) | - | CPU | Per-sample |
| After ToTensor | [3, 32, 32] | float32 | CPU | Values [0, 1] |
| After Normalize | [3, 32, 32] | float32 | CPU | Normalized per channel |
| DataLoader batch | [256, 3, 32, 32] | float32 | CPU | 256 samples stacked |
| On GPU | [256, 3, 32, 32] | float32 | GPU | Transferred for forward pass |

### TEST (CIFAR-10) No Indexing

| Stage | Shape | dtype | Device | Notes |
|-------|-------|-------|--------|-------|
| Disk (clean.npy) | [10000, 32, 32, 3] | uint8 | CPU/Disk | Raw pixels 0-255 |
| After load | [10000, 32, 32, 3] | uint8 | CPU RAM | Loaded into memory |
| After filtering | [10000, 32, 32, 3] | uint8 | CPU RAM | NO FILTERING |
| After ToTensor | [3, 32, 32] | float32 | CPU | Values [0, 1] |
| After Normalize | [3, 32, 32] | float32 | CPU | Normalized per channel |
| DataLoader batch | [512, 3, 32, 32] | float32 | CPU | 512 samples stacked |
| On GPU | [512, 3, 32, 32] | float32 | GPU | Transferred for forward pass |

---

## SUMMARY: TWO DIFFERENT DATA SOURCES

```
PUBLIC DATA (CIFAR-100):
├─ Source: ./Dataset/cifar_100_c/train/clean.npy [50000, 32, 32, 3]
├─ Index Filter: YES → select 5000 random indices
├─ Final: [5000, 32, 32, 3]
├─ Use: MCE teacher target generation (20 batches × 256)
└─ File Called: Dataset/init_dataset.py CIFAR_C class

TEST DATA (CIFAR-10):
├─ Source: ./Dataset/cifar_10_c/test/clean.npy [10000, 32, 32, 3]
├─ Index Filter: NO → use all 10000
├─ Final: [10000, 32, 32, 3]
├─ Use: Evaluation/accuracy (forward pass only, 20 batches × 512)
└─ File Called: Dataset/init_dataset.py CIFAR_C class
```
