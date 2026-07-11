# PWHFL (Personalized Weighted Hierarchical Federated Learning) - Complete Execution Flow

## System Overview
PWHFL is a federated learning framework with **heterogeneous models** across multiple clients. It has two main phases:
1. **Pretraining Phase** (Network/pretrain.py) - Individual model training
2. **Collaborative Phase** (HHF/pwhfl.py) - Federated learning with adaptive weighting

---

## PHASE 1: PRETRAINING (Network/pretrain.py)

### Entry Point Configuration
```
Seed = 7
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
```

### Step 1.1: Initialize Logger & Seeds
**Code Location:** Lines 150-165
```python
logger = init_logs()  # Creates Logs/ directory with pwhfl.log
random.seed(7)
np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
```
**Events:**
- Logger created → writes to `RAHFL/Logs/pretrain.log`
- CUDA Device Order set to GPU 0
- Reproducibility: All randomness fixed with seed=7

### Step 1.2: Data Partition Selection
**Code Location:** Lines 173-186
```
If Data_Partition == "noniid":
  └─ Call partition_data() with Dirichlet distribution
     └─ Returns net_dataidx_map: {client_id: [indices of their samples]}
     
Else (iid):
  └─ Randomly permute 50,000 samples → take first 10,000
  └─ Each of 4 clients gets random 10,000 indices
```
**DATA PICKING FLOW (IID):**
```
50,000 training samples in CIFAR-10
         ↓
    Random Permutation
         ↓
Split equally among 4 participants:
  - Client 0 (ResNet10): indices [1000, 2340, 5670, ...]
  - Client 1 (ResNet12): indices [5400, 3210, 8900, ...]
  - Client 2 (ShuffleNet): indices [2100, 4560, 9234, ...]
  - Client 3 (MobileNetV2): indices [3300, 6780, 1234, ...]
```

### Step 1.3: Model Initialization
**Code Location:** Lines 188-197
```python
net_list = init_nets(
    n_parties=4,
    nets_name_list=['ResNet10', 'ResNet12', 'ShuffleNet', 'Mobilenetv2'],
    num_classes=10
)
```
**MODEL PICKING FLOW:**
```
Heterogeneous Model Selection:
┌─────────────────────────────────────────────────────┐
│ PARTICIPANT 0    │ PARTICIPANT 1    │ PARTICIPANT 2 │ PARTICIPANT 3 │
│  ResNet10        │  ResNet12        │  ShuffleNet   │  MobileNetV2  │
│ (10 layers)      │ (12 layers)      │ (efficient)   │ (lightweight) │
└─────────────────────────────────────────────────────┘

Each model:
- Initialized with random weights
- Wrapped with nn.DataParallel(device_ids=[0])
- Moved to CUDA device 0
```

### Step 1.4: Data Loading
**Code Location:** Lines 199-212 (per participant)
```python
for index in range(N_Participants):  # 0, 1, 2, 3
    train_dl, _, train_ds, _ = get_dataloader(
        dataset='cifar10',
        dataidxs=net_dataidx_map[index],  # Client's assigned indices
        corrupt_rate=0.5,  # 50% label corruption
        augmix_module=None
    )
```
**DATA LOADING FLOW FOR EACH CLIENT:**
```
Client 0 Example:
────────────────────────────────────────
  indices = [1000, 2340, 5670, ...]  (10,000 indices)
         ↓
  Load from CIFAR-10-C dataset at corruption rate 0.5
         ↓
  Apply transforms.ToTensor()
         ↓
  Create DataLoader with batch_size=256
         ↓
  ~39 batches (10000 / 256)
```

### Step 1.5: Pretraining Loop (Per Participant)
**Code Location:** Lines 44-73 (`pretrain_network()` function)

**For each of 4 participants, repeat:**

#### Inner Loop Structure:
```
For epoch in range(40):  # 40 pretraining epochs
    For batch_idx, (images, labels) in train_dl:
        
        FORWARD PASS:
        ├─ images: shape [256, 3, 32, 32]
        ├─ labels: shape [256]
        └─ outputs, _ = network(images)  # [256, 10]
        
        LOSS CALCULATION:
        ├─ criterion = CrossEntropyLoss()
        ├─ loss = criterion(outputs, labels)
        └─ loss value: scalar tensor
        
        BACKWARD PASS:
        ├─ optimizer.zero_grad()
        ├─ loss.backward()  # Compute gradients
        └─ optimizer.step()  # Update weights
        
        LOGGING:
        └─ Log loss value & progress
```

**Example Timeline for Client 0 (ResNet10):**
```
Epoch 1/40:
  Batch 1: loss=2.3456 → optimizer updates ResNet10 weights
  Batch 2: loss=2.1234 → optimizer updates ResNet10 weights
  ...
  Batch 39: loss=1.4567 → optimizer updates ResNet10 weights
  
Epoch 2/40:
  Batch 1: loss=2.0123 → further weight updates
  ...
  
Epoch 40/40:
  Batch 39: loss=0.8432 → final weight updates
```

### Step 1.6: Model Checkpointing
**Code Location:** Lines 214-217
```python
ckpt_path = Model_Save_Dir / f"{Nets_Name_List[index]}_{index}.ckpt"
torch.save(network.state_dict(), str(ckpt_path))
```
**Files Created:**
```
Network/Model_Storage_hetero_4client/cifar10_iid_0/random_corrupt_0.5/
├─ ResNet10_0.ckpt      (weights from pretraining)
├─ ResNet12_1.ckpt
├─ ShuffleNet_2.ckpt
└─ MobileNetV2_3.ckpt
```

### Step 1.7: Final Evaluation
**Code Location:** Lines 222-248
```python
for index in range(N_Participants):
    network = load trained model
    test_dl = get_dataloader for test set
    accuracy = evaluate_network(network, test_dl)
```

**Final Accuracies Logged:**
```
Client 0 (ResNet10): 85.3%
Client 1 (ResNet12): 87.1%
Client 2 (ShuffleNet): 82.5%
Client 3 (MobileNetV2): 80.9%

Average Accuracy: 83.95%
```

---

## PHASE 2: COLLABORATIVE LEARNING (HHF/pwhfl.py)

### Setup & Configuration
**Code Location:** Lines 23-81

#### Parameters Overview:
```
Seed = 0
TrainBatchSize = 256
TestBatchSize = 512
CommunicationEpoch = 40        ← Number of federated rounds

CORRUPTION RATES (each drives different modules):
├─ Private_Corrupt_Rate = 0.5   → Controls ADA-JSD λ weight
├─ Test_Corrupt_Rate = 0.0      → Controls PWKT τ, ARC scale
└─ Public_Corrupt_Rate = 0.5    → Controls MCE crops, teacher temperature

HYPERPARAMETERS (derived from corruption rates):
├─ TEACHER_TEMP = 0.65          [from Public_Corrupt_Rate]
├─ LABEL_SMOOTH = 0.05          [from Private_Corrupt_Rate]
├─ MCE_N_CROPS = 2              [from Public_Corrupt_Rate]
├─ JSD_L_TARGET = 1.0           [from Private_Corrupt_Rate]
├─ PWKT_TAU = 20.0              [from Test_Corrupt_Rate]
└─ ARC_SCALE = 1.0              [from Test_Corrupt_Rate]
```

### Step 2.1: Load Pretrained Models
**Code Location:** Lines 318-329
```python
net_list = init_nets(...)  # Create 4 empty models
for i in range(4):
    network = nn.DataParallel(net_list[i], device_ids=[0]).to(device)
    ckpt_path = Model_Load_Dir / f"{Nets_Name_List[i]}_{i}.ckpt"
    network.load_state_dict(torch.load(str(ckpt_path)))
    logger.info(f"Client {i}: loaded")
```

### Step 2.2: Load Public Dataset (Shared Knowledge)
**Code Location:** Lines 332-339
```python
pub_idx = generate_public_data_indexs(
    'cifar100', Public_Dataset_Dir_Org, Public_Dataset_Length=5000
)
# Randomly selects 5000 indices from CIFAR-100 training set

public_train_dl, _, public_train_ds, _ = get_dataloader(
    dataset='cifar100',
    dataidxs=pub_idx,
    corrupt_rate=0.5,  # Apply corruption to public data
)
```

### Step 2.3: Initialize Optimizers (Persistent)
**Code Location:** Lines 341-345
```python
collab_opts = [
    optim.Adam(net_list[i].parameters(), lr=0.001)
    for i in range(4)
]
# ← These optimizers persist across all communication rounds!
# Each client has its own optimizer with independent momentum buffers
```

### Step 2.4: Initialize Tracking Variables
**Code Location:** Lines 348-352
```python
acc_history = [0.0, 0.0, 0.0, 0.0]     # B5: max accuracy ever seen
acc_prev = [0.0, 0.0, 0.0, 0.0]        # Previous round accuracies
acc_current = [0.0, 0.0, 0.0, 0.0]     # Current round accuracies
```

---

## COMMUNICATION ROUND LOOP (Lines 357-end)

### For each communication epoch (0 to 39):

---

## ⚙️ SUB-PHASE 2A: EVALUATION (Lines 360-396)

### Event Sequence:

**Step 2A.1: Evaluate Each Client on Test Set**
```python
for p in range(4):  # For each participant
    network = net_list[p]
    _, test_dl, _, _ = get_dataloader(...)
    accuracy = evaluate_network(network, test_dl)
    acc_epoch_list.append(accuracy)
```

**Evaluation Details for One Client:**
```
Client 0 (ResNet10) Evaluation:
────────────────────────────────
  network.eval()  → Set to evaluation mode
  
  For each test batch:
    images: shape [512, 3, 32, 32]
    labels: shape [512]
    
    outputs, _ = network(images)  → Forward pass
    predictions = argmax(outputs)
    correct += (predictions == labels).sum()
    
  accuracy = correct / total_samples * 100
  
  RESULT: 85.2%
```

**Step 2A.2: Record Accuracies**
```
Epoch 0 accuracies: [85.2, 87.0, 82.4, 80.8]
Average: 83.85%
```

**Step 2A.3: Update History for Matrix Decisions**
```python
if epoch_index % Matrix_Update_Epoch == 0:  # Every 1 epoch
    acc_history[p] = max(acc_history[p], accuracy)
    acc_current[p] = accuracy
```

**Step 2A.4: Detect Struggling Clients (ARC Module)**
```
arc_lambda(acc_current, acc_prev):
    
For each client p:
  delta = acc_current[p] - acc_prev[p]
  
  IF delta >= 0:  (improving or stable)
    lambda_p = 0.0  (no rescue needed)
  ELSE:  (declining)
    lambda_p = |delta| * ARC_SCALE
    
Example:
  Client 2: acc_current=82.4, acc_prev=83.0, delta=-0.6
  lambda_2 = 0.6 * 1.0 = 0.6  ← NEEDS RESCUE!
  
  Client 0: acc_current=85.2, acc_prev=84.8, delta=0.4
  lambda_0 = 0.0  ← No rescue
```

**Identification of Best Client:**
```python
best_client = argmax(acc_current)  # Index of highest accuracy

Example in Epoch 0:
  acc_current = [85.2, 87.0, 82.4, 80.8]
  best_client = 1  (ResNet12 @ 87.0%)
```

**Output Log:**
```
===== Comm Epoch 0 =====
Avg: 83.85% | Clients: [85.2, 87.0, 82.4, 80.8]
  ARC: Client 2 declining Δ=-0.6% → λ=0.600
  Best client for rescue: 1 (ResNet12 @ 87.0%)
```

---

## ⚙️ SUB-PHASE 2B: COLLABORATIVE LEARNING (Lines 398-end)

### The Public Data Batch Loop:

**Step 2B.1: Iterate Through Public Data**
```python
for batch_idx, (images, _) in enumerate(public_train_dl):
    # images: shape [256, 3, 32, 32] (CIFAR-100 samples)
    # public_train_dl has ~20 batches (5000 / 256)
```

**Step 2B.2: Generate Teacher Targets (MCE - Multi-Crop Ensemble)**

For **EACH participant p** (0, 1, 2, 3):

#### Step 2B.2a: MCE Teacher Target Generation
```
Function: mce_teacher_target(network, images, MCE_N_CROPS=2, TEACHER_TEMP=0.65)

Input images: [256, 3, 32, 32]

CREATE AUGMENTED VIEWS:
├─ View 0: Original image [256, 3, 32, 32]
└─ View 1: Horizontal flip [256, 3, 32, 32]
   (Since MCE_N_CROPS=2)

FOR EACH VIEW:
  ├─ network.eval()  → Teacher in eval mode
  ├─ logits = network(view)  → [256, 10]
  ├─ p = softmax(logits / TEACHER_TEMP)
  │  (temperature 0.65 sharpens probabilities)
  └─ probs_sum += p

AVERAGE ACROSS VIEWS:
  target = probs_sum / 2  → [256, 10], detached (no grad)
  
Result for Client p:
  target_list[p] = [256, 10] probability distribution
```

**Concrete Example:**
```
Image batch: [256, 3, 32, 32] from public data

View 0 (Original):
  network(images) → logits [256, 10]
  softmax(logits/0.65) → probabilities [256, 10]
  Example for 1 sample: [0.15, 0.08, 0.05, 0.12, 0.08, 0.20, 0.08, 0.10, 0.07, 0.07]

View 1 (H-flip):
  network(h_flip) → logits [256, 10]
  softmax(logits/0.65) → probabilities [256, 10]
  Example for 1 sample: [0.16, 0.07, 0.06, 0.11, 0.09, 0.19, 0.09, 0.10, 0.08, 0.05]

Average (target):
  (prob_view0 + prob_view1) / 2
  [0.155, 0.075, 0.055, 0.115, 0.085, 0.195, 0.085, 0.100, 0.075, 0.060]
```

#### Repeat for all 4 participants:
```
target_list[0]: Teacher target from ResNet10
target_list[1]: Teacher target from ResNet12
target_list[2]: Teacher target from ShuffleNet
target_list[3]: Teacher target from MobileNetV2

All shape [256, 10], detached (gradients not tracked)
```

**Step 2B.3: Student Forward Pass & Log-softmax**

For **EACH participant p**:
```python
network = net_list[p]
network.train()  # Set to training mode
logits, _ = network(images)  # [256, 10], requires grad
logsoft_list[p] = F.log_softmax(logits, dim=1)  # [256, 10]
```

Now we have:
```
target_list[p]:    Detached teacher targets [256, 10]
logsoft_list[p]:   Student log-probabilities [256, 10] (requires grad)
```

---

## ⚙️ SUB-PHASE 2C: KNOWLEDGE DISTILLATION WITH ADAPTIVE WEIGHTING

### Step 2C.1: For Each Student Client p (0, 1, 2, 3):

```python
for p in range(4):
    network = net_list[p]
    network.train()
    
    loss = 0.0
    n_teach = 0.0  # Count of active teachers
```

### Step 2C.2: Calculate PWKT Weights (Personalized Weighting via Knowledge Transfer)

**Function:** `pwkt_weights(student_acc, all_acc, student_idx, tau)`
```
Example for Client 0 as student:
─────────────────────────────────
  student_acc = acc_history[0] = 85.2%
  all_acc = [85.2, 87.0, 82.4, 80.8]  (from Matrix_Update_Epoch)
  tau = PWKT_TAU = 20.0
  
  For each teacher j:
    IF j == student (0):
      weight[0] = 0.0  ← Self-distillation excluded
      
    ELSE IF all_acc[j] <= student_acc:
      weight[j] = 0.0  ← Only learn from better models
      
    ELSE:  (j is better than student)
      gap = all_acc[j] - student_acc
      weight[j] = exp((all_acc[j] - student_acc) / tau)
      
  Example calculation:
    j=0 (self): weight=0.0
    j=1 (87.0% vs 85.2%): gap=1.8, weight=exp(1.8/20)=exp(0.09)=1.094
    j=2 (82.4% vs 85.2%): gap=-2.8 (worse), weight=0.0
    j=3 (80.8% vs 85.2%): gap=-4.4 (worse), weight=0.0
    
  Normalization:
    sum = 1.094
    normalized_weights = [0.0, 1.0, 0.0, 0.0]  ← 100% from ResNet12!
```

### Step 2C.3: Knowledge Distillation Loss with PWKT Weights

**KL Divergence Loss:**
```python
kl = nn.KLDivLoss(reduction='batchmean')

for j in range(4):  # For each teacher
    if j == p or weights[j] < 1e-8:
        continue  # Skip self and zero-weight teachers
    
    # target_list[j]: Teacher's target distribution [256, 10]
    # logsoft_list[p]: Student's log-probabilities [256, 10]
    loss += weights[j] * kl(logsoft_list[p], target_list[j])
    n_teach += weights[j]
```

**Example for Client 0 as Student:**
```
weights = [0.0, 1.0, 0.0, 0.0]

Only Teacher 1 (ResNet12) contributes:
  kl_batch = KL(logsoft[0], target[1])
  loss_student_0 = 1.0 * kl_batch
  n_teach = 1.0
```

### Step 2C.4: ARC Rescue (Accuracy Rescue Correction) for Struggling Clients

```python
if Col_Module == 'apex' and rescue_lambdas[p] > 0 and best_client != p:
    rescue = rescue_lambdas[p] * kl(logsoft_list[p], target_list[best_client])
    loss += rescue
```

**Example: Client 2 is Struggling**
```
rescue_lambdas[2] = 0.6  (calculated from ARC earlier)
best_client = 1

IF client 2 is not best:
  Additional loss = 0.6 * KL(logsoft[2], target[1])
  This forces Client 2 to learn from best student (Client 1)
```

### Step 2C.5: Backward Pass & Weight Update

```python
opt = collab_opts[p]  # Client p's persistent optimizer
opt.zero_grad()
loss.backward()        # Compute gradients
opt.step()             # Update weights using Adam
```

**Example Timeline for Batch 0, Epoch 0:**
```
Client 0:
  ├─ loss_kl = 0.8234  (from PWKT-weighted teachers)
  ├─ loss_arc = 0.0    (not struggling)
  ├─ total_loss = 0.8234
  └─ Gradients computed, weights updated

Client 1:
  ├─ loss_kl = 0.5123  (receives knowledge from 2 teachers)
  ├─ loss_arc = 0.0    (not struggling)
  └─ weights updated

Client 2:
  ├─ loss_kl = 0.7234  (limited teachers, lower quality)
  ├─ loss_arc = 0.3411  (RESCUE from Client 1!)
  ├─ total_loss = 1.0645
  └─ weights updated with rescue boost

Client 3:
  ├─ loss_kl = 0.6789
  ├─ loss_arc = 0.0
  └─ weights updated
```

---

## Complete Round Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│ COMMUNICATION EPOCH e (e.g., Epoch 0)                               │
└─────────────────────────────────────────────────────────────────────┘

PHASE A: EVALUATION
  For p in [0,1,2,3]:
    Load private test data
    accuracy[p] = evaluate_network()
    
  LOG: Accuracies, detect ARC rescue, identify best client

PHASE B: COLLABORATIVE LEARNING
  For batch_idx in range(len(public_train_dl)):  # ~20 batches
  
    For p in [0,1,2,3]:
      Generate MCE teacher target from each participant
      ├─ Multiple augmented views
      ├─ Temperature-scaled softmax
      └─ Average across views
    
    For p in [0,1,2,3]:  # Student loop
      Get student log-softmax from client p
    
    For p in [0,1,2,3]:  # Update loop
      Calculate PWKT weights [how much each teacher helps]
      
      loss = sum(weights[j] * KL_divergence(student[p], teacher[j]))
             + ARC_rescue_if_needed
      
      backward() → gradient computation
      optimizer_step() → update network weights
      
PHASE C: END OF EPOCH
  acc_prev = acc_current  (for next round's ARC calculation)
  
└─ Repeat for next epoch
```

---

## Data Flow Diagram

```
╔═══════════════════════════════════════════════════════════════════════╗
║                     PWHFL COMPLETE DATA FLOW                          ║
╚═══════════════════════════════════════════════════════════════════════╝

PRETRAINING PHASE:
─────────────────
CIFAR-10 (50K train)
    ↓
  IID Partition (random)
    ↓
[10K Client0] [10K Client1] [10K Client2] [10K Client3]
    ↓            ↓              ↓              ↓
[ResNet10]   [ResNet12]   [ShuffleNet]  [MobileNetV2]
    ↓            ↓              ↓              ↓
   Train         Train          Train          Train
  (40 epochs)  (40 epochs)   (40 epochs)   (40 epochs)
    ↓            ↓              ↓              ↓
  Save           Save           Save           Save
Checkpoints    Checkpoints    Checkpoints    Checkpoints


COLLABORATIVE PHASE (40 communication rounds):
───────────────────────────────────────────────
CIFAR-100 (50K train)
    ↓
  Random Sample
    ↓
[5K Public Data]
    ↓
[Batch0] [Batch1] ... [Batch19]  (256 samples each)
    ↓
    ├─→ [ResNet10]  ──→ MCE Target + Student Logits
    │
    ├─→ [ResNet12]  ──→ MCE Target + Student Logits
    │
    ├─→ [ShuffleNet] ──→ MCE Target + Student Logits
    │
    └─→ [MobileNetV2] ──→ MCE Target + Student Logits
    
    ↓
Compute PWKT Weights
    ↓
Knowledge Distillation Loss (weighted by PWKT)
    ↓
Add ARC Rescue Loss (if needed)
    ↓
Backward Pass & Update All 4 Models
    ↓
Next Batch / Next Epoch
```

---

## Key Hyperparameter Effects

### Private Corrupt Rate (0.5) Effects:
```
├─ LABEL_SMOOTH = 0.5 * 0.1 = 0.05
│  └─ Smooths hard labels during training
├─ JSD_L_TARGET = 0.5 * 2.0 = 1.0
│  └─ Target CE loss for ADA-JSD weighting
└─ ADA-JSD λ = ada_jsd_weight(ce_loss)
   └─ Adaptively weights JSD loss based on CE
```

### Test Corrupt Rate (0.0) Effects:
```
├─ PWKT_TAU = max(5.0, 20.0 * (1.0 - 0.0)) = 20.0
│  └─ Controls softness of PWKT weight distribution
└─ ARC_SCALE = 1.0 + 0.0 = 1.0
   └─ Multiplier for ARC rescue strength
```

### Public Corrupt Rate (0.5) Effects:
```
├─ TEACHER_TEMP = max(0.3, 1.0 - 0.5 * 0.7) = 0.65
│  └─ Sharpens teacher targets
└─ MCE_N_CROPS = round(1 + 2 * 0.5) = 2
   └─ Number of augmented views for MCE
```

---

## Summary of Major Events Per Round

**For each of 40 communication epochs:**

1. ✓ Evaluate all 4 models on test set
2. ✓ Track accuracy history (max, current, previous)
3. ✓ Detect struggling clients via ARC
4. ✓ Identify best-performing model
5. ✓ For ~20 public data batches:
   - Generate MCE targets from each model
   - Create student logits from each model
   - Calculate PWKT weights (knowledge transfer matrix)
   - Compute weighted KL divergence loss
   - Add ARC rescue loss for struggling clients
   - Backward pass → update all 4 models

**Total Training Events:** 
- 40 epochs × 20 batches = 800 update steps
- Per step: 4 teachers + 4 students = 8 forward passes, 4 backward passes

---

## Log Output Example

```
===== Comm Epoch 0 =====
Eval Avg: 83.85% | Clients: [85.2, 87.0, 82.4, 80.8]
  ARC: Client 2 declining Δ=-0.6% → λ=0.600
  Best client for rescue: 1 (ResNet12 @ 87.0%)
Local Train [256/5000 (5%)]: loss=0.8234 λ_jsd=12.00
  ARC rescue: client 2 ← client 1, λ=0.600, rescue_loss=0.3411
...

===== Comm Epoch 1 =====
Eval Avg: 84.12% | Clients: [85.8, 87.3, 83.1, 81.4]
  ARC: Client 3 declining Δ=-0.6% → λ=0.600
  Best client for rescue: 1 (ResNet12 @ 87.3%)
...
```
