# api_optimal_transport

## 项目介绍
本项目实现了基于最优传输理论的 API 对齐方法，用于跨版本 API 迁移和代码理解。通过将旧版和新版 API 映射到低维空间，利用最优传输理论计算最优对齐，实现 API 之间的语义对齐。

# Step 1: API Embedding & Pre-Alignment Baseline

本步骤目标：  
将每个 API 的三元组 `(signature, description, code_slice)` 编码为向量，生成旧版和新版的 embedding 矩阵，并做轻量预对齐，准备进入后续的 OT/FGW 对齐。

---

## 0. 数据准备
- **清洗**：
  - signature：统一格式（空格/注解/参数名风格）
  - description：小写、去 HTML/URL
  - code_slice：截取函数定义及必要上下文
- **模板拼接**：
[SIG] <signature>
[DESC] <description>
[CODE] <code_slice>
- - 工具：`regex`, `ftfy`, `tree_sitter` / `libcst`

---

## 1. 编码器选择
- **推荐 baseline**：单编码器（句向量模型）
- 模型：`sentence-transformers` 系列，如 `e5-base-v2` / `bge-base-en`
- 输入：拼接好的模板串
- 输出维度：≈768
- **处理**：
- 最大长度：512 tokens
- 批大小：64（按显存调整）
- **L2-normalize** 每个向量

产出：
- `E_old.npy` (m × d)
- `E_new.npy` (n × d)

---

## 2. 有效维度估计
- 对 `E_old ∪ E_new` 做 PCA
- 取解释 90–95% 方差的维数 r（通常 64–128）
- 工具：`scikit-learn PCA`

---

## 3. 预对齐 (Alignment)
### 3.1 无监督（无锚点）
- 方法：CORAL (Covariance Alignment)
- 去均值 → 白化 → 按新版统计重新着色
- 工具：`numpy`, `scipy.linalg`

### 3.2 弱监督（推荐，有锚点）
- 方法：正交 Procrustes
- 输入：K 对已知旧↔新 API 对应
- 输出：旋转矩阵 R
- 对齐：`E_old' = E_old · R`
- 工具：`scipy.linalg.svd`

### 3.3 校准（可选）
- 在验证锚点集上做相似度温度标定

---

## 4. 锚点对 (Anchor Pairs)
- 数量建议：
- K ≈ 3r–5r （若 r≈64–128 → 200–600 对）
- 最少 50–100 对可跑 Procrustes
- 获取途径：
- 文档/CHANGELOG：明确的替代关系
- 同名/近名 + 签名高度相似
- Deprecation 提示 ("use ... instead")
- 工具：`rapidfuzz` (字符串相似度)

---

## 5. 验收指标
- Dev 集上：
- Top-1 ≥ 0.70
- Top-5 ≥ 0.90
- AUC ≥ 0.90
- 确认对齐后内部最近邻结构未严重破坏

---

## 6. 产出物
- `E_old_aligned.npy`, `E_new.npy`
- `procrustes_R.npy` (若做了)
- `pca_meta.json` (含 r、解释率)
- `anchor_train.csv` / `anchor_dev.csv`
- `scaler_meta.json` (均值/协方差或温度参数)
- `embedding_config.json` (模型名、max_len、batch_size、hash)


# Step 2: Cost Matrix Construction & Sinkhorn OT Alignment

本步骤目标：  
利用旧版和新版的 embedding 矩阵，构造代价矩阵 (cost matrix) 并通过 Sinkhorn 最优传输 (OT) 得到软匹配矩阵 P。

---

## 0. 输入
- `E_old.npy` (m × d)：旧版 API 向量矩阵
- `E_new.npy` (n × d)：新版 API 向量矩阵
- `old_ids.csv` / `new_ids.csv`：索引表（API id 与向量行号对应）

---

## 1. 构造代价矩阵 M
- **基础代价**：余弦距离  
  \( M_{ij} = 1 - \cos(E_{old}[i], E_{new}[j]) \)  
- **可选融合项**：  
  - `signature_gap(i,j)`：参数数量/名称差异  
  - `contract_gap(i,j)`：异常、前置条件、后置条件差异  
  - `incompat_penalty(i,j)`：已知不兼容的直接大罚  
- **输出**：`M.npy` 或稀疏 `M_knn.npz`

---

## 2. kNN 剪枝（推荐）
- 为每个旧版 API，只保留新版本中 **前 k 个最相似候选**（k≈50–200）  
- 其余位置代价设为大值或直接删除  
- **输出**：稀疏代价矩阵 `M_knn.npz`

---

## 3. Sinkhorn OT 求解
- **OT (Optimal Transport)**：最小化总代价 \(\langle P, M\rangle\)，得到运输矩阵 P  
- **Sinkhorn**：带熵正则的快速 OT 算法，数值稳定  
- **参数**：  
  - 熵正则 ε = 0.05 ~ 0.2  
  - 不平衡强度 τ = 0.5 ~ 2.0（如果 API 数量不相等）  
- **输出**：运输矩阵 `P.npy` (m × n)

---

## 4. 后处理
- 从软匹配 P 生成候选映射：  
  - **Top-k**：每行取前 k 个新 API  
  - **阈值化**：丢弃低于阈值的匹配  
  - **容量约束**：避免多个旧 API 挤到同一个新 API

**输出**：`mappings/topk_k=3.csv`, `mappings/final.csv`

---

# Step 3: Anchor-Based Evaluation

本步骤目标：  
利用已知的旧↔新锚点对，评估软匹配结果的质量。

---

## 0. 输入
- `P.npy`：运输矩阵  
- `anchors_train.csv` / `anchors_dev.csv`：锚点对（已知 ground truth）

---

## 1. 指标
- **Top-k 准确率**：预测的前 k 新 API 中是否包含真实匹配  
- **AUC**：软匹配分数对正/负样本的区分能力  
- **结构保持率**（若有图结构）：对齐后邻居是否仍保持

---

## 2. 验收标准
- Dev 集上：  
  - Top-1 ≥ 0.70  
  - Top-5 ≥ 0.90  
  - AUC ≥ 0.90  

---

## 3. 输出
- `reports/metrics.json`：包含 Top-k 准确率、AUC 等  
- `reports/topk.csv`：逐样本匹配情况  
- 可选可视化：置信度分布、PR/ROC 曲线
