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
