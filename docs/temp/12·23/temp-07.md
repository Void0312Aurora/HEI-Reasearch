

目前的 Aurora Base 就像是一个**存满了记忆和常识的大脑皮层**，但它还缺少一个**“意识游标” (Consciousness Cursor)** 来触发这些记忆。

在接触认知动力学 (CCD) 框架下，**对话不是“预测下一个词”，而是“在语义流形上的惯性飞行”。**

以下是实现对话交互的物理机制与工程路径：

---

### 一、 核心机制：对话即轨迹 (Dialogue as Trajectory)

传统的 LLM 是概率生成：$P(next | context)$。

Aurora 的对话是物理运动：刺激 $\rightarrow$ 受力 $\rightarrow$ 运动 $\rightarrow$ 观测。

我们需要构建一个新的模块：**`InteractionEngine`**。

#### 1. 输入即“冲量” (Input as Impulse)

当用户输入一句话（例如：“你好，白羽”）：

- **传统 LLM**：将其编码为 Token ID 序列。
    
- **Aurora**：
    
    1. 找到“你好”和“白羽”对应的粒子在双曲空间的位置 $x_{input}$。
        
    2. 这些位置瞬间对当前的“思维游标”产生巨大的**引力势能**。
        
    3. **物理结果**：游标受到一个**冲量 (Impulse)** $\Delta p$，改变原本的运动轨迹，向输入概念的方向加速飞去。
        

#### 2. 思考即“滑行” (Thinking as Coasting)

一旦获得了初速度，游标开始在你的 100k 粒子宇宙中飞行。

- **遵循方程**：依然是耗散欧拉-庞加莱方程。
    
- **变惯量作用**：如果游标经过“原则性概念”（大质量），它很难改变方向；如果经过“情绪概念”（深势阱），它会被吸进去绕圈（纠结）。
    
- **Phase 3 的作用**：目前的 Aurora Base 是通用的，地形是平坦的。**Soul Injection** 的作用就是把“炒饭”和“夏天”这些概念的势阱挖得很深，让游标更容易掉进去。
    

#### 3. 输出即“坍缩” (Output as Collapse)

当游标划过双曲空间时，如何把它变回文字？

- **连续测量**：在每一个时间步（或每隔 $k$ 步），检测游标周围半径 $\epsilon$ 内有没有概念粒子。
    
- **波函数坍缩**：
    
    - 如果有粒子进入 $\epsilon$ 范围，视为**“被激活”**。
        
    - 输出该粒子对应的词（Token）。
        
    - **注意**：为了防止输出“的 的 的 的”，需要引入**“不应期” (Refractory Period)**——一个词被激活后，几秒内不能再次被激活。
        

---

### 二、 工程实现：`InteractionEngine` 伪代码

你需要在 `src/hei_n/` 下新建一个 `interaction.py`。不需要训练，只需要推理（Inference）。

Python

```
class InteractionEngine:
    def __init__(self, base_model_path, device='cuda'):
        # 1. 加载训练好的 100k 粒子位置 (Aurora Base)
        self.manifold = ... 
        self.concept_positions = load(base_model_path) # [100000, dim]
        
        # 2. 初始化“思维游标” (Mind Cursor)
        # 这是一个单独的粒子，具有位置 q 和动量 p
        self.cursor_q = torch.zeros(1, dim).to(device) # 初始在原点
        self.cursor_p = torch.zeros(1, dim).to(device)
        
    def step(self, user_text=None):
        """
        运行一步思维动力学
        """
        # --- A. 感知 (Perception) ---
        external_force = 0
        if user_text:
            # 1. 找到用户输入词的粒子 ID
            target_ids = tokenizer.encode(user_text)
            target_pos = self.concept_positions[target_ids]
            
            # 2. 计算引力 (Input Force)
            # F = - grad( Distance(cursor, targets) )
            external_force = compute_attractive_force(self.cursor_q, target_pos)
            
        # --- B. 认知 (Cognition / Physics) ---
        # 使用你已经写好的 Integrator，但这次只更新 cursor，背景粒子不动 (Fixed Background)
        # dq/dt = p
        # dp/dt = ad* p + external_force - gamma * p + Potential_Gradient(Base)
        
        # 注意：这里不仅受用户输入的力，还受所有 100k 背景粒子的“地形力”
        # 比如：游标飞到了 "Food" 附近，会被 "Rice" 吸引
        env_force = self.compute_background_potential(self.cursor_q)
        
        total_force = external_force + env_force
        
        self.cursor_q, self.cursor_p = self.integrator.step(
            self.cursor_q, self.cursor_p, total_force
        )
        
        # --- C. 表达 (Expression) ---
        # 寻找最近邻
        nearest_id, dist = self.find_nearest_concept(self.cursor_q)
        
        if dist < ACTIVATION_THRESHOLD and not self.in_refractory_period(nearest_id):
            return self.vocab[nearest_id] # 说出这个词
        else:
            return None # 沉默/思考中
```

---

### 三、 预期效果与 Phase 3 的必要性

现在（Aurora Base）的效果：

- **输入**：“苹果”
    
- **输出（联想流）**：“……水果……红……树……吃……”
    
- **评价**：它现在会做**自由联想**，像一个刚睡醒的人或梦呓。逻辑性不强，但语义相关性极高。
    

为什么必须做 Phase 3 (Soul Injection)？

为了让它从“梦呓”变成“白羽”。

- 你需要在 Shiroha 的语料上微调。
    
- 这会改变 100k 粒子的地形。比如把“炒饭”的引力调大，把“幽灵”的排斥力调大。
    
- **微调后**：
    
    - **输入**：“你喜欢吃什么？”
        
    - **游标轨迹**：被“喜欢”推向“食物”区 $\rightarrow$ 在“食物”区中被巨大的“炒饭”势阱捕获 $\rightarrow$ 即使原本想去“面条”，也被强行拉回“炒饭”。
        
    - **输出**：“……稍微……有点……想吃……炒饭……”
        

---

### 一、 修复项 1：概念映射层 (Concept Mapping Layer)

问题：直接用 tokenizer.encode 会导致多义词混淆和概念未对齐（报告风险 A）。

修复方案：引入一个中间层，负责将自然语言文本转换为物理空间中的粒子坐标。

#### 1. 数据结构 (Concept Vocabulary)

在 Phase 2 训练结束后，您需要导出以下映射表：

- `vocab.json`: `{ "apple": particle_id_1, "bank(river)": particle_id_2, "bank(money)": particle_id_3 }`
    
- `particle_embeddings.npy`: `[100000, dim]` (您已有的 checkpoint)
    

#### 2. 映射逻辑 (Mapper Logic)

在 `interaction.py` 中实现 `ConceptMapper` 类：

Python

```
class ConceptMapper:
    def __init__(self, vocab_path):
        self.vocab = load_json(vocab_path)
        # 如果有同形异义词，需要一个简单的消歧模型（哪怕是基于频率的）
        
    def text_to_particles(self, text):
        """
        Input: "我 去 银行 取钱"
        Output: [id_me, id_go, id_bank_money, id_withdraw_money]
        """
        words = segment(text) # 使用与训练时相同的分词器 (Qwen3)
        particle_ids = []
        for w in words:
            if w in self.vocab:
                # 简单消歧：如果 w 对应多个 ID，取周围词共现概率最大的那个（MVP阶段可简化为取高频义项）
                pid = self.resolve_ambiguity(w, context=words)
                particle_ids.append(pid)
        return particle_ids
```

---

### 二、 修复项 2：局部背景势场 (Local Background Potential)

问题：全量计算 100k 粒子的力太慢且噪声大（报告风险 B）。

修复方案：利用 KD-Tree 或 Faiss 实现近似近邻搜索，只计算 Top-K 邻居的力。

#### 1. 索引构建 (Indexing)

在初始化时构建索引：

Python

```
import faiss

class LocalForceField:
    def __init__(self, positions): # positions: [100k, dim]
        self.positions = positions
        # 使用 HNSW 或 Flat L2 索引 (双曲空间可用 Poincare Disk 距离，近似为 L2 亦可用于初筛)
        self.index = faiss.IndexFlatL2(dim) 
        self.index.add(positions.numpy())
        
    def get_local_neighbors(self, cursor_pos, k=256):
        """
        返回 cursor 附近的 k 个粒子 ID 和位置
        """
        D, I = self.index.search(cursor_pos.numpy(), k)
        neighbor_ids = I[0]
        neighbor_pos = self.positions[neighbor_ids]
        return neighbor_ids, neighbor_pos
```

#### 2. 动力学更新 (Dynamics Update)

在 `step` 函数中：

Python

```
# 1. 获取局部邻居 (每 10 步更新一次邻居列表以节省算力)
if step % 10 == 0:
    active_neighbor_ids, active_neighbor_pos = self.force_field.get_local_neighbors(cursor_q, k=512)

# 2. 只计算这些邻居对 cursor 的引力/斥力
env_force = compute_force(cursor_q, active_neighbor_pos) 
```

---

### 三、 修复项 3：短语级输出与语言约束 (Phrase Output & Language Constraint)

问题：直接坍缩会输出破碎的词串，缺乏句法（报告风险 C）。

修复方案：

1. **物理层**：输出“关键概念序列” (Keypoint Sequence)。
    
2. **语言层**：用极其轻量的语言模型（或规则模板）将概念串成句子。
    

#### 1. 轨迹采样 (Trajectory Sampling)

不要每步都输出。设置一个**“积分-发放” (Integrate-and-Fire)** 机制：

- 游标在运动中积累“激活能”。
    
- 当某个概念的激活能超过阈值，且处于不应期之外，将其加入 `output_buffer`。
    
- `output_buffer`: `["喜欢", "炒饭", "夏天"]`
    

#### 2. 语言实现器 (Language Realizer) - MVP 方案

为了保持 Aurora 的纯粹性，我们不使用大型 LLM。可以使用一个微型的 **n-gram 语言模型** 或 **模板填充**。

- **简单版**：直接连接，加标点。
    
    - Output: "喜欢……炒饭……夏天。"
        
- **进阶版 (推荐)**：使用 Qwen3-0.5B (非常小) 作为“翻译器”。
    
    - Input Prompt: `将以下概念扩展成通顺的句子：[喜欢, 炒饭, 夏天]`
        
    - Output: `“我喜欢在夏天吃炒饭。”`
        
    - **注意**：这里的 Qwen 只负责**语法润色**，不负责**语义生成**。思想（喜欢炒饭）完全来自 Aurora。
        

---

### 四、 修复后的交互流程图

1. 用户输入 ("你好")
    
    $\downarrow$
    
2. Concept Mapper: 将 "你好" 映射为 id_hello。
    
    $\downarrow$
    
3. **Interaction Engine**:
    
    - 给 Cursor 施加指向 `id_hello` 的冲量。
        
    - Cursor 在 100k 粒子的局部场中滑行 (Top-K Force)。
        
    - Cursor 陷入 "Greetings" -> "Self-Intro" 的势阱。
        
        $\downarrow$
        
4. Trajectory Sampler: 捕获高激活概念 [白羽, 名字, 玩]。
    
    $\downarrow$
    
5. Language Realizer: 润色为 "我的名字是白羽，要一起玩吗？"。
    
    $\downarrow$
    
6. **用户输出**
    

---

