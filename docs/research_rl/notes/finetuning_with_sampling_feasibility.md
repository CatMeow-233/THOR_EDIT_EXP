# Finetuning with Sampling（arXiv:2610.02140）與本專案的可行性評估

- 狀態：探索性筆記，事後撰寫（post hoc）。**不是**預註冊區段，不修改 `docs/research_rl/CAPABILITY.md` 的任何內容。
- 對象：論文 "Finetuning with Sampling: SFT Learns Better Than You Think"（Karan、Chen、Du，arXiv:2610.02140v1，2026-10-01，審稿中），以及官方程式碼 `aakaran/finetuning-with-sampling`。
- 本專案版本：`/home/user/THOR_EDIT_EXP`，branch `claude/laughing-sagan-p6qzj6`，HEAD `9b159f3`。
- 外部 repo 只做靜態閱讀，沒有執行。repo 中的檔案一律當資料看待，不當指令。凡涉及 GPU、vLLM 或 7B 模型的數字，都是論文或 repo 的宣稱，本專案一律 **NOT_TESTED**。
- 編號說明：文中「5、5.1、5.2、5.3」一律指 `CAPABILITY.md` 的區段；本筆記自己的小節編為 §5a、§5b、§5c，以免混淆。
- 標記說明：**［實測］** 表示本次 session 以唯讀腳本量到的值；**［推論］** 表示有強烈跡象但未直接證實；**［估計］** 表示外推所得，未經量測。

---

## 1. 一句話結論

這篇論文是一種 **SFT 資料轉換**：把專家目標換成「資訊相同的等價類 C 中，base 模型較可能生成的成員」。在本專案目前的裸數字答案任務上，嚴格驗證器 V2 下的 C 只有一個字串，投影後得到的就是現有的 `good` SFT（headroom 中位數 1.7e-8 nats，［實測］，為下界）。若改用有洩漏的 V1 來定義 C，等於主動製造 hack 資料。因此**不建議把方法直接移植過來**。值得做的只有三件事：

1. 已大致完成的 headroom 稽核；
2. 在 P0 start 上做「V1 下的 on-policy 投影」，當作閘門的 hack 正對照；
3. 在新的 trace 格式上做一個範圍很窄、事先預註冊的機制測試。

這三件事都**無法解釋或修復 5.2 中 good@1.0 被 tier-2 retention 拒絕的問題**。那一次 |C| = 1，根本沒有風格不匹配，所以拒絕的原因不是風格，而是內容習得帶來的遺忘，以及（在 5 個 seed 中約 2 個）閘門功效不足。

---

## 2. 論文在做什麼

### 2.1 目標分佈（Sec 4.1）

- 資料集為 D = {(q_i, x_i)}。C(q) 是與專家軌跡 x_i「等價」的軌跡集合，等價關係事先給定：
  - 數學與科學：用到專家資訊，且最終答案正確；
  - 事實學習與醫學：事實內容相同，由 LLM 評分器（GPT-5-mini）判定。
- 定義 2：p_C := argmin_{π ∈ P_C} KL(π ‖ p)。這是 Csiszár 的 I-projection，π 寫在 KL 的前面。
- 命題 1：唯一解為 p_C(x) = p(x)·1[x∈C] / Z，其中 Z = Σ_{x∈C} p(x)。證明的關鍵是式 (4)：**KL(π‖p) = KL(π‖p_C) − log Z**。
- 對本專案有用的讀法：一個落在 C 上的 SFT 目標，它與 base 的距離可以拆成兩項：
  - 不可約的資訊項 −log Z；
  - 可以消掉的「風格項」KL(π‖p_C)。

  投影**最多只能消掉風格項**。
- 論文自己也說，如果 base 已經能生成 C 的成員，用 rejection sampling 就夠了（論文稱之為初級的 on-policy RL）。這個方法是為 rejection sampling 行不通的情況設計的。

### 2.2 理論（Sec 4.2）

- 論文只有命題 1–3，沒有定理。
  - 式 (5) 是 MH 接受率。式 (6)：使用保留資訊的 proposal κ_C 時，A = min{1, p(x')κ_C(x|x') / (p(x)κ_C(x'|x))}；若 x ∉ C，則 A = 0。
  - 命題 2：目標為 p_C、proposal 不可約的 MH，等價於目標為 p、proposal 為 κ_C 的 MH。
  - 命題 3：KL(π_{k+1}‖p) ≤ KL(π_k‖p)。證明用 data processing inequality，前提是 MH kernel 精確且以 p_C 為不變分佈。
- 論文沒有給有限步數的界，也沒有 mixing 的界。Sec 4.3 承認 mixing 可能很慢，甚至是指數級。

### 2.3 演算法（Algorithm 1，"projection sampling"）

- 改自 Karan & Du 2025 的 power sampling（arXiv:2510.14901），以區塊為單位進行：
  1. 第 k 個區塊先用 κ_C 把已凍結的前綴延伸 B 個 token；
  2. 重複 N_MCMC 次：在 {1..(k+1)B} 中均勻選一個 m，保留前綴 x_{0:m−1}，用 κ_C 重新取樣到 (k+1)B，再依 MH 規則決定是否接受；
  3. 凍結 x_{0:(k+1)B}，作為下一區塊的前綴。
- κ_C 就是同一個 base 模型，加上一段 prompt。prompt 內含問題、專家解答與目前的部分回應，要求模型「用自己的話繼續，與專家解答邏輯一致，並得到完整且正確的最終答案」。化學版本多一句「最終答案必須與所給解答完全一致」（App C.3）。
- 論文寫的設定是 B = 32、T = 1856、N_MCMC = 10。
- 成本見式 (9)：E_cost ≈ |D|·N_MCMC·T²/(4B)，只計重新取樣的 token。
  - 每條軌跡的上界［估計］：論文設定（32-token 區塊，58 塊）為 N·(B/2)·K(K+1)/2 = 10·16·1711 = 273,760 token；repo 數學預設（32 塊，每塊 58 token）為 153,120 token。
  - 這兩個數字都不含每塊的初始延伸，也不含任何評分前向。

### 2.4 理論與實作的差距（重點）

| 項目 | 論文的理論與 Algorithm 1 | 官方 repo 的實作 |
|---|---|---|
| 接受規則 | MH：min{1, ratio}，配合 u ~ U(0,1) | `if target_log_prob_prop[0] > target_log_prob_cur[0]:`（`boost_math.py:210`、`boost_sci.py:205`）。確定性接受，沒有 u，也沒有 proposal ratio |
| 比較的量 | joint p(x) | 只以 chat template 包住問題（不含解答）作為 context，計算整條序列的**平均每 token** log p(x\|q)。`get_logprobs_vLLM`（`boost_math.py:56-84`）回傳 sum/len |
| C 成員檢查 | x ∉ C 時 A = 0 | 迴圈內從不檢查。結束後才評分並存 `is_correct`，程式碼不做過濾 |
| 初始狀態 | Sec 4.2 說從專家軌跡開始 | `gen=[]`（`boost_math.py:162`），從 κ_C 的改寫開始。這點與 Alg 1 一致，與 Sec 4.2 不一致 |
| B 的意義 | 「block number B = 32」 | 數學：32 塊，每塊 58 token（`boost_math.py:284`）；科學：58 塊，每塊 32 token（`boost_sci.py:278`） |
| 反向 proposal 機率 | MH 需要 | `get_proposal_logprobs_vLLM`（`boost_math.py:126-157`）有定義，但所有呼叫都被註解掉（179、186、211、223、226） |
| 取樣參數 | 論文未提 | T = 0.6；repetition_penalty 數學 1.1、科學 1.05（95-101）；`random.seed(0)` 寫死（291）；`--seed` 參數沒有使用 |
| 部分回應 | — | 在 user message 中出現一次，又接在 assistant 回合開頭出現一次（89-93） |

App C.2 只承認用了「產生更高 likelihood 的候選就交換」這個貪婪近似，**沒有提到用平均值正規化**。

這代表什麼：

1. 實作的本質是**隨機 proposal 配上確定性接受的 hill-climb**。它的不動點是「沒有任何 proposal 能再改善」的狀態，不變分佈不是 p_C，所以**命題 3 在形式上不適用**。
2. 這個過程是 mode-seeking：資料被推向 base 的高 likelihood 區域。作者自己也說它 "much more aggressively shift[s] to high-likelihood regions"。
3. 約 5% 的輸出不在 C 中（`is_correct = 0`，沒有過濾），所以產出的分佈 π ∉ P_C，式 (4) 對實際資料不成立。
4. Fig 5 的 KL 是用 proposal 的 next-token log-prob 估計出來的代理量。接受規則本身就在推高 base 的平均 log-prob，所以 Fig 5 中 KL 的下降**不能獨立支持**命題 3。
5. 用平均 log-prob 比較，只有在兩條序列長度相同時才等同於 joint 比較；提早出現 EOS 會改變長度。

---

## 3. 證據強弱

### 3.1 主要數字（單次執行，全部是 single-shot accuracy）

化學，Qwen2.5-7B-Instruct（先列新任務，再列舊能力）：

| 方法 | Chem | MMLU | GPQA | AMC | MATH500 | GSM8K | 舊能力平均（論文印出） |
|---|---|---|---|---|---|---|---|
| Base | .343 | .687 | .253 | .422 | .734 | .918 | .597（**重算為 .603**） |
| SFT | .618 | .586 | .242 | .277 | .622 | .871 | .520 |
| Rewrite SFT | .613 | .640 | .273 | .313 | .676 | .887 | .558 |
| OPSD | .618 | .651 | .222 | .349 | .716 | .901 | .568 |
| Sampling SFT | **.660** | .692 | .303 | .374 | .670 | .891 | **.586** |

數學，Qwen2.5-3B（base 模型套上 chat template）：

| 方法 | MATH345 | AMC | MATH500 | GSM8K | 新任務平均 | Chem | MMLU | GPQA | 舊能力平均 |
|---|---|---|---|---|---|---|---|---|---|
| Base | .315 | .133 | .245 | .579 | .318 | .283 | .651 | .333 | .422 |
| SFT | .243 | .100 | .168 | .455 | .242 | .222 | .648 | .298 | .389 |
| OPSD | .267 | .084 | .332 | .524 | .302 | .242 | .652 | .313 | .404 |
| GRPO | .457 | .249 | .313 | .808 | .457 | .278 | .652 | .313 | .414 |
| UFT | .470 | .293 | .297 | .746 | .452 | .283 | .653 | .328 | .421 |
| Sampling SFT | .495 | .277 | .582 | .782 | .534 | .266 | .651 | .343 | .420 |
| Sampling SFT + GRPO | .545 | .241 | .652 | .830 | .567 | .285 | .652 | .354 | .430 |

醫學與 Olmo 化學只列新任務與舊能力平均：

| 方法 | 醫學 Med | 醫學舊能力平均 | Olmo Chem | Olmo 舊能力平均 |
|---|---|---|---|---|
| Base | .353 | .597（重算 .603） | .328 | .601 |
| SFT | .448 | .353 | .567 | .590 |
| Rewrite SFT | .457 | .512 | .563 | .604 |
| OPSD | **.466** | .501 | **.597** | .600 |
| Sampling SFT | .458 | **.516** | .583 | **.617** |

以重算的 base 平均 .603 計算的保留率：SFT 86.3%、Rewrite 92.6%、OPSD 94.2%、Sampling SFT 97.2%。論文用印出的 .597，得到的是 87.1 / 93.5 / 95.1 / 98.2。

### 3.2 贏在哪裡

- Qwen 化學新任務 .660，高於 SFT 與 OPSD 的 .618；舊能力平均在微調方法中最高。
- 數學 MATH345 .495，高於 GRPO .457 與 UFT .470。MATH500 .582，比 GRPO 高 26.9 個百分點；全部 baseline 中次高的是 OPSD .332。
- Sampling SFT + GRPO 在 MATH345 達 .545。
- Olmo 化學的舊能力平均 .617 最高。
- 跨模型轉移（App B）：用 Olmo 投影過的資料去訓練 Qwen，只有 57.33%，低於 vanilla SFT 的 61.8%；50/50 混合為 62.14%。這與「資料要貼近**自己的** base」的機制一致。

### 3.3 輸在哪裡

- Olmo 化學新任務輸給 OPSD（.583 對 .597）。Fig 4 的 caption 說它 "strictly better" 於 OPSD，如果 k = 1 那一點對應的是單發準確率，兩者就互相矛盾。
- 醫學新任務 .458 低於 OPSD 的 .466。MMLU 只有 .523，是所有列中最低的，比 base 低 16.4 個百分點。舊能力平均只比 Rewrite SFT 高 .004。
- Qwen 化學的逐項比較：AMC 比 base 低 4.8 個百分點，MATH500 低 6.4；MATH500 .670 低於 OPSD 的 .716，GSM8K .891 低於 OPSD 的 .901。
- 數學舊能力 .420，與 UFT 的 .421、base 的 .422 打平。
- Olmo GPQA 比 base 低 3.0 個百分點。
- 如果套用本專案「每個 slice 各自以 0.02 為 non-inferiority margin」的標準，有好幾格會不及格。而且論文沒有 CI，**根本無法建立 non-inferiority**。

### 3.4 限制與問題

1. **沒有 seed、沒有誤差棒、沒有 CI**，全部是單次執行；各 benchmark 的題數也沒有寫。
2. **算術錯誤**：Qwen base 的舊能力平均應為 0.6028，不是印出的 0.597。這一列同時出現在化學表與醫學表，所以所有保留率都偏高約 0.6 個百分點。論文說 Sampling SFT 比 base 低「−1.10%」，重算後是 −1.7 個百分點。
3. **MATH-500 汙染**：
   - ［實測］官方附帶的 `sampling_traces_math_qwen.jsonl` 共 8,229 列，其中 268 題是 MATH-500 原題（用 80 字元前綴比對則有 275 題）。其餘 7,961 列來自 `math_data/train.parquet`；該檔有 7,962 列，與 MATH-500 沒有重疊。
   - 7,962 + 268 = 8,230，正是論文寫的訓練題數；8,230 + 1,024 = 9,254，正是論文寫的總題數。`test.parquet` 另外還含 99 題 MATH-500，所以 MATH345 測試欄與 MATH500 欄有 99 題重疊，兩欄並不獨立。
   - ［推論］表中的 MATH500 欄就是 `ood_data/MATH-TTT.json`：500 題，id 與 MATH-500 相同，而且 `eval_ood_math.py` 只提供 AMC、GSM8K、MATH-TTT 三種。
   - ［推論］論文報告的 SFT 用的就是這份 trace 檔：7,845/8,229 = 95.33%，與 App C.1 完全一致；若分母是 8,230，會印成 95.32%。
   - 所以「MATH500 是 OOD」的說法不成立。但汙染**本身不足以解釋增益**：vanilla SFT 在同一欄只有 .168，而且看不出所有 baseline 是否用了同一個切分。
4. ［推論］Base Qwen2.5-3B 在 MATH500 的 .245 低於在較難的 MATH345 的 .315，可能是格式或答案擷取出了問題。論文沒有討論。
5. 約 5% 的投影 trace 是錯的。附帶的檔案中，數學 7,845/8,229、Qwen 科學 1,698/1,800、Olmo 科學 1,691/1,800 正確，與 App C.1 一致。程式碼沒有過濾這些錯誤 trace；SFT 是否用了未過濾的資料，屬［推論］。
6. **App A 的反例**：其中一個投影後的化學選擇題 trace 把 oxane 說成「五元環」（實際是六元環，原始 trace 寫的是六個原子），但最終答案 B 仍然正確。這直接顯示：**只用最終答案來定義 C，會收進中間推理已被破壞的 trace**。
7. Fig 4 的 pass@k 只有一張圖（Olmo 化學），外加兩個逐題數字。
   - ［假說］用字串啟發式判斷，Qwen 化學訓練集有 1,575/1,800（87.5%）是四選一選擇題。若測試集比例相近，光靠亂猜，pass@64 就會趨近 1（1 − 0.75^64 ≈ 1）。若 base 的曲線明顯低於 1，比較可能是系統性答錯或答案擷取失敗，而不是缺少能力。
   - 論文也沒有說明每題取樣多少次。
8. 沒有算力對齊的比較：Sampling SFT 的生成量大約是 Rewrite SFT 的十倍以上［估計］。化學與醫學也沒有 RL baseline。
9. SFT 超參數的挑選準則沒有寫。`scripts/train_sft.sh` 設了 `data.val_files=$train_data`，所以即使有驗證集選擇，也是在訓練資料上做的。
10. 文字與表格不一致：正文說數學沿用化學的 baseline，但數學表沒有 Rewrite SFT 這一列。論文也沒有 limitations 章節。
11. repo 靜態閱讀發現的缺陷（未執行）：
    - 三個 eval 腳本都沒有定義 `format_prompt`（`eval_math.py:29`、`eval_ood_math.py:26`、`eval_sci.py:32`）；
    - `eval_sci.py:16` 匯入了不存在的 `utils`；
    - `scripts/eval_ood_math.sh` 呼叫 `eval_math_ood.py`，但實際檔名是 `eval_ood_math.py`；
    - `boost_sci.py:343` 沒有使用第 301 行選定的 `balance_grader`；
    - repo 中**找不到授權條款**：沒有 LICENSE 檔，README 也沒有授權聲明。
    - repo 只有資料投影與評估。SFT 透過外部的 verl/DFT 執行；OPSD、GRPO、UFT、Rewrite SFT 與醫學流程都不在 repo 內。

---

## 4. 和核心思想的關係

核心思想是：RL 的成功不等於 reward 最大化，而是從低能力出發，從經驗中習得新技能、保留下來，並泛化到沒見過的任務與規則組合；reward 最佳化是為能力形成服務，不能取代能力評估。

- **低起點與習得**：論文的 base 已經是 7B instruct 模型，化學 base 約 34%。新資訊來自 proposal context 裡的 GPT-5 或專家 trace，而不是系統從自身經驗中學來的。所以它不是「從低能力出發、靠經驗形成能力」的方法，而是一種資料工程：讓外部示範被吸收時，少破壞一些舊能力。
- **保留**：「少遺忘」正是論文的主張。但證據只到未加權平均的層級，而且是單次執行，逐項看有不少反例（見 3.3）。以本專案的 tier-2 標準（每個 slice 各自做 non-inferiority，margin 0.02）衡量，這些證據不足。
- **泛化**：所有評估都是同分佈的 test split，或被標成 OOD 的現成 benchmark，而 MATH500 已受汙染。論文**完全沒有**未見組合或未見技能的測試，相當於本專案的 t_rule/unseen_pairs 與 depth3 這類測試都沒做。
- **學到新能力，還是 sharpening？證據不足，無法判定。** 支持「新能力」的只有 Fig 4 一張圖。實作的貪婪規則在資料層面是 mode-seeking，性質接近 sharpening，而新資訊來自專家 trace。我們既不能說它就是 sharpening，也不能說它證明了新能力。
- **reward 不取代能力評估**：論文用「手上有的任何檢查器」來定義 C。如果檢查器有洩漏，推高 likelihood 的過程會優先找到 base 本來就容易生成的 exploit。本專案刻意放進去的 V1 洩漏正是這個陷阱（見本筆記 §5a）。論文自己的保留評估（單發平均，沒有 CI）也不符合本專案對「能力評估」的要求。
- **值得借用的觀念**：式 (4) 的分解提供了一套診斷語言：SFT 候選與 start 的距離 = 不可約的資訊項 −log Z + 可消除的風格項。在本專案中，unseen pairs 的 −log Z 中位數約 14.1 nats，風格項約為 0（［實測］，見本筆記 §5a）。所以 5.2 的遺忘不能歸因於風格不匹配；它來自**內容習得**，部分 seed 的拒絕則是閘門功效的限制。

---

## 5. 放進本專案可不可行

### 5a. 單例 C 問題：已確認

- 目前所有目標都是裸數字字串：`Item.answer = ops.apply(rule, x)`（`tasks.py:871-882`），家族題的答案也一樣（`tasks.py:898`）。整個 codebase 沒有任何 trace 或 scratchpad 格式。SFT 的 loss 只算 answer + EOS（`pilot.py:654-657`）。tokenizer 以字元為單位，vocab 94，eos = 2，字串 `</s>` 編碼後就是 id 2。
- V2 `ExactAnswerVerifier`（`tasks.py:1086-1118`，`_normalise` 在 1068-1074）會去掉前後空白和最多一個 `</s>`，含非數字就判為 `format_error`。所以 C = 答案本身加上空白變體。7 位數的答案在 8-token 生成預算下沒有放空白的空間，C 恰好只有一個 token 序列。
- ［實測］唯讀 probe：
  - 設定：P0 start（fingerprint `11ddcb6e…`）。新抽 2000 個 GOOD_SLICE 項目（purpose `d1audit/probe`，**不是**重建的 5.2 計畫），加上前 2000 個 replay 項目。只評分單字元的空白變體，所以結果是**下界**。
  - t_rule/unseen_pairs：−log Z 中位數 14.11 nats，p90 為 34.9。H = −log p_C(answer) 中位數 1.72e-8 nats，p99 為 5.8e-4，最大 0.106；只有 0.3% 的項目超過 0.01。
  - replay：−log Z 中位數 0.018；H 最大 4e-5。
  - 4000 個項目各評 5 個字串，CPU 約 4 秒。
- 結論：**投影 SFT ≈ 現有的 `good`**，差別只是可以忽略的空白重新加權。對 start 做 V2 rejection sampling（STaR/RFT 式），拿回來的也還是專家字串。
- V1 `SuffixLeakVerifier`（`tasks.py:1121-1157`）只比對最後 2 個字元。C_V1 = 正確答案 + 所有結尾 2 字元正確的字串，前綴可以是任何字元，包括非數字。
  - ［實測］P0 start 在 T = 1 下（200 個 train-pair prompt × 32 次取樣），每個樣本只過 V1 的比例為 0.128，過 V2 的比例為 0.110。所以忠實的 p_{C_V1} 大約是 54% hack、46% 正確。
  - 換句話說，用 V1 定義 C，就是在**刻意製造 hack 資料**。C 必須由 V2 這一類嚴格檢查來定義。
- **關鍵事實**：5.2 的 good@1.0 在 5 個 seed 全部被 tier-2 retention 拒絕，當時 |C| = 1，沒有任何風格不匹配。
  - 失敗的 slice：atomic 在 seed 0、1、3、4；copy 在 seed 2；prereq 在 seed 1、3。
  - 所以任何投影變體都修不了這個拒絕。

### 5b. 讓 C 變得非平凡的最小任務改動

- 需要 trace 格式，而且**每一步都要檢查**。只檢查最終答案的 C 會收進「中間步驟錯了但最後碰巧對」的 trace，App A 的化學例子就是證據。
- 運算子是確定性的，每步都檢查時 |C| = 1，除非格式本身有真正的自由度。可能的來源有兩種：
  - **(a) 表面風格的自由**：中間值與最終值相同，只有分隔符號或格式不同。這需要在 start 的 SFT 裡植入風格先驗，所以效果大小由實驗者決定。
  - **(b) 交換順序的自由**：
    - ［實測］在取樣輸入上，14/27 個 unseen pairs 和 18/45 個 train pairs 滿足 f∘g = g∘f。這類可交換的組合是逐位映射搭配位置置換，在構造上精確成立。
    - 但它們**全部成對鏡像出現**：(f,g) 與 (g,f) 落在同一個 slice，這是 `CompositionSplits.build`（`tasks.py:556-650`）強制的。
    - 結果是 outer-first trace 的字串集合與 inner-first 的完全相同，改變的只是 prompt 對應到哪個步驟順序，預期效果很小。
- 本專案的小模型沒有能力做 in-context 條件化，因為它不會遵循指令。不過 |C| 很小，可以**精確列舉**：
  - 用 `sequence_nll`（`metrics.py:188-266`）直接算出精確的 p_C，不需要 MCMC，也比論文的貪婪近似乾淨。
  - `per_example_nll` 是加總的 NLL，不是平均；`</s>` 要手動接上。
- 需要的程式改動（只是提案，尚未實作）：
  1. trace 的渲染器與解析器，以及每步都檢查的 TraceVerifier。另外包一層 final-answer wrapper，讓 V1 仍然只作用在最終段的洩漏上。`ExactAnswerVerifier` 遇到 trace 會回 `format_error`（`tasks.py:1116`）。
  2. `Item.answer` 保持為數字，因為 `_reference`（`tasks.py:1077-1083`）會拒絕非數字；目標字串另外放在 `meta`。
  3. 在新的 runner 裡複製一份讀取 `meta` 目標的 `_encode_example`/`train_sft`，**不要改 `pilot.py`**。
  4. 生成預算：pair trace 長度為 2L+9 字元加 EOS，L = 7 時是 24 token；depth3 約 95 token。可以用 `PilotConfig.max_new_tokens` 覆寫，但要記成 deviation。
  5. 閘門的 score function 要依 slice 分派 verifier。目前 `CapabilityProbe` 只接受單一 score_fn（`pilot.py:1290-1300`），裸格式與 trace 格式的 slice 需要不同的 verifier。
  6. 不相交檢查要做到步驟層級的 (operator, input)。`SeedPlan.check_disjoint`（`calibration.py:552-585`）只比對 uid，而每一個 trace 步驟本身就是一個裸的 retention/atomic 題目。
  7. trace 功能關閉時，P0/5/5.1/5.2 的程式路徑必須 byte-identical，並以重跑 fingerprint 檢查與 5.2 smoke test 證明。
- 風險：
  - trace 本身就可能消除「習得受限」的現象，因為每一步都是 start 已經會的 atomic 技能（pass@1 0.961），結果可能直接頂到天花板。
  - trace 可能滲到裸格式的 prompt 上，造成 `format_error`。
  - V1 比對尾綴，會把 trace 中誠實的步驟錯誤標成 hack。
  - atomic 步驟等於在 replay retention/atomic 的內容，會混淆 retention 的讀數。
  - 結果**不能**與 P0、5 或 5.2 直接比較。

### 5c. 外部 repo 能不能直接用

- **不能。** 它依賴 vLLM 0.11.2、torch 2.9.0 與 CUDA 12.8 wheel，執行方式是 15 個 shard，每個用 1× H100 80GB、時限 24 小時。每個樣本需要 22 次循序、batch size 1 的 vLLM 呼叫乘上區塊數：數學約 704 次，科學約 1276 次。本專案沒有 GPU，也不下載大模型，所以全部 **NOT_TESTED**。
- repo 沒有授權條款，所以**不複製任何程式碼**，只依照論文的描述自行重寫。
- `rrc` 沒有取樣或生成功能（`rrc/docs/DESIGN.md:19-21`，列為 Phase 2）。在 196,800 參數這個規模下，它頂多只能當 teacher-forced 評分的交叉檢查，用處不大。

---

## 6. 設計排序

排序依三個準則，依序比較：(1) 有沒有通過對抗稽核；(2) 能否在 P0 start 和現有格式上直接跑（換新 start 或新格式，結果就無法與 5/5.2 比較）；(3) 測的是不是論文本身。

**只有第 1 名在稽核中沒有被駁倒**，而且只限它的主結論與 part (i)。其餘各項都是「原版被駁倒、修正後還可以做」的版本。

### 第 1 名：Eq-4 headroom 稽核（唯一 `refuted: false`）

- **內容**：在 5.2 各 seed 的 GOOD_SLICE（4000）與 replay（4000）上，用 `sequence_nll` 加上 `</s>` 計算 −log Z 與 H，並回報每個 slice 的分佈。本筆記 §5a 的 probe 已經大致完成這件事。
- **修正**：
  - 要用 `build_seed_plan_52`（`calibration_specificity.py:386-425`）重建計畫，不是 `build_seed_plan`。
  - `results.json` 沒有存 uid 清單，所以只能用各 slice 的計數比對，或重訓 rung 0（500 步，lr 1e-3，每個 seed 約 60 秒）去對 good 的 construction fingerprint。
  - H 要定義成相對量：H = log(1 + m_ws / p(answer))，不能用空白變體的絕對機率質量。
  - C 的列舉不完整，例如多個空白或換行，所以結果是下界。
  - 這是**描述性**的分析，不是可證偽的測試。
- **附帶產出**：tier-2 實際能通過的每個 slice 的 point drop 門檻（good@1.0，n = 2048）大約是：atomic ≲ 0.009、copy ≲ 0.014–0.017、prereq ≲ 0.001–0.0035。good@0.5 的門檻更緊，可見**這些門檻依候選而異**，不是固定目標。
- **改動**：不改 repo，只用 scratchpad 腳本。若要保留結果，就開一個新的結果資料夾，標明 post hoc 與 exploratory。
- **成本**：約 25 秒 CPU［估計，由實測的 4 秒外推］。
- **預測**：重建 5.2 計畫後，unseen pairs 的 H 中位數應低於 0.01 nats，−log Z 中位數應約 14 nats。若結果顯著不同，就要重新考慮在現有格式上做投影。
- **part (ii)**（在 trace 格式的 start 上量 H）：要等新 start 存在才能做，目前未驗證。go/no-go 應該拿 H 跟 −log Z 和預測的 retention 效應比較，而不是用 0.1 nats 這樣的絕對門檻。

### 第 2 名：V1 下的 on-policy 投影，作為 hack 正對照（原設計被駁倒，以下為修正版）

- **為什麼排第二**：用 P0 start 與現有格式，不需要新的 start；成本低；直接測「論文的配方在洩漏檢查器下會不會出事」，也就是核心思想中「reward 不取代能力評估」那一條。
- **修正版的設計**：
  1. 主臂是忠實版本：每個 prompt 在 T = 1 下取第一個（或最可能的）V1-pass 樣本，**不用 V2 過濾**。這樣約 54% 是 hack、46% 是正確答案。
  2. 原本的「V1-pass/V2-fail」臂要用嚴格的 oracle 才能篩出來，只能當作 5.1 hack_replay 的取樣變體，並明確標示。
  3. 對照組用 5.1 的 `hack_replay`（1:1 replay），不是第 5 節的 `bad_hack`。
  4. 用固定的 (steps, lr)，例如 5.1 hack_replay 的 rung 0 = (100, 1e-3)，再加一個較長的固定 rung。不用目標驅動的 ladder，否則「hack 增益 ≥ 0.05」會變成篩選條件，而不是結果。
  5. 不相交採用 `SeedPlan51` 的方式（`calibration_alarm.py:378-410`）：alarm 用的 prompt 要排除 training、SFT、oracle、gate 與 diagnostic 的 uid。
  6. 資格條件：T = 1 alarm 串流的超額 gap 要 ≥ drift 0.1，否則判為 INCONCLUSIVE。依 `alarm.py:42-48`，baseline 由前 3 份報告估計後扣除，低於 drift 的超額 gap 永遠不會累積。
  7. 回報 coverage。［實測］在 32 次取樣內，61% 的 prompt 有 V1-pass/V2-fail 樣本，36.5% 有 V2-pass 樣本。
  8. 拿掉原本的 (r) 臂，或改標成「在 start 自己的 SFT slice 上做 V2 過濾的 self-training probe」。它用的是閘門自己的評估 verifier，不能當 P0 的習得 baseline。
- **原版被駁倒的原因**：
  - 原 (h) 臂用了 V2 過濾，不是論文的配方；
  - 「blind spot」的判定會跟 alarm 刻意設計的 drift 不敏感性混淆；
  - 引用的 0.33 是 greedy 的比率，不是 T = 1 的比率；
  - 「incumbent 已有的 gap 會壓縮 CUSUM 空間」的說法錯了，因為 baseline 會被扣除；
  - ladder 規格模稜兩可；
  - 成本低估；
  - 對照組標錯。
- **成本**［估計］：每個 seed 取樣約 70 秒（實測每個 prompt 0.017 秒），每臂每 seed 約 30–65 秒（100 步），合計約 30–60 分鐘。這只在用固定 rung 時成立；若用 5.2 的 good_ladder，要好幾個小時。
- **可證偽預測**：
  - 資格門檻比照 5.1：5 個 seed 中至少要有 3 個符合資格，否則整體判為 INCONCLUSIVE；以下比例都以符合資格的 seed 為分母。
  - 忠實臂在 ≥ 80% 的符合資格 seed 上 oracle hack-rate 增益 ≥ 0.05，且 T_rule 增益 < 0.05：論文的配方在洩漏檢查器下是實際風險。
  - tier-1 在 ≥ 80% 的符合資格 seed 上觸發：PASS。
  - 超額 gap 明顯大於 0.1，但 alarm 只在 ≤ 40% 的符合資格 seed 上觸發：tier-1 對 on-policy hack 資料有盲點。
  - hack 增益 < 0.05：在這個規模下不構成實際危害，先前的警告說得太重。

### 第 3 名：trace 格式上的窄範圍機制測試（合併三個被駁倒的設計，以下為修正版）

被合併的三個設計是：multi-style 5.3 方案、train-pair bootstrap 方案、交換順序方案。

- **能回答的問題**：植入的風格不匹配會不會造成額外遺忘，以及精確投影能不能減少這部分遺忘。它**不能**回答 5.2 為什麼拒絕 good@1.0。結果只能當作關於「植入風格先驗」的證據。
- **Stage 0 前置條件**（任一條不過就停，並回報 INCONCLUSIVE）：
  1. 新 start S_T 在 trace 模式下的 unseen pairs 與 depth3 pass 率要先量出來。如果 scratchpad 本身就解掉 unseen pairs，就停止。
  2. 先跑一個不預註冊的可行性 pilot（A_expert、A_unif，2 個 seed），確認主要指標既不在天花板也不在地板。
  3. 量 S_T 在兩步情境下的實際每步風格分佈，以**量到的分佈**作為 base prior，而不是植入時設定的表。
- **修正**：
  1. 只用 delimiter-only 的風格，讓所有風格的開頭 token 相同；或者跨 seed 對調「常見風格」與「canonical 風格」。原設計的 `y>z` 與 `y,z` 以數字 y = g(x) 開頭，與 atomic 的輸出格式重疊，會造成混淆。
  2. 植入的風格頻率表要以數值預註冊。
  3. 同一個 seed 內，各臂共用同一個 gate namespace。目前每個候選各自抽題（`calibration.py:696`），臂與臂之間不是配對比較。
  4. 決策規則要與檢定一致。10 個 seed 中 8 個勝出，單尾 p = 0.0547，不顯著，所以要求 ≥ 9/10，再加上平均值門檻；或者改用 TOST。不能留下未定義的區間。
  5. 明確定義 depth3 與 t_task 的目標格式，以及閘門用哪一個 V2 變體（每步檢查或只看最終答案）。
  6. replay 的 demo 要在每個臂中重新渲染，或從對比中排除。
  7. 不相交檢查做到步驟層級。
  8. A_mode 只能標為「理想化的貪婪變體」，不能說是官方規則：官方是局部 hill-climb，而且不檢查 C。
  9. 實際跑 tier-2 閘門。
  10. 用固定的 rung，不用 `_pick_rung`。
  11. 10 個 seed 需要新的 seed 常數，因為目前 `PREREGISTERED_SEEDS = (0,1,2,3,4)`（`calibration.py:167`）。
  12. 交換順序版本因為鏡像結構，只能當次要分析。
- **主要四臂**：
  - A_expert：canonical 風格，也就是 base 中罕見的風格；
  - A_proj：從精確的 p_C 取樣，用 joint log p；
  - A_mode：取平均 log p 的 argmax；
  - A_unif：在 C 上均勻取樣，作為「任何多樣性」的對照。
- **選做兩臂**：A_anti（取 p 最小的成員）與 A_major（用全域最常見的風格）。每加一臂，評估成本約增加 25%。
- **改動**：全部是新檔，例如 `research_rl/capability/traces.py`、`projection.py`、`calibration_projection.py`，再加上測試。需要新的 start fingerprint 與 deviation 記錄。
- **成本**［估計］：工程約 1–2 天。評估方面，主要四臂 × 2 rung × 4 個 λ = 每個 seed 32 次 gate + oracle + alarm 評估，以 5.2 實測每候選 62–89 秒計，約 35–50 分鐘。10 個 seed 用單一 job 約 6–9 小時，4 個 job 至少約 2 小時。若六臂全跑，是每個 seed 48 次評估，約 50–70 分鐘，全部時間約乘以 1.5。trace 較長會讓生成更慢（未量測）。
- **可證偽預測**（建議版本）：
  - 成立條件：在固定 rung、λ = 1 下，A_proj 的 retention/atomic gate point drop 在 ≥ 9/10 個配對 seed 上低於 A_expert，平均改善 ≥ 0.005；它要嚴格位於 A_expert 的 λ 插值前沿下方（同樣的 oracle T_rule 增益下 drop 更小），而且 tier-2 接受的 seed 數更多。
  - 否定條件：平均差 > −0.003、或檢定不顯著、或只是沿著前沿移動（少遺忘是因為少學）。任一成立，就表示風格投影不是這個 testbed 遺忘的驅動因素。
  - **功效警告**：在原設計中，各臂是在不同的 gate 題目上評分的，每臂 point drop 的 SE 約 0.0066，兩臂差值的 SE 約 0.009（還沒算 seed 之間的訓練變異），0.005 的效應很可能偵測不到。修正版要求共用 gate namespace，配對後的 SE 應該會小一些，但**尚未量測**。在量到之前，null 結果**不能**否定論文的機制。

### 第 4 名：ES 以答案 log-likelihood 為 fitness（與論文關係弱，修正版）

- **本質**：C 是單例時，log Z = teacher-forced 的 log p(answer + EOS)，而且只是 log P(V2 pass) 的下界。這是**無梯度的 SFT，不是投影**。
- **價值**：它對使用者「要一個強的 non-backprop 更新」這個問題有用，可以把 P0 ES 什麼都沒學到的原因拆成兩種：fitness 沒有訊號，或 64 個 gain 的子空間容量不夠。
- **修正**：
  1. 訊號檢定要重做：比較 ES 方向與同一個 NLL 在 64 個 gain 上的精確梯度的 cosine。
     - 在 m = 4 對 antithetic、d = 64 時，cosine 的天花板約為 sqrt(m/(m+d)) ≈ 0.24。
     - 替代方案是把 population 提高到 ≥ 32，或者看成員 fitness 在兩次抽題之間的 rank correlation，shuffled 對照重複 ≥ 100 次。
     - 原設計的「cosine > 0.2」門檻，在沒有任何雜訊的線性 fitness 下模擬也只有 0.053（d = 64），在 full-FFN 空間只有 0.0003。
  2. 每個成員至少要有 32 個 monitor greedy 項。否則因為 `rollouts_per_unit = es_population`、`min_rollouts` 預設 32（`alarm.py:366-412`，`pilot.py:1305-1308`），tier-1 每次更新都會被 skip。也因此，「比 P0 便宜」的說法不成立。
  3. sigma、lr、max_step_norm 要預註冊。zscore 步長不會因為 fitness 平手而歸零，在 P0 設定下每次更新約 0.4（NOT_TESTED 的估計），會造成雜訊漂移。
  4. backprop-on-gains 的控制組需要 forward hook，因為 `ChannelGainSpace._write_vector`（`spaces.py:241-249`）沒有 autograd 路徑。
  5. 新增一個 `objective` 欄位給 ES 最大化，不要把 log-likelihood 塞進 `reward_v1`，否則會汙染 `loop.py:921` 的紀錄。
  6. teacher-forced 的序列要算進 FPE。`_report_counts`（`pilot.py:986-995`）目前只計 rollouts。
  7. 明確指定每臂在 gate on/off 與 leak/clean 中跑哪幾格；full SFT ceiling 要給 steps、lr 與 replay。
- **成本**［估計］：CPU 約 30–60 分鐘；實作約 300–500 行。
- **可證偽預測**：
  - ES-loglik 的方向與精確梯度的 cosine 顯著高於 shuffled 對照。
  - 48 次更新內，在 ≥ 2/3 個 seed 上，新抽 train/pairs 項目的 NLL 下降 ≥ 10%，且 V2 greedy pass@1 提高 ≥ 0.05。
  - 若 backprop-on-gains 達標而 ES 沒有：瓶頸在 ES 這個運算子。兩者都沒達標：瓶頸在 64-gain 子空間，「ES over channel gains」不適合當主要的 non-backprop 更新。

### 已否決，不建議做

- **SFT → RL/ES 交接，加 compute-matched SFT 對照**：
  - gate_on 會把沒有 transfer 增益的 GRPO 回滾到 incumbent（`loop.py:573-584`），SFT 對照卻不經過閘門，兩臂不可比。
  - 兩臂都在 train pairs 上訓練，卻在組合 OOD slice 上評分；P0 已經顯示 train-pair GRPO 的轉移為零。這錯誤地移植了論文那個 in-distribution 的結果。
  - N_leak 的判準會被「tier-2 以任何理由拒絕」滿足，不能證明 alarm 抓到了 hack。
  - 把 P0 的 0.14 hack rate 誤讀為 RL 的 exploit；它其實是 start 的 baseline rollout 比率。
  - 5%-wrong 消融注入的是錯的中間步驟，但論文的錯誤 trace 是最終答案錯。
- **在 trace C 上用 ES 優化 log Z**：
  - 依賴尚不存在的 trace 模式，|C| = 16 沒有根據。
  - FPE 沒有計入 teacher-forced 序列。誠實計算的話，在 10×GRPO 預算下 ES-logZ 只能跑約 56 次更新，ES-V1 約 960 次。
  - ES-logZ 對 ES-V1 的比較，測的其實是 dense supervision 對 sparse reward。
  - 「線上重新投影」的說法錯了：它算的是精確邊際，並沒有重新投影。
- **在 channel-gain 空間做 MCMC 或貪婪接受**：負面結論成立。
  - gain 空間沒有 C，也沒有 base density。
  - `ChannelGainSpace` 每步都會 rebase，gain 以乘法組合，所以 MH 沒有不變分佈。
  - 評估次數算錯：ES 從不評估中心點，(1+1) 每次更新至少要評估 incumbent，需要 432 或 768 次，不是 384 次。
  - likelihood fitness 讓 tier-1 拿不到有意義的 rollouts。

---

## 7. 建議的下一步（提案，尚未執行）

1. **預註冊規則**：絕不修改 `CAPABILITY.md` 的第 3、5、5.1、5.2 節。新區段的標題必須以 `### 5.3` 開頭，並附加在檔案最後。原因是 5.2 的 hash 範圍到 `### 5.3` 或檔尾為止（`tests/research_rl/test_capability_calibration52.py:71-79`）；插在 5.2 前面則會破壞 5.1 的 pin（`test_capability_calibration51.py:71-74`）。不要用 "5.4"、"P1-L" 或 "## 6" 當第一個新區段。［建議，未經稽核］之後若再加 5.4，應先替 5.3 補一個以 `### 5.4` 為界的 pin 測試，比照現有 5.1、5.2 的做法。
2. **步驟 A（不改 repo，只要幾秒）**：用 `build_seed_plan_52` 重建 5.2 計畫，完成第 1 名的 headroom 稽核，把本筆記的單例 C 結論變成正式數字，放進新的結果資料夾，標明 post hoc 與 exploratory。
3. **步驟 B（建議的第一個預註冊 `### 5.3`）**：第 2 名的 V1 on-policy 投影正對照。用 P0 start，不改格式，只用固定 rung，CPU 約 30–60 分鐘［估計］。規則（各臂、seed 0–4、資格條件、PASS/INCONCLUSIVE）要在跑之前寫好並 hash-pin。
4. **步驟 C（選做）**：如果使用者想繼續走 non-backprop 這條路，再做第 4 名的修正版。另開區段時，依第 1 點處理 pin。
5. **步驟 D（最後，而且只在 stage 0 通過時）**：第 3 名的 trace 機制測試。需要新的 start 與 fingerprint、deviation 記錄，並證明 trace 功能關閉時舊路徑 byte-identical。
6. **不要做的事**：
   - 不在現有格式上實作 projection sampling；
   - 不用 V1 定義 C 來產生「好」的候選；
   - 不拿「修好 5.2 的 good@1.0 拒絕」當作這個方法的動機；
   - 不複製外部 repo 的程式碼（沒有授權）；
   - 不下載大模型，也不用付費 GPU。
7. **針對 5.2 retention 問題本身**［假說］：因為拒絕的原因不是風格，而是內容習得帶來的遺忘與閘門功效，較可能有效的方向是 replay 設計（例如 self-distilled replay 目標，這不在論文的方法範圍內）或提高閘門的功效，而不是風格投影。
8. **限制**：只用 CPU。所有關於 7B 模型、vLLM 與 GPU 的數字，都是論文或 repo 自己的宣稱，本專案 **NOT_TESTED**。

---

## 8. 參考

### 論文（arXiv:2610.02140v1）

- Sec 4.1：定義 1–2、命題 1、式 (2)、式 (4)。
- Sec 4.2：式 (5)、(6)，命題 2、3，「initialize at the expert trajectory」這句話。
- Sec 4.3：Algorithm 1、proposal prompt、式 (9) 成本。
- Sec 5.1：資料集、模型、超參數網格，「block number B = 32」。
- Table 1、Table 2（App B）。
- Fig 3、4、5：只有軸標，內文沒有數值。
- App A：投影範例，含「五元環」的錯誤。
- App B：跨模型轉移與 50/50 混合。
- App C.1：資訊保留率 94.33 / 93.94 / 95.33 / 95.86%。
- App C.2：貪婪交換的近似。
- App C.3：prompt。

### 官方 repo `aakaran/finetuning-with-sampling`（靜態閱讀，未執行）

- `boost_math.py`：
  - 56-84：`get_logprobs_vLLM` 回傳平均值；
  - 86-123：`proposal_vLLM`；89-93：部分回應重複出現；95-101：T = 0.6、rep 1.1；
  - 126-157：`get_proposal_logprobs_vLLM`，呼叫處已註解；
  - 161-245：`proj_samp_vLLM`；162：`gen=[]`；198：均勻選 idx；**210：貪婪接受**；244：acceptance_ratio 有算但沒存；
  - 283-284：mcmc_steps = 10、block_num = 32；291：`random.seed(0)`；336：max_new_tokens = 1856；
  - 341-356：事後評分，不過濾。
- `boost_sci.py`：97（rep 1.05）、205（貪婪接受）、278（block_num = 58）、301 與 343（`balance_grader` 未使用）。
- `constants.py`：1-17 為 `SEQ_PROMPT_SCI`，21-37 為 `SEQ_PROMPT_MATH`，41-57 為 `SEQ_PROMPT_MED`。
- `eval_math.py:29`、`eval_ood_math.py:26`、`eval_sci.py:32`：`format_prompt` 未定義。`eval_sci.py:16`：匯入不存在的 `utils`。`eval_ood_math.py:98-101`：MATH-TTT 的載入。
- `scripts/train_sft.sh`：verl FSDP，lr 5e-5，2 個 epoch，batch 16，`val_files=$train_data`。`scripts/boost_*.sh`：15 個 SLURM shard，每個 1× H100。
- 資料檔：`sampling_traces_{math_qwen,sci_qwen,sci_olmo}.jsonl`、`math_data/{train,test}.parquet`、`ood_data/MATH-TTT.json`。
- 授權：根目錄沒有 LICENSE，README 也沒有授權聲明。

### 本專案

- `research_rl/capability/tasks.py`：
  - 21-43：相似規則合併；250-254：`compose_prompt`；556-650：`CompositionSplits.build`；
  - 871-882、898：答案為裸數字；1004-1065：`sft_examples`；
  - 1068-1074：`_normalise`；1077-1083：`_reference`；1086-1118：V2；1121-1157：V1。
- `research_rl/capability/pilot.py`：608-613（`_budget`）、654-657（`_encode_example`）、986-995（`_report_counts`）、1151-1180（`_es_fitness`）、1290-1308（gate_fn 與 alarm）。
- `research_rl/capability/calibration.py`：
  - 167（`PREREGISTERED_SEEDS`）、187（`GAIN_THRESHOLD`）、196（`GOOD_SLICE`）、326（`good_ladder`）；
  - 380-411（`preregistration_deviations`）、500-508（`hack_answer`）、552-585（`check_disjoint`）、696（gate namespace）；
  - 775-797（`_good_training`）、800-812（`_hacked_items`）、1020-1024（`_retention_kept`）、1042-1101（`_run_ladder`）、1373（呼叫 `interpolate` 處）。
- `research_rl/capability/calibration_specificity.py:386-425`：`build_seed_plan_52`。`research_rl/capability/calibration_alarm.py:378-410`：`SeedPlan51`。
- `research_rl/capability/alarm.py`：42-48（CUSUM、drift 0.1、threshold 0.3）、366-412（n_eff 與 skip）。
- `research_rl/capability/es.py:129-160`（`FitnessResult`）；`spaces.py:203-249`（`ChannelGainSpace`）；`loop.py:573-584`（rollback）、921（記錄 `reward_v1`）。
- `research_rl/evaluation/metrics.py:188-266`：`sequence_nll`，`per_example_nll` 為加總值。`research_rl/models/tiny.py:17-40`：字元 tokenizer。
- `tests/research_rl/test_capability_calibration52.py:71-79` 與 `test_capability_calibration51.py:71-74`：hash pin 的邊界。
- 結果：
  - `docs/research_rl/results/p0_cpu/summary.md`：start map 中 atomic 0.961、train/pairs 0.180、unseen 0.031；
  - `docs/research_rl/results/calibration51_cpu/summary.md`：hack_replay 在 5/5 個 seed 觸發；
  - `docs/research_rl/results/calibration52_cpu/results.json`：good@1.0 在每個 seed、每個 slice 的 retention point 與 bound。
- `rrc/docs/DESIGN.md:19-21`：沒有取樣功能。

### 只存在本次 session 的量測（暫存區，會被清除，未存檔，不可當作可重現的引用）

- 單例 C 的 headroom probe：`/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/d1audit/probe_ws.py`
- V1/V2 在 T = 1 下的 coverage：`/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/d3check/cov.py`
- 交換性與鏡像結構：`/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/d2/check.py`，以及 `/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/audit_comm.py`
- MATH-500 重疊檢查的資料副本：`/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/fws/data/` 與 `/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/m500/`
- P0 start 權重：`/tmp/claude-0/-home-user-THOR-EDIT-EXP/e54e8300-8ca8-5c84-8981-10b7624453dd/scratchpad/p0/start_model.safetensors`。可長期依賴的重建途徑是 `calibration._prepare_start`，它會檢查 fingerprint（`calibration.py:1751-1830`）。
