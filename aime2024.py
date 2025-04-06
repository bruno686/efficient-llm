from vllm import LLM, SamplingParams
import pandas as pd
import re
import numpy as np

# 超参数定义
N_SAMPLES = 4  # 采样次数，可调整，例如 4
K_VALUES = [1, 2]  # 要计算的 pass@k 的 k 值列表，可调整，例如 [1, 2]

def extract_boxed_answer(text):
    match = re.search(r'\\boxed\{(.*?)\}', text)
    return match.group(1).strip() if match else None

def estimate_pass_at_k(num_samples: np.ndarray, num_correct: np.ndarray, k: int) -> np.ndarray:
    """Estimates pass@k of each problem and returns them in an array."""
    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

# 加载数据集
df = pd.read_parquet("/home/kaiyu/cheers/test_LLM/aime2024/aime_2024_problems.parquet")
questions = df['Problem'].tolist()
answers = df['Answer'].astype(str).tolist()

# 初始化模型和采样参数
llm = LLM(model="/home/kaiyu/cheers/LLMs/r1-distill-qwen-7b")
params = SamplingParams(temperature=0.7, n=N_SAMPLES, top_p=0.95, max_tokens=8000)

# 存储每个问题的采样总数和正确答案数
num_samples = N_SAMPLES  # 每个问题采样 N_SAMPLES 次
correct_counts = []

for q, gt in zip(questions, answers):
    prompt = f"Question: {q}\nPut your final answer within \\boxed{{}}"
    outputs = llm.generate(prompt, params)
    generated_texts = [out.outputs[0].text for out in outputs]
    preds = [extract_boxed_answer(t) for t in generated_texts if extract_boxed_answer(t)]
    
    # 统计该问题正确答案的数量（不去重，基于原始 N_SAMPLES 个生成结果）
    num_correct = sum(1 for pred in preds if pred == gt)
    correct_counts.append(num_correct)

# 将 correct_counts 转换为 numpy 数组
correct_counts = np.array(correct_counts)

# 计算 pass@k
for k in K_VALUES:
    pass_at_k_values = estimate_pass_at_k(num_samples, correct_counts, k)
    # 对所有问题的 pass@k 取平均值，作为整体指标
    avg_pass_at_k = np.mean(pass_at_k_values)
    print(f"pass@{k}: {avg_pass_at_k:.3f}")
