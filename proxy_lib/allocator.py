"""MultiModelAllocator — Token Plan credit aware multi-model routing.

SELECT is the sole budget gate: ratio determines what fraction of traffic
reaches TP models. No per-model secondary threshold.
"""
import hashlib


class MultiModelAllocator:
    """多模型分配器。

    (complexity, task_type) → 查映射表 → hash 门(ratio) → 返回 TP model_name。
    ratio 是唯一大门，过了就分配，不加第二道门槛。
    不在映射表中的组合返回 None（留 DeepSeek）。
    """

    # (complexity, task_type) → TokenPlan 模型名（ROUTES key）
    _MAPPING: dict[tuple[str, str], str] = {
        ("moderate", "code"):          "kimi-k2.7-code",
        ("moderate", "creative"):      "glm-5.2",
        ("moderate", "long_context"):  "qwen3.6-flash",
        ("complex",  "code"):          "kimi-k2.7-code",
        ("complex",  "creative"):      "glm-5.2",
        ("complex",  "reasoning"):     "qwen3.7-max",
        ("complex",  "long_context"):  "qwen3.7-plus",
        ("complex",  "general"):       "qwen3.7-max",
    }

    def compute_ratio(self, remaining: float, total: float, days: float) -> float:
        """连续比例分配。sqrt 衰减：满额 50%，递减时平滑下降。"""
        if days <= 0:
            return 0.0
        if remaining < 50:
            return 0.0

        pct = remaining / total if total > 0 else 0.0
        ratio = 0.5 * (pct ** 0.5)

        # 冲刺：最后 5 天且余额 > 10%，至少 30%
        if days <= 5 and pct > 0.1:
            ratio = max(ratio, 0.3)

        return max(0.0, min(1.0, ratio))

    def select(self, complexity: str, task_type: str, req_id: str, ratio: float) -> str | None:
        """查映射表 → hash 门 → 返回 TokenPlan 模型名或 None。

        Parameters
        ----------
        complexity : str
            L2 分类结果 (trivial/simple/moderate/complex)
        task_type : str
            L2 分类结果 (code/creative/reasoning/long_context/general)
        req_id : str
            请求唯一 ID，保证确定性路由
        ratio : float
            compute_ratio 的输出 (0.0~1.0)
        """
        if complexity not in ("moderate", "complex"):
            return None

        target = self._MAPPING.get((complexity, task_type))
        if target is None:
            return None

        if ratio <= 0:
            return None
        if ratio >= 1.0:
            return target

        if self._hash_gate(f"{req_id}:{complexity}:{task_type}", ratio):
            return target
        return None

    def _hash_gate(self, key: str, ratio: float) -> bool:
        """确定性 hash 门。返回 True 表示通过（ratio% 的概率）。"""
        h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10000
        return h < ratio * 10000
