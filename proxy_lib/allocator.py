"""Allocator — Token Plan credit aware continuous routing ratio."""
import hashlib


class Allocator:
    """连续比例分配器。

    sqrt 衰减：满额 50%，余额递减时平滑下降。
    prefer_level 已移除 — 无阶梯、无离散状态。
    """

    def compute_ratio(self, remaining: float, total: float, days: float) -> float:
        if days <= 0:
            return 0.0
        if remaining < 50:
            return 0.0

        pct = remaining / total if total > 0 else 0.0

        # 满额最多 50%，余额递减时 sqrt 衰减（前快后慢）
        ratio = 0.5 * (pct ** 0.5)

        # 冲刺：最后 5 天且余额 > 10%，至少 30%
        if days <= 5 and pct > 0.1:
            ratio = max(ratio, 0.3)

        return max(0.0, min(1.0, ratio))

    def should_route_to_max(self, complexity: str, req_id: str, ratio: float) -> bool:
        """Hash-based 概率分流。

        Parameters
        ----------
        complexity : str
            L2 分类结果 (trivial/simple/moderate/complex)
        req_id : str
            请求唯一 ID，保证确定性路由
        ratio : float
            compute_ratio 的输出
        """
        if complexity != "moderate":
            return False
        if ratio <= 0:
            return False
        if ratio >= 1.0:
            return True

        # 确定性 hash，不依赖随机数
        h = int(hashlib.md5(f"{req_id}:moderate".encode()).hexdigest(), 16) % 100
        return h < ratio * 100
