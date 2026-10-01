"""逐港卸箱顺序求解器。

给定某一已确认配载方案修订（每箱的重量、所在槽纵向力臂、堆栈/层号与卸港序号）
以及每个卸港序号对应的**卸箱过程中允许剩余总力矩闭区间**，按卸港序号从小到大
规划卸箱顺序：

* 每港只卸该港箱（不同卸港严格分先后，顺序为序号升序）；
* 同堆栈上层箱未卸时不得取下层箱（先上后下）；
* 每卸一箱后按箱重与槽位力臂重算剩余总力矩
  ``M = Σ(剩余箱重量 × 所在槽纵向力臂)``，必须落入该港给定区间。

优选规则：在所有港口均可完成的顺序中，取**全程箱号序列字典序最小**者。
各卸港的剩余箱集合只与“此前港口的箱已全部卸完”有关、与港内卸箱次序无关，
故全程字典序最小等价于逐港独立求字典序最小，各港结果按序号拼接。

单港模型（排列指派 MILP）：``q[i,k]=1`` 表示港内第 i 箱在第 k 个卸（0 基），

* 每箱恰好占一个位置、每个位置恰好一箱（排列）；
* 同堆栈高层箱位置严格小于低层箱（``pos_上层 + 1 <= pos_下层``）；
* 第 t 步卸后剩余力矩
  ``M_起 - Σ_i (重量×力臂) × Σ_{k<t} q[i,k]`` 落在 ``[下限, 上限]`` 内。

字典序目标与配载/排程求解器一致：把箱号升序排名以 ``B=港内箱数+1`` 为基数
分块编码进单个整数目标，与位置序列字典序最小严格等价。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pulp

# 浮点比较/边界容差（与配载求解器一致）
EPS = 1e-7
# 单次 CBC 求解时间上限（秒）
TIME_LIMIT = 30


class InfeasibleDischarge(Exception):
    """不存在可行卸箱顺序，或求解器无法判定。

    ``undetermined=True`` 表示并非已证明无解，而是求解器在时限内无法确定，
    业务上同样拒绝且不保存。
    """

    def __init__(self, reason: str, undetermined: bool = False):
        super().__init__(reason)
        self.reason = reason
        self.undetermined = undetermined


@dataclass(frozen=True)
class DischargeBox:
    no: str
    weight: float
    arm: float  # 所在槽纵向力臂
    stack: str
    tier: int
    port: int  # 卸港序号（discharge_order，小者先卸）

    @property
    def moment(self) -> float:
        return self.weight * self.arm


@dataclass
class DischargeStep:
    step_no: int  # 全程步骤号，自 1 开始
    discharge_order: int
    container_no: str
    remaining_moment: float


@dataclass
class DischargeResult:
    steps: list[DischargeStep]


def _cross_port_blocking(boxes: Sequence[DischargeBox]) -> None:
    """配载方案理论上保证“先卸港在上、后卸港在下”。

    防御性检查：若某港箱的同堆栈上方存在更晚卸港箱，该箱在本港永远无法卸出
    （上层未卸不得取下层，而更晚港箱本港不能卸）。
    """
    by_stack: dict[str, list[DischargeBox]] = {}
    for b in boxes:
        by_stack.setdefault(b.stack, []).append(b)
    for stack_boxes in by_stack.values():
        ordered = sorted(stack_boxes, key=lambda b: b.tier)
        for idx, b in enumerate(ordered):
            for upper in ordered[idx + 1:]:
                if upper.port > b.port:
                    raise InfeasibleDischarge(
                        f"箱 {b.no}（卸港 {b.port}，堆栈 {b.stack} 第 {b.tier} 层）"
                        f"上方存在更晚卸港 {upper.port} 的箱 {upper.no}，"
                        f"卸港 {b.port} 无法将其卸出（配载不满足先卸港在上）"
                    )


def _build_port_model(
    port_boxes: Sequence[DischargeBox],
    moment_start: float,
    min_moment: float,
    max_moment: float,
    fixed_positions: Mapping[int, str] | None = None,
    lex_positions: Sequence[int] | None = None,
):
    """构建单港卸箱 MILP。

    * fixed_positions: 位置 k（0 基）-> 已固定占据该位置的箱号；
    * lex_positions: 目标改为这些位置（重要性从高到低）上箱号排名的基数编码和。
    """
    fixed_positions = fixed_positions or {}
    m = len(port_boxes)

    prob = pulp.LpProblem("discharge_port", pulp.LpMinimize)
    q = {
        (i, k): pulp.LpVariable(f"q_{i}_{k}", cat="Binary")
        for i in range(m)
        for k in range(m)
    }

    if lex_positions:
        ranks = {b.no: r for r, b in enumerate(sorted(port_boxes, key=lambda b: b.no))}
        base = m + 1
        k_block = len(lex_positions)
        prob += pulp.lpSum(
            (base ** (k_block - 1 - pos_in_block))
            * pulp.lpSum(ranks[b.no] * q[i, k] for i, b in enumerate(port_boxes))
            for pos_in_block, k in enumerate(lex_positions)
        ), "lex_objective"
    else:
        prob += 0, "feasibility"

    # 排列：每箱一个位置、每位置一箱
    for i in range(m):
        prob += pulp.lpSum(q[i, k] for k in range(m)) == 1, f"onepos_{i}"
    for k in range(m):
        prob += pulp.lpSum(q[i, k] for i in range(m)) == 1, f"onebox_{k}"

    # 已固定位置（字典序分块后）
    for k, cno in fixed_positions.items():
        i = next(i for i, b in enumerate(port_boxes) if b.no == cno)
        prob += q[i, k] == 1, f"fix_{k}"

    # 同堆栈先上后下：高层箱位置严格小于低层箱
    by_stack: dict[str, list[DischargeBox]] = {}
    for b in port_boxes:
        by_stack.setdefault(b.stack, []).append(b)
    index_of = {b.no: i for i, b in enumerate(port_boxes)}
    for stack_boxes in by_stack.values():
        stack_boxes.sort(key=lambda b: b.tier)
        for lower, upper in zip(stack_boxes, stack_boxes[1:]):
            iu, il = index_of[upper.no], index_of[lower.no]
            prob += (
                pulp.lpSum(k * q[iu, k] for k in range(m)) + 1
                <= pulp.lpSum(k * q[il, k] for k in range(m)),
                f"topdown_{upper.no}_{lower.no}",
            )

    # 每卸一箱（t=1..m）后剩余力矩落入该港区间。
    # 容差只依据实际力矩量级（不用区间界值，避免一侧给很宽的界时容差被放大，
    # 从而错误放宽另一侧的紧约束），另加小的绝对容差。
    tol = EPS * max(1.0, abs(moment_start)) + 1e-6
    for t in range(1, m + 1):
        removed = pulp.lpSum(
            b.moment * pulp.lpSum(q[i, k] for k in range(t))
            for i, b in enumerate(port_boxes)
        )
        remaining = moment_start - removed
        prob += remaining >= min_moment - tol, f"moment_min_{t}"
        prob += remaining <= max_moment + tol, f"moment_max_{t}"

    return prob, q


def _solve(prob: pulp.LpProblem) -> str:
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT, gapRel=0)
    prob.solve(solver)
    return pulp.LpStatus[prob.status]


def _solve_port(
    port: int,
    port_boxes: Sequence[DischargeBox],
    moment_start: float,
    bounds: tuple[float, float],
) -> list[DischargeBox]:
    """求单港字典序最小卸箱序列；不可行/无法判定时抛异常。"""
    lo, hi = bounds
    m = len(port_boxes)

    # 第一阶段：可行性
    prob1, q1 = _build_port_model(port_boxes, moment_start, lo, hi)
    status = _solve(prob1)
    if status == "Infeasible":
        raise InfeasibleDischarge(
            f"卸港 {port} 不存在满足同堆栈先上后下且每卸一箱后剩余总力矩均落在 "
            f"闭区间 [{lo}, {hi}] 内的卸箱顺序"
        )
    if status != "Optimal":
        raise InfeasibleDischarge(
            f"求解器在 {TIME_LIMIT}s 内无法判定卸港 {port} 的卸箱顺序"
            f"（状态：{status}）",
            undetermined=True,
        )

    # 第二阶段：按卸箱位置从前到后分块，最小化位置上箱号升序排名的基数编码，
    # 与港内箱号序列字典序最小严格等价（做法同配载/排程求解器）。
    digits = max(1, len(str(m)))
    chunk = max(1, 13 // digits)
    fixed_positions: dict[int, str] = {}
    last_q = q1
    for start in range(0, m, chunk):
        block = list(range(start, min(start + chunk, m)))
        prob, q = _build_port_model(
            port_boxes, moment_start, lo, hi,
            fixed_positions=fixed_positions, lex_positions=block,
        )
        bstatus = _solve(prob)
        if bstatus != "Optimal":
            raise InfeasibleDischarge(
                f"求解器在 {TIME_LIMIT}s 内无法完成卸港 {port} 的字典序优选"
                f"（状态：{bstatus}）",
                undetermined=True,
            )
        for k in block:
            for i, b in enumerate(port_boxes):
                v = pulp.value(q[i, k])
                if v is not None and v > 0.5:
                    fixed_positions[k] = b.no
                    break
        last_q = q

    order: list[DischargeBox] = []
    for k in range(m):
        for i, b in enumerate(port_boxes):
            v = pulp.value(last_q[i, k])
            if v is not None and v > 0.5:
                order.append(b)
                break
        else:  # pragma: no cover - 正常求解下不会发生
            raise InfeasibleDischarge(
                f"卸港 {port} 求解结果解析失败（位置 {k} 无箱）", undetermined=True
            )
    return order


def solve_discharge(
    boxes: Sequence[DischargeBox],
    windows: Mapping[int, tuple[float, float]],
) -> DischargeResult:
    """求全程逐港卸箱顺序。

    ``windows`` 的键必须与箱清单中出现的全部卸港序号完全一致（由业务层校验）。
    """
    by_port: dict[int, list[DischargeBox]] = {}
    for b in boxes:
        by_port.setdefault(b.port, []).append(b)

    _cross_port_blocking(boxes)

    total_moment = sum(b.moment for b in boxes)
    steps: list[DischargeStep] = []
    remaining_moment = total_moment
    step_no = 0

    for port in sorted(by_port):
        port_boxes = by_port[port]
        ordered = _solve_port(port, port_boxes, remaining_moment, windows[port])
        for b in ordered:
            remaining_moment -= b.moment
            step_no += 1
            steps.append(
                DischargeStep(
                    step_no=step_no,
                    discharge_order=port,
                    container_no=b.no,
                    remaining_moment=round(remaining_moment, 6),
                )
            )

    return DischargeResult(steps=steps)
