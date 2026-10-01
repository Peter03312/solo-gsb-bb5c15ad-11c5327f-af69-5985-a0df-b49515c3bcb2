"""逐港卸箱顺序求解器测试（含与穷举交叉验证）。"""

import itertools

import pytest

from app.discharge import DischargeBox, InfeasibleDischarge, solve_discharge


def _box(no, port, stack, tier, weight=10.0, arm=2.0):
    return DischargeBox(no=no, weight=weight, arm=arm, stack=stack,
                        tier=tier, port=port)


def _brute_force_min(boxes, windows):
    """穷举所有合法逐港顺序，返回全程箱号序列字典序最小者，无可行则 None。"""
    by_port = {}
    for b in boxes:
        by_port.setdefault(b.port, []).append(b)

    best = None

    def feasible_port(order, moment_start, lo, hi):
        m = moment_start
        for b in order:
            m -= b.moment
            if not (lo - 1e-6 <= m <= hi + 1e-6):
                return False
        return True

    def topdown_ok(order):
        # 同堆栈上层（tier 大）必须先于下层（tier 小）出现
        pos = {b.no: i for i, b in enumerate(order)}
        for i, a in enumerate(order):
            for b2 in order[i + 1:]:
                if a.stack == b2.stack and a.tier < b2.tier:
                    if pos[a.no] < pos[b2.no]:
                        return False
        return True

    def rec(port_idx, moment, prefix):
        nonlocal best
        ports = sorted(by_port)
        if port_idx == len(ports):
            seq = tuple(b.no for b in prefix)
            if best is None or seq < best:
                best = seq
            return
        port = ports[port_idx]
        lo, hi = windows[port]
        for p in itertools.permutations(by_port[port]):
            order = list(p)
            if not topdown_ok(order):
                continue
            if feasible_port(order, moment, lo, hi):
                rec(port_idx + 1,
                    moment - sum(b.moment for b in order),
                    prefix + order)

    rec(0, sum(b.moment for b in boxes), [])
    return best


def test_simple_single_port_lex_min():
    # 两箱不同堆栈，无先后限制：全程字典序 C001 先
    boxes = [_box("C002", 1, "B", 1), _box("C001", 1, "A", 1)]
    r = solve_discharge(boxes, {1: (-1e9, 1e9)})
    assert [s.container_no for s in r.steps] == ["C001", "C002"]
    # 初始力矩 = 10*2+10*2=40；C001 后 20；全卸 0
    assert [s.remaining_moment for s in r.steps] == [20.0, 0.0]
    assert [s.discharge_order for s in r.steps] == [1, 1]
    assert [s.step_no for s in r.steps] == [1, 2]


def test_top_down_constraint_forces_order():
    # 同堆栈：C001 在 tier2（上层），C002 在 tier1（下层）。
    # 虽 C001 字典序本就小；反过来再验强制：
    boxes = [_box("A_TOP", 1, "A", 2), _box("B_BOT", 1, "A", 1)]
    r = solve_discharge(boxes, {1: (-1e9, 1e9)})
    assert [s.container_no for s in r.steps] == ["A_TOP", "B_BOT"]

    # 下层箱号字典序更小，仍必须后卸
    boxes = [_box("A_BOT", 1, "A", 1), _box("B_TOP", 1, "A", 2)]
    r = solve_discharge(boxes, {1: (-1e9, 1e9)})
    assert [s.container_no for s in r.steps] == ["B_TOP", "A_BOT"]


def test_port_order_strict_and_multi_port():
    # 港1 的箱在同堆栈上层，港2 在下层
    boxes = [
        _box("P2B", 2, "A", 1, weight=5.0, arm=2.0),
        _box("P1A", 1, "A", 2, weight=5.0, arm=3.0),
    ]
    r = solve_discharge(boxes, {1: (-1e9, 1e9), 2: (-1e9, 1e9)})
    seq = [(s.discharge_order, s.container_no) for s in r.steps]
    assert seq == [(1, "P1A"), (2, "P2B")]
    # 初始 5*3+5*2=25；卸 P1A(-15) -> 10；卸 P2B(-10) -> 0
    assert [s.remaining_moment for s in r.steps] == [10.0, 0.0]


def test_moment_window_infeasible():
    # 单箱，力矩 10*2=20。窗口不允许卸后值 0
    boxes = [_box("C001", 1, "A", 1)]
    with pytest.raises(InfeasibleDischarge):
        solve_discharge(boxes, {1: (100, 200)})


def test_moment_window_steers_order_lex_violation():
    """窗口迫使全程序列采用字典序劣后选择（首箱）。"""
    # 港1 两箱分处不同堆栈上层，港2 两箱在其正下方：
    #   C001 力矩 5（字典序小），C002 力矩 45（字典序大）；港2 恒重 40。
    # 初始 90；港1 末步（C001、C002 均卸完）恒剩 40。
    boxes = [
        _box("C001", 1, "A", 2, weight=5.0, arm=1.0),
        _box("C002", 1, "B", 2, weight=45.0, arm=1.0),
        _box("L001", 2, "A", 1, weight=20.0, arm=1.0),
        _box("L002", 2, "B", 1, weight=20.0, arm=1.0),
    ]
    # 宽区间：字典序 C001 先（第一步剩 85），再 C002（剩 40）
    r = solve_discharge(boxes, {1: (40, 90), 2: (0, 40)})
    assert [s.container_no for s in r.steps] == ["C001", "C002", "L001", "L002"]

    # 港1 区间 [40,45]：C001 先 -> 第一步剩 85，越上界，被排除；
    # C002 先 -> 第一步剩 45 ∈，第二步（末步）剩 40 ∈，可行。
    r = solve_discharge(boxes, {1: (40, 45), 2: (0, 40)})
    seq = [s.container_no for s in r.steps]
    assert seq[:2] == ["C002", "C001"]
    assert [s.remaining_moment for s in r.steps] == [45.0, 40.0, 20.0, 0.0]


def test_boundary_closed_interval_inclusive():
    # 单箱 20；区间含端点 0 与 20? 卸后=0，[0,0] 可行
    boxes = [_box("C001", 1, "A", 1)]
    r = solve_discharge(boxes, {1: (0.0, 0.0)})
    assert r.steps[0].remaining_moment == 0.0


def test_empty_plan():
    r = solve_discharge([], {})
    assert r.steps == []


def test_cross_port_blocking_detected():
    # 港1 箱在下层、港2 箱在上层：港1 永远卸不出
    boxes = [
        _box("EARLY", 1, "A", 1),
        _box("LATE", 2, "A", 2),
    ]
    with pytest.raises(InfeasibleDischarge):
        solve_discharge(boxes, {1: (-1e9, 1e9), 2: (-1e9, 1e9)})


def test_three_box_chain_window_each_step():
    # 同堆栈三层 + 额外箱，每步窗口约束
    boxes = [
        _box("T1", 1, "A", 1, weight=4.0, arm=1.0),  # 力矩 4，最下
        _box("T2", 1, "A", 2, weight=3.0, arm=2.0),  # 6
        _box("T3", 1, "A", 3, weight=2.0, arm=3.0),  # 6，最上
    ]
    r = solve_discharge(boxes, {1: (-100, 100)})
    # 只能自顶向下
    assert [s.container_no for s in r.steps] == ["T3", "T2", "T1"]
    assert [s.remaining_moment for s in r.steps] == [10.0, 4.0, 0.0]


@pytest.mark.parametrize("seed", range(40))
def test_brute_force_cross_check(seed):
    """随机小实例：MILP 结果与穷举的字典序最小合法序列一致。"""
    import random

    rng = random.Random(seed)
    boxes = []
    for st in ["A", "B"]:
        height = rng.randint(1, 3)
        # 自底向上的卸港序号必须单调不减（先卸在上），故取排序后的随机港号
        ports = sorted((rng.choice([1, 2]) for _ in range(height)), reverse=True)
        for tier, port in enumerate(ports, start=1):
            boxes.append(
                _box(f"{st}{tier}", port, st, tier,
                     weight=float(rng.randint(1, 5)),
                     arm=float(rng.randint(1, 3)))
            )

    present_ports = sorted({b.port for b in boxes})
    # 随机窗口：在“单箱力矩/0”网格值中取闭区间，制造可行/不可行分支
    moments = sorted({0.0} | {b.moment for b in boxes})
    lo, hi = sorted(rng.sample(moments, 2)) if len(moments) >= 2 else (0.0, 0.0)
    windows = {p: (lo, hi) for p in present_ports}

    expected = _brute_force_min(boxes, windows)
    if expected is None:
        with pytest.raises(InfeasibleDischarge):
            solve_discharge(boxes, windows)
    else:
        r = solve_discharge(boxes, windows)
        assert tuple(s.container_no for s in r.steps) == expected
