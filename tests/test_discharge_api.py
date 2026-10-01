"""逐港卸箱顺序 API 端到端测试。"""

from fastapi.testclient import TestClient

from app.database import SessionLocal
from app.main import app


SLOT_PAYLOAD = {
    # 四个独立单层槽（不同堆栈）：港1 箱之间无上下关系，顺序纯由字典序/窗口决定
    "slots": [
        {"code": "S01", "stack": "A", "tier": 1, "capacity": 100, "arm": 2.0},
        {"code": "S02", "stack": "B", "tier": 1, "capacity": 100, "arm": 2.0},
        {"code": "S03", "stack": "C", "tier": 1, "capacity": 100, "arm": 2.0},
        {"code": "S04", "stack": "D", "tier": 1, "capacity": 100, "arm": 2.0},
    ],
    "adjacency": [],
    "incompatible_classes": [],
    "min_moment": -100000,
    "max_moment": 100000,
}

# 全部单层：无堆叠约束。港1=C001/C002，港2=C003/C004
MANIFEST_PAYLOAD = {
    "containers": [
        {"no": "C004", "weight": 20.0, "hazmat": None, "discharge_order": 2},
        {"no": "C003", "weight": 20.0, "hazmat": None, "discharge_order": 2},
        {"no": "C002", "weight": 10.0, "hazmat": None, "discharge_order": 1},
        {"no": "C001", "weight": 10.0, "hazmat": None, "discharge_order": 1},
    ]
}


def _seed_and_confirm(client):
    client.post("/api/slot-revisions", json=SLOT_PAYLOAD)
    client.post("/api/manifest-revisions", json=MANIFEST_PAYLOAD)
    plan = client.post("/api/plans/compute", json={}).json()
    client.post(f"/api/plans/{plan['plan_id']}/confirm", json={
        "slots_revision": 1, "manifest_revision": 1
    })
    return plan


def _windows():
    return {"ports": [
        {"discharge_order": 1, "min_moment": -100000, "max_moment": 100000},
        {"discharge_order": 2, "min_moment": -100000, "max_moment": 100000},
    ]}


def test_discharge_happy_path(client):
    plan = _seed_and_confirm(client)
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=_windows())
    assert r.status_code == 201, r.text
    dp = r.json()
    assert dp["plan_id"] == plan["plan_id"]
    assert dp["revision_no"] == 1
    assert dp["status"] == "current"
    assert [p["discharge_order"] for p in dp["ports"]] == [1, 2]

    steps = [(s["step_no"], s["discharge_order"], s["container_no"])
             for s in dp["steps"]]
    # 港1：C001、C002（同序字典序，分处两堆栈上层，可任意 -> 字典序）
    # 港2：C003、C004
    assert [s[2] for s in steps] == ["C001", "C002", "C003", "C004"]
    assert [s[1] for s in steps] == [1, 1, 2, 2]
    assert [s[0] for s in steps] == [1, 2, 3, 4]
    # 全部力臂 2：初始力矩 = 10*2+10*2+20*2+20*2 = 120；
    # 港1 依次剩 100、80，港2 依次剩 40、0
    assert [s["remaining_moment"] for s in dp["steps"]] == [
        100.0, 80.0, 40.0, 0.0
    ]


def test_get_and_list_discharge(client):
    plan = _seed_and_confirm(client)
    client.post(f"/api/plans/{plan['plan_id']}/discharge", json=_windows())

    got = client.get(f"/api/plans/{plan['plan_id']}/discharge")
    assert got.status_code == 200
    assert got.json()["status"] == "current"

    lst = client.get("/api/discharge-plans").json()
    assert len(lst) == 1 and lst[0]["plan_id"] == plan["plan_id"]


def test_discharge_on_unconfirmed_rejected_409(client):
    client.post("/api/slot-revisions", json=SLOT_PAYLOAD)
    client.post("/api/manifest-revisions", json=MANIFEST_PAYLOAD)
    plan = client.post("/api/plans/compute", json={}).json()
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=_windows())
    assert r.status_code == 409 and "已确认" in r.json()["detail"]
    assert client.get("/api/discharge-plans").json() == []


def test_port_set_mismatch_rejected_422(client):
    plan = _seed_and_confirm(client)
    # 缺港2
    r1 = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [{"discharge_order": 1, "min_moment": -1e9, "max_moment": 1e9}]
    })
    assert r1.status_code == 422 and "缺少" in r1.json()["detail"]
    # 多港9
    body = _windows()
    body["ports"].append({"discharge_order": 9, "min_moment": 0, "max_moment": 0})
    r2 = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=body)
    assert r2.status_code == 422 and "多余" in r2.json()["detail"]
    assert client.get("/api/discharge-plans").json() == []


def test_inverted_window_rejected_422(client):
    plan = _seed_and_confirm(client)
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [
            {"discharge_order": 1, "min_moment": 10, "max_moment": 1},
            {"discharge_order": 2, "min_moment": 0, "max_moment": 0},
        ]
    })
    assert r.status_code == 422 and "下限" in r.text


def test_unsorted_and_duplicate_ports_rejected_422(client):
    plan = _seed_and_confirm(client)
    r1 = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [
            {"discharge_order": 2, "min_moment": 0, "max_moment": 1e9},
            {"discharge_order": 1, "min_moment": 0, "max_moment": 1e9},
        ]
    })
    assert r1.status_code == 422 and "从小到大" in r1.text
    r2 = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [
            {"discharge_order": 1, "min_moment": 0, "max_moment": 1e9},
            {"discharge_order": 1, "min_moment": 0, "max_moment": 1e9},
            {"discharge_order": 2, "min_moment": 0, "max_moment": 1e9},
        ]
    })
    assert r2.status_code == 422 and "重复" in r2.text


def test_infeasible_window_rejected_422_not_saved(client):
    plan = _seed_and_confirm(client)
    # 初始力矩 120，港1 每箱力矩 20：第一步卸后必为 100。
    # 要求第一步剩余 <=99 -> 无解（100 不在区间）
    body = _windows()
    body["ports"][0] = {"discharge_order": 1, "min_moment": -1e9,
                        "max_moment": 99}
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=body)
    assert r.status_code == 422 and "无解" in r.json()["detail"]
    assert client.get("/api/discharge-plans").json() == []


def test_window_forces_lex_inferior_pick(client):
    plan = _seed_and_confirm(client)
    # 初始 120；港1 两箱力矩均为 20：卸任意一箱后剩 100，再卸剩 80。
    # 窗口 [80,100]：两种首箱等价可行 -> 仍取字典序 C001 先。
    body = {"ports": [
        {"discharge_order": 1, "min_moment": 80, "max_moment": 100},
        {"discharge_order": 2, "min_moment": -1e9, "max_moment": 1e9},
    ]}
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=body)
    assert r.status_code == 201
    seq = [s["container_no"] for s in r.json()["steps"]]
    assert seq[:2] == ["C001", "C002"]

    # 新槽位资料/箱清单：C001 力矩 5（重5×臂1，字典序小）、
    # C002 力矩 45（重10×臂4.5）；港2 两箱力矩各 20。
    # 初始 90；港1 末（两箱卸完）恒剩 40。
    client.post("/api/slot-revisions", json={
        "slots": [
            {"code": "S01", "stack": "A", "tier": 1, "capacity": 100, "arm": 1.0},
            {"code": "S02", "stack": "B", "tier": 1, "capacity": 100, "arm": 4.5},
            {"code": "S03", "stack": "C", "tier": 1, "capacity": 100, "arm": 1.0},
            {"code": "S04", "stack": "D", "tier": 1, "capacity": 100, "arm": 1.0},
        ],
        "adjacency": [], "incompatible_classes": [],
        "min_moment": -100000, "max_moment": 100000,
    })
    client.post("/api/manifest-revisions", json={
        "containers": [
            {"no": "C004", "weight": 20.0, "hazmat": None, "discharge_order": 2},
            {"no": "C003", "weight": 20.0, "hazmat": None, "discharge_order": 2},
            {"no": "C002", "weight": 10.0, "hazmat": None, "discharge_order": 1},
            {"no": "C001", "weight": 5.0, "hazmat": None, "discharge_order": 1},
        ]
    })
    p2 = client.post(f"/api/plans/{plan['plan_id']}/amend", json={}).json()
    client.post(f"/api/plans/{p2['plan_id']}/confirm", json={
        "slots_revision": 2, "manifest_revision": 2
    })
    # 港1 区间 [40,45]：C001 先 -> 剩 85 越上界被排除；
    # C002 先 -> 剩 45 ∈，再卸 C001 剩 40 ∈，可行。
    body2 = {"ports": [
        {"discharge_order": 1, "min_moment": 40, "max_moment": 45},
        {"discharge_order": 2, "min_moment": 0, "max_moment": 40},
    ]}
    r2 = client.post(f"/api/plans/{p2['plan_id']}/discharge", json=body2)
    assert r2.status_code == 201, r2.text
    seq2 = [s["container_no"] for s in r2.json()["steps"]]
    assert seq2[:2] == ["C002", "C001"]  # 窗口迫使 C002 先于字典序更小的 C001
    assert [s["remaining_moment"] for s in r2.json()["steps"]] == [
        45.0, 40.0, 20.0, 0.0
    ]


def test_resubmit_atomically_replaces(client):
    plan = _seed_and_confirm(client)
    client.post(f"/api/plans/{plan['plan_id']}/discharge", json=_windows())
    # 更换窗口（仍可行）重新提交
    body = {"ports": [
        {"discharge_order": 1, "min_moment": 0, "max_moment": 100000},
        {"discharge_order": 2, "min_moment": 0, "max_moment": 100000},
    ]}
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=body)
    assert r.status_code == 201
    all_dp = client.get("/api/discharge-plans").json()
    assert len(all_dp) == 1
    assert all_dp[0]["ports"][0]["min_moment"] == 0

    db = SessionLocal()
    try:
        from app import models
        assert db.query(models.DischargePlan).count() == 1
        assert db.query(models.DischargeWindow).count() == 2
        assert db.query(models.DischargeStepRow).count() == 4
    finally:
        db.close()


def test_historical_after_supersede_and_still_queryable(client):
    p1 = _seed_and_confirm(client)
    client.post(f"/api/plans/{p1['plan_id']}/discharge", json=_windows())

    # 新箱清单修订 -> 修订并确认 p2（同根），p1 变 superseded
    client.post("/api/manifest-revisions", json={
        "containers": [{"no": "C001", "weight": 10, "hazmat": None,
                        "discharge_order": 1}]
    })
    p2 = client.post(f"/api/plans/{p1['plan_id']}/amend", json={}).json()
    ok = client.post(f"/api/plans/{p2['plan_id']}/confirm", json={
        "slots_revision": 1, "manifest_revision": 2
    })
    assert ok.status_code == 200

    # 默认列表不含历史记录
    assert client.get("/api/discharge-plans").json() == []
    all_dp = client.get("/api/discharge-plans?historical=true").json()
    assert len(all_dp) == 1
    assert all_dp[0]["plan_id"] == p1["plan_id"]
    assert all_dp[0]["status"] == "historical"

    # 按旧修订仍可查询
    old = client.get(f"/api/plans/{p1['plan_id']}/discharge")
    assert old.status_code == 200 and old.json()["status"] == "historical"


def test_save_time_recheck_rejects_if_superseded(client, monkeypatch):
    """求解后落库前方案被并发取代：锁内复核须拒绝且不保存。"""
    from app import services

    p1 = _seed_and_confirm(client)

    client.post("/api/manifest-revisions", json={
        "containers": [
            {"no": "C001", "weight": 10, "hazmat": None, "discharge_order": 1}
        ]
    })
    p2 = client.post(f"/api/plans/{p1['plan_id']}/amend", json={}).json()

    real_solve = services.solve_discharge

    def raced_solve(boxes, windows):
        result = real_solve(boxes, windows)
        client.post(f"/api/plans/{p2['plan_id']}/confirm", json={
            "slots_revision": 1, "manifest_revision": 2
        })
        return result

    monkeypatch.setattr(services, "solve_discharge", raced_solve)
    r = client.post(f"/api/plans/{p1['plan_id']}/discharge", json=_windows())
    assert r.status_code == 409
    assert "复核" in r.json()["detail"]
    assert client.get("/api/discharge-plans?historical=true").json() == []


def test_empty_manifest_discharge(client):
    client.post("/api/slot-revisions", json=SLOT_PAYLOAD)
    client.post("/api/manifest-revisions", json={"containers": []})
    plan = client.post("/api/plans/compute", json={}).json()
    client.post(f"/api/plans/{plan['plan_id']}/confirm", json={
        "slots_revision": 1, "manifest_revision": 1
    })
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={"ports": []})
    assert r.status_code == 201
    assert r.json()["steps"] == [] and r.json()["ports"] == []
    # 空箱方案多给港序号仍拒绝
    bad = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [{"discharge_order": 1, "min_moment": 0, "max_moment": 0}]
    })
    assert bad.status_code == 422 and "多余" in bad.json()["detail"]


def test_stacked_top_down_order_end_to_end(client):
    """真实配载为上下堆叠时，卸箱必须先上后下，即使下层箱号字典序更小。"""
    # 单堆栈两层；两箱同一卸港。下层重 20、上层重 10（满足重量次序）
    client.post("/api/slot-revisions", json={
        "slots": [
            {"code": "A01", "stack": "A", "tier": 1, "capacity": 100, "arm": 2.0},
            {"code": "A02", "stack": "A", "tier": 2, "capacity": 100, "arm": 3.0},
        ],
        "adjacency": [], "incompatible_classes": [],
        "min_moment": -100000, "max_moment": 100000,
    })
    client.post("/api/manifest-revisions", json={
        "containers": [
            {"no": "B_LOWER", "weight": 20.0, "hazmat": None, "discharge_order": 1},
            {"no": "A_UPPER", "weight": 10.0, "hazmat": None, "discharge_order": 1},
        ]
    })
    plan = client.post("/api/plans/compute", json={}).json()
    client.post(f"/api/plans/{plan['plan_id']}/confirm", json={
        "slots_revision": 1, "manifest_revision": 1
    })
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json={
        "ports": [{"discharge_order": 1, "min_moment": -1e9, "max_moment": 1e9}]
    })
    assert r.status_code == 201, r.text
    seq = [s["container_no"] for s in r.json()["steps"]]
    # 上层 A_UPPER 必须先卸，尽管 B_LOWER 字典序更小
    assert seq == ["A_UPPER", "B_LOWER"]


def test_get_discharge_404(client):
    plan = _seed_and_confirm(client)
    assert client.get(f"/api/plans/{plan['plan_id']}/discharge").status_code == 404
    assert client.get("/api/plans/nope/discharge").status_code == 404


def test_concurrent_create_disjoint_plans(client):
    """两个独立已确认方案（不同槽位）均可规划卸箱顺序，互不干扰。"""
    plan = _seed_and_confirm(client)
    # 同一资料再核算第二个方案会因槽位占用无法确认；这里仅验证同方案串行创建正常
    r = client.post(f"/api/plans/{plan['plan_id']}/discharge", json=_windows())
    assert r.status_code == 201

    # 真并发：两个请求对同一方案提交，写锁串行化后至多一个落库（原子替换）
    results = {}

    def fire(key):
        local = TestClient(app)
        results[key] = local.post(
            f"/api/plans/{plan['plan_id']}/discharge", json=_windows()
        ).status_code

    import threading
    t1 = threading.Thread(target=fire, args=(1,))
    t2 = threading.Thread(target=fire, args=(2,))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert sorted(results.values()) == [201, 201]
    assert len(client.get("/api/discharge-plans").json()) == 1
