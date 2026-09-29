"""生成 N4 评测集：data/eval/eval_set.json

覆盖四类用例（共 79 条）：
  precise_query  精确查询（本人订单/物流，含同义改写）
  authz          越权与无效实体（他人订单 403、不存在订单 404）
  knowledge      知识问答（FAQ 原句 + 同义改写）
  multi_turn     多轮追问（指代消解：状态追问 / 政策追问）
  invalid        异常与无效输入（鉴权失败 401、参数非法 422、闲聊、乱码、低置信度转人工）

用法：.venv\\Scripts\\python.exe -m scripts.build_eval_set
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 与 scripts/init_db.py 的种子数据保持一致：order_id = ORD202401{i:05d}
# (order_id, 归属用户, 商品, 状态, 状态文案, 物流单号)
ORDERS = [
    ("ORD20240100001", "user_001", "无线蓝牙耳机 Pro", "delivered", "已签收", "SF123456789012"),
    ("ORD20240100002", "user_001", "87键机械键盘", "shipped", "已发货", "SF123456789034"),
    ("ORD20240100003", "user_001", "20000mAh 便携充电宝", "refunded", "已退款", None),
    ("ORD20240100004", "user_002", "27寸 4K 显示器", "shipped", "已发货", "YT987654321098"),
    ("ORD20240100005", "user_002", "智能手表 S2", "paid", "已支付", None),
    ("ORD20240100006", "user_002", "纯棉基础款T恤", "delivered", "已签收", "YT987654321076"),
    ("ORD20240100007", "user_003", "运动跑鞋", "pending", "待处理", None),
    ("ORD20240100008", "user_003", "无线蓝牙耳机 Pro", "cancelled", "已取消", None),
]

# 知识问答：问题 + 期望答案中应出现的关键词（任一命中即可，规避 LLM 改写措辞差异）
KNOWLEDGE = [
    ("七天无理由退货的条件是什么？", ["7天", "七天"]),
    ("怎么申请退货？", ["售后", "我的订单"]),
    ("退货的运费谁出？", ["买家承担", "运费险", "我们承担"]),
    ("退货退款多久能到账？", ["工作日"]),
    ("可以换货吗？怎么换？", ["换货"]),
    ("商品有质量问题怎么处理？", ["质量问题"]),
    ("退款会退到哪里？", ["原路"]),
    ("已经发货了还能取消订单吗？", ["拦截", "拒收"]),
    ("退货商品寄回地址是什么？", ["退货码", "售后页面", "寄回地址"]),
    ("支持上门取件退换货吗？", ["上门取件"]),
    ("订单发货后多久能到？", ["2-4天", "48小时", "2~4天"]),
    ("可以修改收货地址吗？", ["未发货", "修改"]),
    ("怎么查看物流进度？", ["查看物流", "物流轨迹", "物流单号"]),
    ("快递显示签收但我没收到怎么办？", ["客服", "快递柜", "驿站"]),
    ("支持哪些快递公司？", ["顺丰"]),
    ("可以指定快递公司吗？", ["不支持"]),
    ("发货后可以加急吗？", ["不支持"]),
    ("商品破损了怎么办？", ["拒收", "拍照"]),
    ("可以支持货到付款吗？", ["不支持"]),
    ("支付失败怎么办？", ["重试", "凭证"]),
    ("支持花呗/白条支付吗？", ["花呗"]),
    ("下单后可以改支付方式吗？", ["取消订单", "支付前"]),
    ("怎么开具发票？", ["电子发票", "税号"]),
    ("发票可以开公司抬头吗？", ["税号", "企业抬头", "公司名称"]),
    ("发票多久能开出来？", ["工作日"]),
    ("优惠券怎么领取和使用？", ["优惠券"]),
    ("优惠券过期了还能用吗？", ["过期"]),
    ("怎么成为会员？会员有什么权益？", ["会员"]),
    ("保修期是多久？", ["一年", "保修"]),
    ("如何修改绑定的手机号？", ["安全中心", "验证", "手机号"]),
]

# 同义改写（考察语义检索，问题与原句不重合）
KNOWLEDGE_PARAPHRASE = [
    ("东西买回来发现是坏的能修吗", ["保修", "维修", "质量"]),
    ("我买的东西有问题想退掉咋弄", ["售后", "退货", "退款"]),
    ("退的钱啥时候能打回我账户", ["工作日", "原路", "退款"]),
    ("能不能指定用顺丰给我发", ["不支持", "快递"]),
    ("能开电子发票不", ["发票"]),
    ("发到新疆一般几天到", ["偏远", "天", "快递"]),
]

# 多轮追问中的状态追问措辞（第一轮先问订单，第二轮考察指代消解）
FOLLOWUP_STATUS = ["那什么时候能到？", "它现在到哪了", "什么时候能到？", "那现在是什么状态？"]
FOLLOWUP_POLICY = ["那退货运费谁出？", "那还能换货吗？", "那退款多久到账？", "那保修多久？"]


def _turn(question, **expect):
    return {"question": question, "expect": expect}


def build() -> dict:
    cases = []

    # ---- A. 精确查询（本人订单 8 + 同义 2 + 物流 2 = 12）----
    for oid, owner, _product, _status, status_text, _tracking in ORDERS:
        cases.append({
            "id": f"precise_order_{oid[-3:]}",
            "category": "precise_query",
            "user_id": owner,
            "auth": "valid",
            "turns": [_turn(
                f"我的订单 {oid} 到哪里了？",
                status=200, intent="precise_query", source="database",
                must_contain=[oid, status_text],
            )],
        })
    cases.append({
        "id": "precise_paraphrase_01",
        "category": "precise_query",
        "user_id": "user_001",
        "auth": "valid",
        "turns": [_turn(
            "帮我查下订单 ORD20240100001 现在什么状态",
            status=200, intent="precise_query", source="database",
            must_contain=["ORD20240100001"],
        )],
    })
    cases.append({
        "id": "precise_paraphrase_02",
        "category": "precise_query",
        "user_id": "user_002",
        "auth": "valid",
        "turns": [_turn(
            "订单 ORD20240100004 的快递单号是多少",
            status=200, intent="precise_query", source="database",
            must_contain=["ORD20240100004"],
        )],
    })
    for tracking, owner, status_text in [
        ("SF123456789012", "user_001", "已签收"),
        ("YT987654321098", "user_002", "已发货"),
    ]:
        cases.append({
            "id": f"precise_tracking_{tracking[-4:]}",
            "category": "precise_query",
            "user_id": owner,
            "auth": "valid",
            "turns": [_turn(
                f"查询物流单号 {tracking} 的进度",
                status=200, intent="precise_query", source="database",
                must_contain=[tracking, status_text],
            )],
        })

    # ---- B. 越权与无效实体（6 + 3 = 9）----
    for oid, owner, _p, _s, _st, _t in ORDERS[:5]:
        other = "user_003" if owner != "user_003" else "user_001"
        cases.append({
            "id": f"authz_order_{oid[-3:]}",
            "category": "authz",
            "user_id": other,
            "auth": "valid",
            "turns": [_turn(
                f"我的订单 {oid} 到哪里了？",
                status=403, code=403,
            )],
        })
    cases.append({
        "id": "authz_tracking_01",
        "category": "authz",
        "user_id": "user_003",
        "auth": "valid",
        "turns": [_turn("查询物流单号 SF123456789012 的进度", status=403, code=403)],
    })
    cases.append({
        "id": "notfound_order_01",
        "category": "authz",
        "user_id": "user_001",
        "auth": "valid",
        "turns": [_turn("我的订单 ORD99999999999 到哪了", status=404, code=404)],
    })
    cases.append({
        "id": "notfound_order_02",
        "category": "authz",
        "user_id": "user_001",
        "auth": "valid",
        "turns": [_turn("帮我查订单 ORD20240199999 的状态", status=404, code=404)],
    })
    cases.append({
        "id": "notfound_tracking_01",
        "category": "authz",
        "user_id": "user_001",
        "auth": "valid",
        "turns": [_turn("查询物流单号 SF000000000000", status=404, code=404)],
    })

    # ---- C. 知识问答（30 + 6 = 36）----
    for i, (q, keywords) in enumerate(KNOWLEDGE, 1):
        cases.append({
            "id": f"knowledge_{i:02d}",
            "category": "knowledge",
            "user_id": "user_001",
            "auth": "valid",
            "turns": [_turn(q, status=200, intent="knowledge", source="rag", must_contain_any=keywords)],
        })
    for i, (q, keywords) in enumerate(KNOWLEDGE_PARAPHRASE, 1):
        cases.append({
            "id": f"knowledge_synonym_{i:02d}",
            "category": "knowledge",
            "user_id": "user_001",
            "auth": "valid",
            "turns": [_turn(q, status=200, source="rag", must_contain_any=keywords)],
        })

    # ---- D. 多轮追问（8 状态追问 + 4 政策追问 = 12）----
    combo = [(ORDERS[0], 0), (ORDERS[0], 1), (ORDERS[1], 2), (ORDERS[1], 3),
             (ORDERS[3], 0), (ORDERS[3], 2), (ORDERS[5], 1), (ORDERS[6], 3)]
    for idx, (order, phrase_idx) in enumerate(combo, 1):
        oid, owner = order[0], order[1]
        cases.append({
            "id": f"multi_status_{idx:02d}",
            "category": "multi_turn",
            "user_id": owner,
            "auth": "valid",
            "turns": [
                _turn(f"我的订单 {oid} 到哪里了？", status=200, intent="precise_query", source="database"),
                _turn(
                    FOLLOWUP_STATUS[phrase_idx],
                    status=200, intent="precise_query", source="database",
                    rewritten_contains=oid,
                ),
            ],
        })
    for idx, (order, phrase) in enumerate(zip(ORDERS[:4], FOLLOWUP_POLICY), 1):
        oid, owner = order[0], order[1]
        cases.append({
            "id": f"multi_policy_{idx:02d}",
            "category": "multi_turn",
            "user_id": owner,
            "auth": "valid",
            "turns": [
                _turn(f"我的订单 {oid} 到哪里了？", status=200, source="database"),
                _turn(
                    phrase,
                    status=200, intent="knowledge", source="rag",
                    # 政策类问题必须走 FAQ，且改写结果不得注入订单号（政策护栏）
                    rewritten_not_contains=oid,
                ),
            ],
        })

    # ---- E. 异常与无效输入（10）----
    cases.append({"id": "invalid_auth_01", "category": "invalid", "user_id": "user_001",
                  "auth": "none", "turns": [_turn("退货运费谁出？", status=401, code=401)]})
    cases.append({"id": "invalid_auth_02", "category": "invalid", "user_id": "user_001",
                  "auth": "invalid", "turns": [_turn("我的订单 ORD20240100001 到哪了？", status=401, code=401)]})
    cases.append({"id": "invalid_auth_03", "category": "invalid", "user_id": "user_001",
                  "auth": "malformed", "turns": [_turn("你好", status=401, code=401)]})
    cases.append({"id": "invalid_param_01", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("退" * 600, status=422)]})
    cases.append({"id": "invalid_param_02", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("查询物流单号 SF123456789012 的进度", status=422)], "broken": "no_session"})
    cases.append({"id": "invalid_chitchat_01", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("你好", status=200)]})
    cases.append({"id": "invalid_chitchat_02", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("谢谢", status=200)]})
    cases.append({"id": "invalid_gibberish_01", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("asdfghjkl zxcvbnm", status=200)]})
    cases.append({"id": "invalid_gibberish_02", "category": "invalid", "user_id": "user_001",
                  "auth": "valid", "turns": [_turn("？？？？？？？？", status=200)]})
    cases.append({"id": "invalid_lowconf_handoff_01", "category": "invalid", "user_id": "user_003",
                  "auth": "valid", "turns": [_turn("贵司的量子传送门保修几年", status=200, handoff_ticket=True)]})

    meta = {
        "version": "1.0",
        "generated_by": "scripts/build_eval_set.py",
        "notes": "expect.must_contain=全部命中；must_contain_any=任一命中；rewritten_contains/not_contains 校验指代消解与政策护栏",
        "total_cases": len(cases),
    }
    return {"meta": meta, "cases": cases}


def main() -> None:
    data = build()
    out = ROOT / "data" / "eval" / "eval_set.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from collections import Counter

    dist = Counter(c["category"] for c in data["cases"])
    turns = sum(len(c["turns"]) for c in data["cases"])
    print(f"[ok] 评测集已生成: {out}")
    print(f"[ok] 用例 {data['meta']['total_cases']} 条 / 请求轮次 {turns} 次")
    for cat, n in sorted(dist.items()):
        print(f"     {cat:<14} {n} 条")


if __name__ == "__main__":
    main()
