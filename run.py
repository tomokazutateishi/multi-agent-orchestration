"""
マルチエージェント・オーケストレーション（4役構成・社長削除済み）
Documents/.env を読み込み、LangGraph で会話フローを実行する。
役割: 1) オーケストレーター/プランナー（orchestrator+marketer+planner）
      2) エグゼキューター＋ドメインエキスパート（技術）(engineer)
      3) クリティック/レビュアー＋エグゼキューター（検証・再現）(reviewer)
※ ドメインエキスパート（UI/UX）はワークフロー・手動で参照（ROLE_DESIGNER）。
役割ごとのモデルは config.py で最適割り当て＋環境変数で上書き可能。
"""

import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

# Documents/.env を読み込む（このファイルが multi-agent-orchestration/ にある前提）
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# 役割ごとの LLM（config で最適割り当て。USE_OPENAI_ONLY=1 で全役 OpenAI）
from config import build_llm_registry

_llm_registry = build_llm_registry()
_llms = _llm_registry["by_role"]

# ========== 状態定義 ==========

# 議論を回す回数（「議論を回して」と依頼した時は常に3回）
DISCUSSION_ROUNDS = 3


class OrchestrationState(TypedDict, total=False):
    """オーケストレーション全体で共有する状態"""

    project_idea: str
    discussion_only: bool  # True のとき議論3周までで終了（オーケストレーター・エンジニア・レビュアーは実行しない）
    implement_only: bool  # True のとき オーケストレーター（タスク）から開始（戦略・マーケター・プランナー・議論はスキップ）
    orchestrator_direction: str
    marketer_output: str
    planner_plan: str
    orchestrator_discussion_react: str
    marketer_discussion_react: str
    planner_discussion_react: str
    orchestrator_discussion_2: str
    marketer_discussion_2: str
    planner_discussion_2: str
    orchestrator_discussion_3: str
    marketer_discussion_3: str
    planner_discussion_3: str
    pm_plan: str
    pm_task: str
    engineer_code: str
    review_result: str
    review_approved: bool
    rework_count: int
    skip_review: bool  # True のときレビュアーをスキップしエンジニア実装後にそのまま完了へ
    log: list[str]


# ========== プロンプト ==========
from prompts import (
    DISCUSSION_ORCHESTRATOR_R1,
    DISCUSSION_ORCHESTRATOR_R2,
    DISCUSSION_ORCHESTRATOR_R3,
    DISCUSSION_ORCHESTRATOR_R3_10IDEAS,
    DISCUSSION_MARKETER_R1,
    DISCUSSION_MARKETER_R2,
    DISCUSSION_MARKETER_R3,
    DISCUSSION_PLANNER_R1,
    DISCUSSION_PLANNER_R2,
    DISCUSSION_PLANNER_R3,
    DISCUSSION_PLANNER_R3_10IDEAS,
    ROLE_ORCHESTRATOR,
    ROLE_ORCHESTRATOR_PM,
    ROLE_ENGINEER,
    ROLE_MARKETER,
    ROLE_PLANNER,
    ROLE_REVIEWER,
)


def _log(state: OrchestrationState, role: str, text: str) -> dict:
    """ログを追加した更新を返す"""
    log = list(state.get("log") or [])
    log.append(f"[{role}] {text[:200]}{'...' if len(text) > 200 else ''}")
    return {"log": log}


# ========== ノード定義 ==========


def node_orchestrator(state: OrchestrationState) -> dict:
    """オーケストレーター: ユーザーの企画意図を戦略方針にまとめる"""
    idea = state.get("project_idea") or "（企画内容が未入力です）"
    msg = _llms["orchestrator"].invoke(
        [
            SystemMessage(content=ROLE_ORCHESTRATOR),
            HumanMessage(
                content=f"以下のプロジェクト企画について、戦略的な方針と優先順位を簡潔に示してください。"
                f"**アプリ（に近いもの）に対し課金することに躊躇がないユーザーをターゲット**にした前提で検討してください。\n\n【企画】\n{idea}"
            ),
        ]
    )
    direction = msg.content
    out = {"orchestrator_direction": direction, **_log(state, "Orchestrator", direction)}
    return out


def node_marketer(state: OrchestrationState) -> dict:
    """マーケター: オーケストレーター方針を踏まえ、市場・ユーザー分析とプロダクト要件の骨子を提供する"""
    direction = state.get("orchestrator_direction") or ""
    msg = _llms["marketer"].invoke(
        [
            SystemMessage(content=ROLE_MARKETER),
            HumanMessage(
                content=f"オーケストレーターの戦略方針:\n{direction}\n\n"
                "この方針に基づき、**6R・VALSのフレームワークを活用**して市場・ユーザー分析をまとめてください。"
                "**課金に躊躇しないユーザー**（VALSのInnovators, Achievers, Experiencers等）を明確にし、"
                "6R（市場規模・成長性・到達可能性・競合等）で評価した上で、プランナー・オーケストレーターが参照すべきマーケット要件を3〜5項目で示してください。"
            ),
        ]
    )
    output = msg.content
    out = {"marketer_output": output, **_log(state, "Marketer", output)}
    return out


def node_planner(state: OrchestrationState) -> dict:
    """プランナー: オーケストレーター方針とマーケター分析を踏まえ、プロジェクト計画を立案する"""
    direction = state.get("orchestrator_direction") or ""
    marketer_output = state.get("marketer_output") or ""
    msg = _llms["planner"].invoke(
        [
            SystemMessage(content=ROLE_PLANNER),
            HumanMessage(
                content=f"オーケストレーターの戦略方針:\n{direction}\n\n"
                f"マーケターの分析・要件:\n{marketer_output}\n\n"
                "上記を踏まえ、プロジェクト計画を立案してください。"
                "スコープの整理・WBSの骨子・マイルストーン・工数見積もりの目安・前提・リスクを簡潔に書き、"
                "最後に「オーケストレーターが次にやるべきこと」を1〜3項目で示してください。"
            ),
        ]
    )
    plan = msg.content
    out = {"planner_plan": plan, **_log(state, "Planner", plan)}
    return out


def _discussion_context_r1(state: OrchestrationState) -> str:
    return (
        f"マーケターの分析:\n{state.get('marketer_output') or ''}\n\n"
        f"プランナーの計画:\n{state.get('planner_plan') or ''}\n\n"
    )


def node_orchestrator_discussion(state: OrchestrationState) -> dict:
    """第1周: オーケストレーター - アプリでいくか・商品アイデア候補を述べる"""
    msg = _llms["orchestrator"].invoke(
        [
            SystemMessage(content=DISCUSSION_ORCHESTRATOR_R1),
            HumanMessage(content=_discussion_context_r1(state) + "上記を踏まえ、第1周として方針と商品アイデア候補を述べてください。"),
        ]
    )
    react = msg.content
    out = {"orchestrator_discussion_react": react, **_log(state, "Orchestrator(議論1周)", react)}
    return out


def node_marketer_discussion(state: OrchestrationState) -> dict:
    """第1周: マーケター - 6R・VALSを活用して賛同・補足・懸念"""
    orch_react = state.get("orchestrator_discussion_react") or ""
    msg = _llms["marketer"].invoke(
        [
            SystemMessage(content=DISCUSSION_MARKETER_R1),
            HumanMessage(content=f"オーケストレーターの結論・方針:\n{orch_react}\n\nオーケストレーターのコメントを踏まえ、6R・VALSを活用して議論に参加してください。"),
        ]
    )
    react = msg.content
    out = {"marketer_discussion_react": react, **_log(state, "Marketer(議論1周)", react)}
    return out


def node_planner_discussion(state: OrchestrationState) -> dict:
    """第1周: プランナー - 候補リストと第2周の論点を整理"""
    orch_react = state.get("orchestrator_discussion_react") or ""
    marketer_react = state.get("marketer_discussion_react") or ""
    msg = _llms["planner"].invoke(
        [
            SystemMessage(content=DISCUSSION_PLANNER_R1),
            HumanMessage(content=f"オーケストレーター:\n{orch_react}\n\nマーケター:\n{marketer_react}\n\n上記を踏まえ、第1周の整理と第2周で議論すべき論点を述べてください。"),
        ]
    )
    react = msg.content
    out = {"planner_discussion_react": react, **_log(state, "Planner(議論1周)", react)}
    return out


def node_orchestrator_discussion_2(state: OrchestrationState) -> dict:
    """第2周: オーケストレーター - 候補の絞り込み方針"""
    r1 = f"第1周・オーケストレーター:\n{state.get('orchestrator_discussion_react') or ''}\n\n第1周・マーケター:\n{state.get('marketer_discussion_react') or ''}\n\n第1周・プランナー:\n{state.get('planner_discussion_react') or ''}\n\n"
    msg = _llms["orchestrator"].invoke(
        [SystemMessage(content=DISCUSSION_ORCHESTRATOR_R2), HumanMessage(content=r1 + "上記第1周を踏まえ、第2周として絞り込み方針を述べてください。")]
    )
    react = msg.content
    out = {"orchestrator_discussion_2": react, **_log(state, "Orchestrator(議論2周)", react)}
    return out


def node_marketer_discussion_2(state: OrchestrationState) -> dict:
    """第2周: マーケター - 6R・VALSで絞り込み案"""
    r1 = f"第1周・プランナー:\n{state.get('planner_discussion_react') or ''}\n\n"
    r2_orch = state.get("orchestrator_discussion_2") or ""
    msg = _llms["marketer"].invoke(
        [SystemMessage(content=DISCUSSION_MARKETER_R2), HumanMessage(content=r1 + f"第2周・オーケストレーター:\n{r2_orch}\n\n上記を踏まえ、6R・VALSで絞り込み案を述べてください。")]
    )
    react = msg.content
    out = {"marketer_discussion_2": react, **_log(state, "Marketer(議論2周)", react)}
    return out


def node_planner_discussion_2(state: OrchestrationState) -> dict:
    """第2周: プランナー - 中間整理（7つ以内に絞り込み）"""
    r2_orch = state.get("orchestrator_discussion_2") or ""
    r2_marketer = state.get("marketer_discussion_2") or ""
    msg = _llms["planner"].invoke(
        [SystemMessage(content=DISCUSSION_PLANNER_R2), HumanMessage(content=f"第2周・オーケストレーター:\n{r2_orch}\n\n第2周・マーケター:\n{r2_marketer}\n\n上記を踏まえ、第2周の中間整理を述べてください。")]
    )
    react = msg.content
    out = {"planner_discussion_2": react, **_log(state, "Planner(議論2周)", react)}
    return out


def node_orchestrator_discussion_3(state: OrchestrationState) -> dict:
    """第3周: オーケストレーター - 商品アイデアを5つ（または10個）までに絞り込む判断"""
    r2 = f"第2周・プランナー:\n{state.get('planner_discussion_2') or ''}\n\n"
    use_10 = os.environ.get("USE_10_IDEAS") == "1"
    prompt_orch = DISCUSSION_ORCHESTRATOR_R3_10IDEAS if use_10 else DISCUSSION_ORCHESTRATOR_R3
    suffix = "商品アイデアを10個、最有力から順に並べる方針を述べてください。" if use_10 else "商品アイデアを5つまでに絞り込む方針を述べてください。"
    msg = _llms["orchestrator"].invoke(
        [SystemMessage(content=prompt_orch), HumanMessage(content=r2 + "上記第2周を踏まえ、第3周として" + suffix)]
    )
    react = msg.content
    out = {"orchestrator_discussion_3": react, **_log(state, "Orchestrator(議論3周)", react)}
    return out


def node_marketer_discussion_3(state: OrchestrationState) -> dict:
    """第3周: マーケター - 5案への賛同・修正"""
    r3_orch = state.get("orchestrator_discussion_3") or ""
    msg = _llms["marketer"].invoke(
        [SystemMessage(content=DISCUSSION_MARKETER_R3), HumanMessage(content=f"第3周・オーケストレーター:\n{r3_orch}\n\n上記を踏まえ、6R・VALSで最終5案への賛同・修正を述べてください。")]
    )
    react = msg.content
    out = {"marketer_discussion_3": react, **_log(state, "Marketer(議論3周)", react)}
    return out


def node_planner_discussion_3(state: OrchestrationState) -> dict:
    """第3周: プランナー - 商品アイデアを5つ（または10個）までに絞り込み、MVP推奨を出す"""
    r3_orch = state.get("orchestrator_discussion_3") or ""
    r3_marketer = state.get("marketer_discussion_3") or ""
    use_10 = os.environ.get("USE_10_IDEAS") == "1"
    prompt_planner = DISCUSSION_PLANNER_R3_10IDEAS if use_10 else DISCUSSION_PLANNER_R3
    suffix = "商品アイデアを10個、最有力から順にリスト化し、MVP推奨（1番）を出してください。" if use_10 else "商品アイデアを5つまでに絞り込み、番号付きリストとMVP推奨を出してください。"
    msg = _llms["planner"].invoke(
        [
            SystemMessage(content=prompt_planner),
            HumanMessage(content=f"第3周・オーケストレーター:\n{r3_orch}\n\n第3周・マーケター:\n{r3_marketer}\n\n上記を踏まえ、{suffix}"),
        ]
    )
    react = msg.content
    out = {"planner_discussion_3": react, **_log(state, "Planner(議論3周)", react)}
    return out


def node_orchestrator_task(state: OrchestrationState) -> dict:
    """オーケストレーター（タスク定義）: 議論3周の結論を踏まえ、エグゼキューター（技術）に渡すタスクを1つ定義する"""
    planner_3 = state.get("planner_discussion_3") or ""
    msg = _llms["orchestrator"].invoke(
        [
            SystemMessage(content=ROLE_ORCHESTRATOR_PM),
            HumanMessage(
                content=f"議論3周の結果（商品アイデア5つまでに絞り込み・MVP推奨）:\n{planner_3}\n\n"
                "上記の「商品アイデア（5つまで）」と「MVPとして最初に実装する推奨」を踏まえ、"
                "**推奨された1つ**について、いま実装すべき「エグゼキューター（技術）への指示」を1つ、明確に書いてください。"
            ),
        ]
    )
    plan = msg.content
    task = plan
    if "エンジニアへの指示" in plan or "エグゼキューター" in plan or "##" in plan:
        parts = re.split(r"##?\s*(?:エンジニア|エグゼキューター).*への指示|タスク[:：]|実装タスク", plan, maxsplit=1, flags=re.I)
        if len(parts) > 1:
            task = parts[-1].strip()
    out = {"pm_plan": plan, "pm_task": task, **_log(state, "Orchestrator(タスク)", plan)}
    return out


def node_engineer(state: OrchestrationState) -> dict:
    """エグゼキューター（技術）: オーケストレーターのタスクに基づきコードを実装する"""
    task = state.get("pm_task") or state.get("pm_plan") or ""
    rework_hint = state.get("review_result", "")
    if state.get("rework_count"):
        prompt = (
            f"【差し戻し指摘】\n{rework_hint}\n\n"
            f"【元のタスク】\n{task}\n\n上記指摘を反映してコードを修正し、完成したコードのみを出力してください。"
        )
    else:
        prompt = f"以下のタスクに従い、動作するコードをそのまま出力してください。説明は最小限でよいです。\n\n【タスク】\n{task}"
    msg = _llms["engineer"].invoke(
        [SystemMessage(content=ROLE_ENGINEER), HumanMessage(content=prompt)]
    )
    code = msg.content
    out = {"engineer_code": code, **_log(state, "Engineer", code)}
    return out


def node_reviewer(state: OrchestrationState) -> dict:
    """クリティック／レビュアー（検証・再現）: コードをレビューし、承認 or 差し戻しを判定する。差し戻し時は rework_count を1にする。"""
    code = state.get("engineer_code") or ""
    msg = _llms["reviewer"].invoke(
        [
            SystemMessage(content=ROLE_REVIEWER),
            HumanMessage(
                content=f"以下のコードをレビューし、最後に「判定: 承認」または「判定: 差し戻し」を1行で書いてください。\n\n```\n{code}\n```"
            ),
        ]
    )
    result = msg.content
    approved = "判定: 承認" in result or "承認" in result[-50:]
    out = {
        "review_result": result,
        "review_approved": approved,
        **_log(state, "Reviewer", result),
    }
    if not approved and (state.get("rework_count") or 0) < 1:
        out["rework_count"] = 1
    return out


def node_pm_final(state: OrchestrationState) -> dict:
    """オーケストレーター: 完了報告をまとめる"""
    discussion_3 = (state.get("planner_discussion_3") or "")[:400]
    plan = state.get("pm_plan", "")
    report = (
        f"【完了報告】\n"
        f"議論3周の結論（商品アイデア5つ・MVP推奨）:\n{discussion_3}...\n\n"
        f"タスク計画: {plan[:250]}...\n"
        f"実装サマリ: コード長 {len(state.get('engineer_code') or '')} 文字。レビュー承認済み。"
    )
    return {**_log(state, "Orchestrator(報告)", report)}


def route_after_review(state: OrchestrationState) -> Literal["engineer", "pm_final"]:
    """レビュー後: 差し戻しかつ1回目なら engineer、それ以外は pm_final"""
    if state.get("review_approved"):
        return "pm_final"
    rework = state.get("rework_count") or 0
    if rework < 1:
        return "engineer"
    return "pm_final"


def route_after_discussion(state: OrchestrationState) -> Literal["orchestrator_task", "end"]:
    """議論3周後: discussion_only なら終了、そうでなければ オーケストレーター（タスク）へ"""
    if state.get("discussion_only"):
        return "end"
    return "orchestrator_task"


def route_start(state: OrchestrationState) -> Literal["orchestrator", "orchestrator_task"]:
    """開始: implement_only なら オーケストレーター（タスク）から、そうでなければ オーケストレーター（戦略）から"""
    if state.get("implement_only"):
        return "orchestrator_task"
    return "orchestrator"


def route_after_engineer(state: OrchestrationState) -> Literal["reviewer", "pm_final"]:
    """エンジニア実装後: skip_review なら確認せず完了へ、そうでなければレビュアーへ"""
    if state.get("skip_review"):
        return "pm_final"
    return "reviewer"


# ========== グラフ構築 ==========


def build_graph() -> CompiledStateGraph:
    builder = StateGraph(OrchestrationState)

    builder.add_node("orchestrator", node_orchestrator)
    builder.add_node("marketer", node_marketer)
    builder.add_node("planner", node_planner)
    builder.add_node("orchestrator_discussion", node_orchestrator_discussion)
    builder.add_node("marketer_discussion", node_marketer_discussion)
    builder.add_node("planner_discussion", node_planner_discussion)
    builder.add_node("orchestrator_discussion_2", node_orchestrator_discussion_2)
    builder.add_node("marketer_discussion_2", node_marketer_discussion_2)
    builder.add_node("planner_discussion_2", node_planner_discussion_2)
    builder.add_node("orchestrator_discussion_3", node_orchestrator_discussion_3)
    builder.add_node("marketer_discussion_3", node_marketer_discussion_3)
    builder.add_node("planner_discussion_3", node_planner_discussion_3)
    builder.add_node("orchestrator_task", node_orchestrator_task)
    builder.add_node("engineer", node_engineer)
    builder.add_node("reviewer", node_reviewer)
    builder.add_node("pm_final", node_pm_final)

    builder.add_conditional_edges(START, route_start, {"orchestrator": "orchestrator", "orchestrator_task": "orchestrator_task"})
    builder.add_edge("orchestrator", "marketer")
    builder.add_edge("marketer", "planner")
    builder.add_edge("planner", "orchestrator_discussion")
    builder.add_edge("orchestrator_discussion", "marketer_discussion")
    builder.add_edge("marketer_discussion", "planner_discussion")
    builder.add_edge("planner_discussion", "orchestrator_discussion_2")
    builder.add_edge("orchestrator_discussion_2", "marketer_discussion_2")
    builder.add_edge("marketer_discussion_2", "planner_discussion_2")
    builder.add_edge("planner_discussion_2", "orchestrator_discussion_3")
    builder.add_edge("orchestrator_discussion_3", "marketer_discussion_3")
    builder.add_edge("marketer_discussion_3", "planner_discussion_3")
    builder.add_conditional_edges("planner_discussion_3", route_after_discussion, {"orchestrator_task": "orchestrator_task", "end": END})
    builder.add_edge("orchestrator_task", "engineer")
    builder.add_conditional_edges("engineer", route_after_engineer, {"reviewer": "reviewer", "pm_final": "pm_final"})
    builder.add_conditional_edges("reviewer", route_after_review, {"engineer": "engineer", "pm_final": "pm_final"})
    builder.add_edge("pm_final", END)

    return builder.compile()


# ========== 実行 ==========

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="マルチエージェント・オーケストレーション（オーケストレーター・マーケター・プランナー→議論3周→エグゼキューター・クリティック）")
    p.add_argument("idea", nargs="*", help="企画内容（未指定時はデフォルト企画）")
    p.add_argument("-d", "--discussion-only", action="store_true",
                    help="議論を回してと依頼した時用。オーケストレーター・マーケター・プランナーで議論を3回回し、実装は実行しない")
    p.add_argument("-i", "--implement", action="store_true",
                    help="実装専用モード。オーケストレーター（タスク）・エグゼキューター・クリティックのみ実行し、完成まで進める")
    p.add_argument("-n", "--no-verify", action="store_true",
                    help="アプリの完成まで確認（レビュー）しない。エンジニア実装後にそのまま完了報告へ進む（-i と併用推奨）")
    p.add_argument("-o", "--output-dir", default="", help="実装時、生成コードをこのディレクトリに保存する")
    args = p.parse_args()
    idea_str = " ".join(args.idea).strip() if args.idea else "ターミナルで入力されたテキストの語数を数えるPython CLIツールを作りたい。"
    return idea_str, args.discussion_only, args.implement, getattr(args, "no_verify", False), getattr(args, "output_dir", "")


def _synthetic_planner_for_implement(task: str) -> str:
    """実装専用モード用: PM に渡す合成 planner_discussion_3"""
    return (
        "## 商品アイデア（1つ・実装タスク）\n"
        "1. 実装タスク: " + task + "\n\n"
        "MVPとして最初に実装する推奨: 上記の1番。"
    )


if __name__ == "__main__":
    idea, discussion_only, implement_only, skip_review, output_dir = _parse_args()
    print("【企画】", idea, "\n")
    if discussion_only:
        print(f"【モード】議論のみ（オーケストレーター・マーケター・プランナーで議論を{DISCUSSION_ROUNDS}回回します）\n")
    if implement_only:
        print("【モード】実装専用（オーケストレーター・エグゼキューター・クリティックで相談しながら完成まで進めます）\n")
    if skip_review:
        print("【確認スキップ】レビュアーを実行せず、エンジニア実装後にそのまま完了します。\n")

    graph = build_graph()
    initial: OrchestrationState = {
        "project_idea": idea,
        "discussion_only": discussion_only,
        "implement_only": implement_only,
        "skip_review": skip_review,
        "log": [],
        "rework_count": 0,
    }
    if implement_only:
        initial["planner_discussion_3"] = _synthetic_planner_for_implement(idea)
    result = graph.invoke(initial, config={"recursion_limit": 35})

    print("\n========== ログ ==========")
    for line in result.get("log") or []:
        print(line)

    if discussion_only:
        print("\n========== 議論3周の結論（商品アイデア5つ・MVP推奨） ==========")
        print(result.get("planner_discussion_3", "(なし)"))
    else:
        print("\n========== 最終成果物（コード） ==========")
        code = result.get("engineer_code") or ""
        print(code[:2000])
        if len(code) > 2000:
            print("... (省略)")
        if output_dir and code:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "generated_code.txt").write_text(code, encoding="utf-8")
            print(f"\n[保存] {out / 'generated_code.txt'}")
