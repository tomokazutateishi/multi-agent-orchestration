# マルチエージェント・オーケストレーション

**社長は削除**し、**4役**（オーケストレーター/プランナー・エグゼキューター＋ドメインエキスパート（技術）・ドメインエキスパート（UI/UX）・クリティック/レビュアー＋エグゼキューター（検証・再現））が、**役割ごとに最適な生成AI**（OpenAI / Gemini / Claude）で会話しながらプロジェクトを進めるデモです。LangGraph 上ではオーケストレーター/プランナーを orchestrator・marketer・planner の3ノードで実装しています。

## 流れ

- **ターゲット**: アプリ（に近いもの）に対し**課金することに躊躇がないユーザー**を前提とする。
- **マーケター**: **超優秀なマーケター**として **6R・VALS** 等のフレームワークを活用して議論に参加する。

1. **オーケストレーター** … 企画意図を戦略方針にまとめる（課金ユーザーをターゲットに）。
2. **マーケター** … 6R・VALSを活用して市場・ユーザー分析とプロダクト要件の骨子を提供。
3. **プランナー** … オーケストレーター方針＋マーケター要件を踏まえ、プロジェクト計画を立案。
4. **議論3周** … オーケストレーター・マーケター・プランナーで「アプリでいいか／何をするか」を**常に3回**議論し、**商品アイデアを5つまでに絞り込み**、MVP推奨を出す。（**議論を回してと依頼した時は必ず3回回す**。）
5. **オーケストレーター（タスク定義）** … 5案とMVP推奨を踏まえ、エグゼキューター（技術）に渡すタスクを1つ定義。
6. **エグゼキューター（技術）** … タスクに基づきコードを実装。
7. **クリティック／レビュアー** … コードをレビューし、承認 or 差し戻し。
8. 差し戻しなら **エグゼキューター（技術）** に1回だけ再実装 → 再度レビュー。
9. **オーケストレーター** が完了報告 → 終了。

## 役割ごとの最適モデル（デフォルト）

| 役割 | 推奨 | 理由 |
|------|------|------|
| オーケストレーター | **Gemini** | 戦略・タスク定義の構造化出力、ビジネス向け |
| マーケター | **OpenAI (gpt-4o-mini)** | 市場・ユーザー分析のデータドリブン・構造化が得意 |
| プランナー | **OpenAI (gpt-4o-mini)** | 計画立案・WBS・見積もりの論理・構造化が得意 |
| エグゼキューター（技術） | **OpenAI**（Claude 利用可） | コード生成（Claude は SWE-bench で最高クラス） |
| クリティック／レビュアー | **Gemini** | 客観的レビュー・建設的フィードバック |

Claude（Anthropic）の API キーを設定していない場合は、エグゼキューター・レビュアーは自動で OpenAI / Gemini にフォールバックします。

## 必要なもの

- Python 3.10+
- `Documents/.env` に以下を設定:
  - **必須**: `OPENAI_API_KEY`（OpenAI）、`GOOGLE_API_KEY`（Gemini）
  - **任意**: `ANTHROPIC_API_KEY`（Claude を使う場合。`pip install langchain-anthropic` も必要）

## 実行方法

```bash
cd /Users/tomokazu.tateishi/Documents
source venv/bin/activate
python multi-agent-orchestration/run.py あなたの企画内容
```

- **議論を回してと依頼した時**: `-d` または `--discussion-only` を付けると、オーケストレーター・マーケター・プランナーで**議論を3回だけ**回し、商品アイデア5つとMVP推奨まで出して終了（実装は実行しない）。
  ```bash
  python multi-agent-orchestration/run.py -d 直近半年でバズったトレンドを踏まえた無料Webサービス企画
  ```
- **実装専用モード**: `-i` または `--implement` を付けると、オーケストレーター（タスク）・エグゼキューター・クリティックのみ実行（戦略・マーケター・プランナー・議論はスキップ）。
  ```bash
  python multi-agent-orchestration/run.py -i 企画内容
  ```
- **10案・最有力順**: 環境変数 `USE_10_IDEAS=1` を付けると、議論の結論で商品アイデアを10個・最有力から順に出力。
- **Gemini が使えないとき**: `USE_OPENAI_ONLY=1 python multi-agent-orchestration/run.py ...` で全役を OpenAI に。

## 役割ごとのモデルを上書きする

環境変数で「プロバイダー:モデル」を指定できます。

```bash
export ROLE_ORCHESTRATOR_MODEL=google:gemini-2.0-flash
export ROLE_ENGINEER_MODEL=anthropic:claude-3-5-sonnet-20241022
export ROLE_REVIEWER_MODEL=anthropic:claude-3-5-sonnet-20241022
python multi-agent-orchestration/run.py 企画内容
```

`.env` に同じキーで書いても反映されます。

## Claude（Anthropic）を使う場合

1. [Anthropic Console](https://console.anthropic.com/) で API キーを取得  
2. `pip install langchain-anthropic`  
3. `Documents/.env` に `ANTHROPIC_API_KEY=sk-ant-...` を追加  

するとエグゼキューター・レビュアーが自動で Claude に切り替わります。

## フォルダ構成

```
Documents/
├── .env                 # APIキー
├── venv/
└── multi-agent-orchestration/
    ├── config.py        # 役割別モデル設定・フォールバック（4役構成・社長削除済み）
    ├── prompts.py       # 各役のプロンプト（役割名・スキル補完済み）
    ├── run.py           # オーケストレーション本体
    └── README.md
```

## カスタマイズ

- **プロンプト**: `prompts.py` の `ROLE_ORCHESTRATOR` / `ROLE_ORCHESTRATOR_PM` / `ROLE_MARKETER` / `ROLE_PLANNER` / `ROLE_ENGINEER` / `ROLE_REVIEWER` / `ROLE_DESIGNER`（ドメインエキスパート UI/UX、ワークフロー・手動用）
- **最適割り当て・フォールバック**: `config.py` の `OPTIMAL_ROLE_MODELS` / `FALLBACK_ROLE_MODELS` / `DEFAULT_MODELS`

## 役割・スキル一覧（サービス作成ワークフロー側）

`coading/2025-01-30/ROLES_AND_SKILLS.md` に、オーケストレーター／プランナー・エグゼキューター（技術）・ドメインエキスパート（UI/UX）・クリティック／レビュアーの各スキルを補完した一覧があります。

## Author / リポジトリ

- GitHub: [tomokazutateishi](https://github.com/tomokazutateishi/)
