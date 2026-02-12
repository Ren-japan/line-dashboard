import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="LINE広告 ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .kpi-card.green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .kpi-card.orange { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .kpi-card.red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .kpi-card.blue { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .kpi-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .kpi-card h1 { margin: 0.3rem 0 0 0; font-size: 1.8rem; font-weight: 700; }
    .kpi-card p { margin: 0.2rem 0 0 0; font-size: 0.75rem; opacity: 0.8; }
    .ai-box {
        background: #1a1a2e;
        border: 1px solid #16213e;
        border-radius: 12px;
        padding: 1.2rem;
        color: #e0e0e0;
        margin: 0.5rem 0;
    }
    .ai-box h4 { color: #00d2ff; margin-top: 0; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.3rem;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Loading
# ============================================================
@st.cache_data(ttl=300)
def load_data_from_sheets():
    """Google Sheets APIからデータ取得"""
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        creds_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "credentials.json")
        spreadsheet_id = os.getenv(
            "SOURCE_SPREADSHEET_ID",
            "1XnSOo0lzOmGBrn-oqxoGR2QRlKppaGYXbVLHmwOcHNk",
        )

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets.readonly",
            "https://www.googleapis.com/auth/drive.readonly",
        ]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        gc = gspread.authorize(creds)

        sh = gc.open_by_key(spreadsheet_id)
        worksheet = sh.worksheet("新データ収集")
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        return df, None
    except FileNotFoundError:
        return None, "credentials.json が見つかりません。サービスアカウントの設定が必要です。"
    except Exception as e:
        return None, f"Google Sheets接続エラー: {str(e)}"


def load_demo_data():
    """デモ用サンプルデータ（API未設定時）"""
    import numpy as np

    np.random.seed(42)
    dates = pd.date_range("2026-01-01", "2026-02-09", freq="D")

    records = []
    accounts = ["包茎手術"]
    genres = ["seo_ed", "ad_aga", "lis_aga"]
    tag_names = [
        "/seo_ed/DMME_包茎",
        "/seo_ed/DMME_包茎2",
        "/seo_ed/DMME_包茎3",
        "/ad_aga/包茎LP1",
        "/ad_aga/包茎LP2",
        "/lis_aga/包茎リス1",
    ]
    tag_ids = [1445, 1444, 1443, 667, 668, 1248]

    for date in dates:
        for i, tag in enumerate(tag_names):
            lp_imp = np.random.randint(5, 500)
            pu_imp = int(lp_imp * np.random.uniform(0.3, 0.9))
            pu_click = int(pu_imp * np.random.uniform(0, 0.5))
            pu_rate = f"{pu_click / pu_imp * 100:.1f}%" if pu_imp > 0 else "0%"
            added_friends = np.random.randint(0, 10)
            records.append(
                {
                    "data_date": date.strftime("%Y/%m/%d"),
                    "account": accounts[0],
                    "genre": genres[i % len(genres)],
                    "tag_name": tag,
                    "lp_imp": lp_imp,
                    "pu_imp": pu_imp,
                    "pu_click": pu_click,
                    "pu_rate": pu_rate,
                    "added_friends": added_friends,
                    "tag_id": tag_ids[i],
                }
            )

    return pd.DataFrame(records)


def prepare_data(df):
    """データの前処理"""
    df = df.copy()

    # 日付変換
    if "data_date" in df.columns:
        df["data_date"] = pd.to_datetime(df["data_date"], errors="coerce")

    # 数値変換
    numeric_cols = ["lp_imp", "pu_imp", "pu_click", "added_friends", "tag_id"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # pu_rate をfloatに
    if "pu_rate" in df.columns:
        df["pu_rate_pct"] = (
            df["pu_rate"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

    # PU CTR計算（pu_click / pu_imp）
    df["pu_ctr"] = df.apply(
        lambda r: (r["pu_click"] / r["pu_imp"] * 100) if r["pu_imp"] > 0 else 0,
        axis=1,
    )

    # PU表示率計算（pu_imp / lp_imp）
    df["pu_display_rate"] = df.apply(
        lambda r: (r["pu_imp"] / r["lp_imp"] * 100) if r["lp_imp"] > 0 else 0, axis=1
    )

    # チャネル分類（genre列ベース）: SEO > アド > リス（それ以外）
    def classify_channel(genre):
        g = str(genre).lower()
        if "seo" in g:
            return "SEO"
        elif "ad" in g or "yda" in g:
            return "アド"
        else:
            return "リス"

    if "genre" in df.columns:
        df["channel"] = df["genre"].apply(classify_channel)

    # tag_name解析: チャネル > ジャンル > サイト > 記事 > PU
    # ルール:
    #   パス1段目 = genre(チャネル+ジャンル) → スキップ
    #   パス2段以上: 最後から2番目 = サイト、最後のセグメントを_分割
    #   パス1段: _区切りの最初 = サイト
    #   _区切りの最後 = PU訴求テキスト、中間 = 記事
    def parse_tag(tag_name):
        """tag_nameからサイト・記事・PUを抽出"""
        tag = str(tag_name).strip()
        if not tag or tag == "nan":
            return "", "", tag

        clean = tag.replace(" ", "").lstrip("/")

        # /なし (test_01, テスト, POP_7_xxx)
        if "/" not in clean:
            segs = clean.split("_")
            if len(segs) == 1:
                return "", "", clean
            if len(segs) == 2:
                return segs[0], "", segs[1]
            return segs[0], "_".join(segs[1:-1]), segs[-1]

        parts = clean.split("/")
        rest_parts = parts[1:]  # genre部分をスキップ

        if len(rest_parts) == 0:
            return "", "", parts[0]

        # パス2段以上 (/genre/site/last_part)
        if len(rest_parts) >= 2:
            site = rest_parts[-2]
            last = rest_parts[-1]
            segs = [s for s in last.split("_") if s]
            if len(segs) == 0:
                return site, "", ""
            elif len(segs) == 1:
                return site, "", segs[0]
            else:
                return site, "_".join(segs[:-1]), segs[-1]

        # パス1段 (/genre/xxx_yyy_zzz)
        last = rest_parts[0]
        segs = [s for s in last.split("_") if s]
        if len(segs) == 0:
            return "", "", last
        elif len(segs) == 1:
            return "", "", segs[0]
        elif len(segs) == 2:
            return segs[0], "", segs[1]
        else:
            return segs[0], "_".join(segs[1:-1]), segs[-1]

    if "tag_name" in df.columns:
        parsed = df["tag_name"].apply(lambda t: pd.Series(parse_tag(t)))
        df["site"] = parsed[0]
        df["article"] = parsed[1]
        df["pu_label"] = parsed[2]

    return df


# ============================================================
# AI Assist
# ============================================================
def get_ai_insights(df_filtered, period_label="選択期間"):
    """Gemini APIでAIアシストコメントを生成"""
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key or gemini_key == "your_gemini_api_key_here":
        return get_rule_based_insights(df_filtered, period_label)

    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # データサマリーを作成
        summary = create_data_summary_for_ai(df_filtered)

        prompt = f"""あなたはLINE広告運用のプロマーケターです。
以下のデータを分析して、具体的なアクションにつながる洞察を3〜5個、箇条書きで簡潔に日本語で提示してください。

分析観点:
1. 成果が良い記事/PU → 「派生展開」や「予算増」の提案
2. 成果が悪い記事/PU → 「停止検討」や「改善案」
3. トレンド変化 → 「先週比で◯◯が上昇/下降」
4. 友だち追加効率 → コスパの良い/悪い記事

データ:
{summary}

重要: 具体的な数値を使って根拠を示し、「〜すべき」「〜を検討」など明確なアクション提案をしてください。"""

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        return get_rule_based_insights(df_filtered, period_label)


def create_data_summary_for_ai(df):
    """AI用のデータサマリーを作成"""
    lines = []

    # タグ別集計
    tag_summary = (
        df.groupby("tag_name")
        .agg(
            total_lp_imp=("lp_imp", "sum"),
            total_pu_imp=("pu_imp", "sum"),
            total_pu_click=("pu_click", "sum"),
            total_friends=("added_friends", "sum"),
            days=("data_date", "nunique"),
        )
        .reset_index()
    )
    tag_summary["pu_ctr"] = tag_summary.apply(
        lambda r: f"{r['total_pu_click']/r['total_pu_imp']*100:.1f}%"
        if r["total_pu_imp"] > 0
        else "0%",
        axis=1,
    )
    tag_summary["pu_display_rate"] = tag_summary.apply(
        lambda r: f"{r['total_pu_imp']/r['total_lp_imp']*100:.1f}%"
        if r["total_lp_imp"] > 0
        else "0%",
        axis=1,
    )

    lines.append("【タグ別パフォーマンス】")
    for _, row in tag_summary.iterrows():
        lines.append(
            f"  {row['tag_name']}: LP imp={row['total_lp_imp']}, "
            f"PU表示率={row['pu_display_rate']}, PU CTR={row['pu_ctr']}, "
            f"友だち追加={row['total_friends']} ({row['days']}日間)"
        )

    # 直近7日 vs 前7日の比較
    if "data_date" in df.columns and len(df) > 0:
        max_date = df["data_date"].max()
        recent = df[df["data_date"] > max_date - timedelta(days=7)]
        prev = df[
            (df["data_date"] <= max_date - timedelta(days=7))
            & (df["data_date"] > max_date - timedelta(days=14))
        ]

        if len(recent) > 0 and len(prev) > 0:
            lines.append("\n【直近7日 vs 前7日】")
            lines.append(
                f"  LP imp: {recent['lp_imp'].sum()} vs {prev['lp_imp'].sum()}"
            )
            lines.append(
                f"  PU click: {recent['pu_click'].sum()} vs {prev['pu_click'].sum()}"
            )
            lines.append(
                f"  友だち追加: {recent['added_friends'].sum()} vs {prev['added_friends'].sum()}"
            )

    return "\n".join(lines)


def get_rule_based_insights(df, period_label="選択期間"):
    """ルールベースの洞察（Gemini未接続時）"""
    insights = []

    if len(df) == 0:
        return "データがありません。フィルタ条件を確認してください。"

    # タグ別集計
    tag_stats = (
        df.groupby("tag_name")
        .agg(
            lp_imp=("lp_imp", "sum"),
            pu_imp=("pu_imp", "sum"),
            pu_click=("pu_click", "sum"),
            friends=("added_friends", "sum"),
        )
        .reset_index()
    )
    tag_stats["ctr"] = tag_stats.apply(
        lambda r: r["pu_click"] / r["pu_imp"] * 100 if r["pu_imp"] > 0 else 0, axis=1
    )
    tag_stats["friend_rate"] = tag_stats.apply(
        lambda r: r["friends"] / r["pu_click"] * 100 if r["pu_click"] > 0 else 0,
        axis=1,
    )

    # トップパフォーマー
    if len(tag_stats) > 0:
        best = tag_stats.sort_values("friends", ascending=False).iloc[0]
        if best["friends"] > 0:
            insights.append(
                f"🏆 **ベスト記事**: `{best['tag_name']}` — "
                f"友だち追加 **{int(best['friends'])}件**、CTR {best['ctr']:.1f}%。"
                f"**派生コンテンツの作成を推奨します。**"
            )

    # ワーストパフォーマー
    low_performers = tag_stats[
        (tag_stats["lp_imp"] > 10) & (tag_stats["friends"] == 0)
    ]
    if len(low_performers) > 0:
        worst = low_performers.sort_values("lp_imp", ascending=False).iloc[0]
        insights.append(
            f"⚠️ **要注意**: `{worst['tag_name']}` — "
            f"LP imp **{int(worst['lp_imp'])}** あるのに友だち追加 **0件**。"
            f"**PUの見直しまたは停止を検討してください。**"
        )

    # CTR高いがimp低い（隠れた優良記事）
    hidden_gems = tag_stats[(tag_stats["ctr"] > 30) & (tag_stats["lp_imp"] < 100)]
    if len(hidden_gems) > 0:
        gem = hidden_gems.sort_values("ctr", ascending=False).iloc[0]
        insights.append(
            f"💎 **隠れた優良記事**: `{gem['tag_name']}` — "
            f"CTR **{gem['ctr']:.1f}%** と高いがimp少ない（{int(gem['lp_imp'])}）。"
            f"**露出を増やせば友だち追加が伸びる可能性あり。**"
        )

    # 直近トレンド
    if "data_date" in df.columns:
        max_date = df["data_date"].max()
        recent_3d = df[df["data_date"] > max_date - timedelta(days=3)]
        prev_3d = df[
            (df["data_date"] <= max_date - timedelta(days=3))
            & (df["data_date"] > max_date - timedelta(days=6))
        ]
        if len(recent_3d) > 0 and len(prev_3d) > 0:
            r_friends = recent_3d["added_friends"].sum()
            p_friends = prev_3d["added_friends"].sum()
            if p_friends > 0:
                change = (r_friends - p_friends) / p_friends * 100
                emoji = "📈" if change > 0 else "📉"
                insights.append(
                    f"{emoji} **直近3日トレンド**: 友だち追加 "
                    f"{'+'if change>0 else ''}{change:.0f}% "
                    f"（{int(p_friends)} → {int(r_friends)}）"
                )

    if not insights:
        insights.append(
            "📊 データを蓄積中です。十分なデータが集まると自動分析が始まります。"
        )

    return "\n\n".join(insights)


# ============================================================
# Main App
# ============================================================
def main():
    # --- Header ---
    st.markdown("# 📊 LINE広告 ダッシュボード")
    st.markdown("##### LINE広告 パフォーマンス分析 | チャネル別 × ジャンル別")

    # --- Load Data ---
    df_raw, error = load_data_from_sheets()

    if error or df_raw is None:
        st.warning(f"⚠️ Google Sheets API未接続: {error or ''}")
        st.info("デモデータで表示します。API接続するには `.env` に `GOOGLE_SERVICE_ACCOUNT_JSON` を設定してください。")
        df_raw = load_demo_data()

    df = prepare_data(df_raw)

    # 包茎手術関連のみフィルタ（account or tag_nameで判別）
    # 元データの構造上、accountやgenre、tag_nameで絞り込む
    # まず全データを見せて、サイドバーでフィルタ

    # --- Sidebar Filters ---
    with st.sidebar:
        st.markdown("## 🔍 フィルタ")

        # 日付範囲
        if "data_date" in df.columns and df["data_date"].notna().any():
            min_date = df["data_date"].min().date()
            max_date = df["data_date"].max().date()
            date_range = st.date_input(
                "📅 期間",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
            )
            if len(date_range) == 2:
                df = df[
                    (df["data_date"].dt.date >= date_range[0])
                    & (df["data_date"].dt.date <= date_range[1])
                ]

        # チャネル（SEO / アド / リス）
        if "channel" in df.columns:
            channels = sorted(df["channel"].unique())
            selected_channels = st.multiselect(
                "📡 チャネル", channels, default=channels,
                help="SEO=seo含む / アド=ad,yda含む / リス=それ以外",
            )
            if selected_channels:
                df = df[df["channel"].isin(selected_channels)]

        # ジャンル（= account列）
        if "account" in df.columns:
            accounts = sorted(df["account"].unique())
            selected_accounts = st.multiselect(
                "📁 ジャンル", accounts, default=accounts,
            )
            if selected_accounts:
                df = df[df["account"].isin(selected_accounts)]

        # サイトフィルタ
        if "site" in df.columns:
            sites = sorted(df["site"].unique())
            if len(sites) > 1:
                selected_sites = st.multiselect(
                    "🌐 サイト", sites, default=[],
                    help="空=全サイト表示。絞り込みたいサイトを選択",
                )
                if selected_sites:
                    df = df[df["site"].isin(selected_sites)]

        st.markdown("---")
        num_sites = df["site"].nunique() if "site" in df.columns else 0
        num_articles = df["article"].nunique() if "article" in df.columns else 0
        num_pus = df["tag_name"].nunique() if "tag_name" in df.columns else 0
        st.markdown(
            f"📊 表示中: **{len(df):,}行** / {num_sites}サイト / {num_articles}記事 / {num_pus} PU"
        )

        # データ再読み込み
        if st.button("🔄 データ再読み込み"):
            st.cache_data.clear()
            st.rerun()

    # --- KPI Cards ---
    st.markdown('<p class="section-header">📈 KPIサマリー</p>', unsafe_allow_html=True)

    total_lp_imp = int(df["lp_imp"].sum())
    total_pu_imp = int(df["pu_imp"].sum())
    total_pu_click = int(df["pu_click"].sum())
    total_friends = int(df["added_friends"].sum())
    avg_pu_display = (total_pu_imp / total_lp_imp * 100) if total_lp_imp > 0 else 0
    avg_pu_ctr = (total_pu_click / total_pu_imp * 100) if total_pu_imp > 0 else 0
    friend_per_click = (
        (total_friends / total_pu_click * 100) if total_pu_click > 0 else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(
            f"""<div class="kpi-card">
            <h3>LP imp</h3>
            <h1>{total_lp_imp:,}</h1>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="kpi-card green">
            <h3>PU imp</h3>
            <h1>{total_pu_imp:,}</h1>
            <p>表示率 {avg_pu_display:.1f}%</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="kpi-card orange">
            <h3>PU Click</h3>
            <h1>{total_pu_click:,}</h1>
            <p>CTR {avg_pu_ctr:.1f}%</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="kpi-card blue">
            <h3>友だち追加</h3>
            <h1>{total_friends:,}</h1>
            <p>Click→友だち {friend_per_click:.1f}%</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col5:
        num_pus = df["tag_name"].nunique() if "tag_name" in df.columns else 0
        num_sites = df["site"].nunique() if "site" in df.columns else 0
        st.markdown(
            f"""<div class="kpi-card red">
            <h3>稼働PU数</h3>
            <h1>{num_pus}</h1>
            <p>{num_sites}サイト</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ============================================================
    # サイト別 → 記事別 → PU別 ドリルダウン
    # ============================================================

    # --- 共通: 率系指標の計算ヘルパー ---
    def add_rate_cols(summary):
        summary["display_rate"] = summary.apply(
            lambda r: r["pu_imp"] / r["lp_imp"] * 100 if r["lp_imp"] > 0 else 0, axis=1)
        summary["ctr"] = summary.apply(
            lambda r: r["pu_click"] / r["pu_imp"] * 100 if r["pu_imp"] > 0 else 0, axis=1)
        summary["friend_rate"] = summary.apply(
            lambda r: r["friends"] / r["pu_click"] * 100 if r["pu_click"] > 0 else 0, axis=1)
        summary["imp_fr"] = summary.apply(
            lambda r: r["friends"] / r["pu_imp"] * 100 if r["pu_imp"] > 0 else 0, axis=1)
        summary["pv_fr"] = summary.apply(
            lambda r: r["friends"] / r["lp_imp"] * 100 if r["lp_imp"] > 0 else 0, axis=1)
        return summary

    def fmt_pct_cols(summary):
        pct_fmt = lambda v: f"{v:.2f}%" if v != 0 else "0.00%"
        for col in ["display_rate", "ctr", "friend_rate", "imp_fr", "pv_fr"]:
            if col in summary.columns:
                summary[col] = summary[col].apply(pct_fmt)
        return summary

    rate_col_rename = {
        "lp_imp": "LP imp", "pu_imp": "PU imp", "display_rate": "PU表示率",
        "pu_click": "PUクリック", "ctr": "PU CTR", "friends": "友だち追加数",
        "friend_rate": "友だち追加率", "imp_fr": "impFR", "pv_fr": "pvFR",
    }
    rate_col_order = [
        "LP imp", "PU imp", "PU表示率", "PUクリック", "PU CTR",
        "友だち追加数", "友だち追加率", "impFR", "pvFR",
    ]

    if "site" in df.columns:
        # ======== Level 1: サイト別 ========
        st.markdown(
            '<p class="section-header">🌐 サイト別パフォーマンス</p>',
            unsafe_allow_html=True,
        )
        site_summary = (
            df.groupby("site")
            .agg(channel=("channel", "first"), account=("account", "first"),
                 article_count=("article", lambda x: x[x != ""].nunique()),
                 pu_count=("tag_name", "nunique"),
                 lp_imp=("lp_imp", "sum"), pu_imp=("pu_imp", "sum"),
                 pu_click=("pu_click", "sum"), friends=("added_friends", "sum"),
                 days=("data_date", "nunique"))
            .reset_index()
        )
        site_summary = add_rate_cols(site_summary)
        site_summary = site_summary.sort_values("friends", ascending=False)
        site_display = fmt_pct_cols(site_summary.copy())
        site_display = site_display.rename(columns={
            "site": "サイト", "channel": "チャネル", "account": "ジャンル",
            "article_count": "記事数", "pu_count": "PU数", **rate_col_rename,
        })
        st.dataframe(
            site_display[["サイト", "チャネル", "ジャンル", "記事数", "PU数"] + rate_col_order],
            use_container_width=True, height=320,
        )

        # ======== Level 2: サイト → 記事別 ========
        st.markdown(
            '<p class="section-header">🔍 サイト → 記事別ドリルダウン</p>',
            unsafe_allow_html=True,
        )
        site_list = site_summary["site"].tolist()
        site_labels = {
            r["site"]: f"{r['site']}  ({int(r['friends'])}友だち / {int(r['pu_count'])}PU)"
            for _, r in site_summary.iterrows()
        }
        selected_site = st.selectbox(
            "サイトを選択", site_list,
            format_func=lambda x: site_labels.get(x, x), key="site_drilldown",
        )

        if selected_site:
            df_site = df[df["site"] == selected_site]
            # 記事がない（空文字）PUもあるので、空文字は "(直PU)" に置換
            df_site = df_site.copy()
            df_site["article_label"] = df_site["article"].apply(lambda x: x if x else "(直PU)")

            art_summary = (
                df_site.groupby("article_label")
                .agg(pu_count=("tag_name", "nunique"),
                     lp_imp=("lp_imp", "sum"), pu_imp=("pu_imp", "sum"),
                     pu_click=("pu_click", "sum"), friends=("added_friends", "sum"),
                     days=("data_date", "nunique"))
                .reset_index()
            )
            art_summary = add_rate_cols(art_summary)
            art_summary = art_summary.sort_values("friends", ascending=False)

            # 記事KPI
            ak1, ak2, ak3, ak4 = st.columns(4)
            with ak1:
                st.metric("友だち追加", f"{int(art_summary['friends'].sum()):,}")
            with ak2:
                st.metric("LP imp", f"{int(art_summary['lp_imp'].sum()):,}")
            with ak3:
                st.metric("記事数", f"{len(art_summary)}")
            with ak4:
                st.metric("PU数", f"{int(art_summary['pu_count'].sum())}")

            art_display = fmt_pct_cols(art_summary.copy())
            art_display = art_display.rename(columns={
                "article_label": "記事", "pu_count": "PU数", **rate_col_rename,
            })
            st.dataframe(
                art_display[["記事", "PU数"] + rate_col_order],
                use_container_width=True, height=min(350, len(art_summary) * 40 + 60),
            )

            # ======== Level 3: 記事 → PU別 ========
            st.markdown(
                '<p class="section-header">🔍 記事 → PU別ドリルダウン</p>',
                unsafe_allow_html=True,
            )
            art_list = art_summary["article_label"].tolist()
            art_labels = {
                r["article_label"]: f"{r['article_label']}  ({int(r['friends'])}友だち / {int(r['pu_count'])}PU)"
                for _, r in art_summary.iterrows()
            }
            selected_art = st.selectbox(
                "記事を選択", art_list,
                format_func=lambda x: art_labels.get(x, x), key="art_drilldown",
            )

            if selected_art:
                art_val = "" if selected_art == "(直PU)" else selected_art
                df_art = df_site[df_site["article"] == art_val]

                pu_summary = (
                    df_art.groupby("tag_name")
                    .agg(pu_label=("pu_label", "first"),
                         lp_imp=("lp_imp", "sum"), pu_imp=("pu_imp", "sum"),
                         pu_click=("pu_click", "sum"), friends=("added_friends", "sum"),
                         days=("data_date", "nunique"), tag_id=("tag_id", "first"))
                    .reset_index()
                )
                pu_summary = add_rate_cols(pu_summary)
                pu_summary = pu_summary.sort_values("friends", ascending=False)
                # PUラベルが空なら tag_name の最後部分を使用
                pu_summary["pu_label"] = pu_summary["pu_label"].apply(
                    lambda x: x if x else "(デフォルト)")

                # PU比較チャート
                if len(pu_summary) > 1:
                    pc1, pc2 = st.columns(2)
                    with pc1:
                        fig_bar = px.bar(
                            pu_summary.sort_values("friends", ascending=True),
                            x="friends", y="pu_label", orientation="h",
                            title="PU別 友だち追加",
                            labels={"friends": "友だち追加", "pu_label": "PU"},
                            color="ctr", color_continuous_scale="RdYlGn",
                        )
                        fig_bar.update_layout(
                            height=max(250, len(pu_summary) * 35 + 80),
                            margin=dict(l=20, r=20, t=40, b=20),
                            coloraxis_colorbar_title="CTR%",
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with pc2:
                        fig_ctr = px.bar(
                            pu_summary.sort_values("ctr", ascending=True),
                            x="ctr", y="pu_label", orientation="h",
                            title="PU別 CTR(%)",
                            labels={"ctr": "PU CTR(%)", "pu_label": "PU"},
                            color="friends", color_continuous_scale="Viridis",
                        )
                        fig_ctr.update_layout(
                            height=max(250, len(pu_summary) * 35 + 80),
                            margin=dict(l=20, r=20, t=40, b=20),
                            coloraxis_colorbar_title="友だち",
                        )
                        st.plotly_chart(fig_ctr, use_container_width=True)

                # PUテーブル
                pu_display = fmt_pct_cols(pu_summary.copy())
                pu_display = pu_display.rename(columns={
                    "pu_label": "PU訴求", "tag_id": "Tag ID", **rate_col_rename,
                })
                st.dataframe(
                    pu_display[["PU訴求"] + rate_col_order + ["Tag ID"]],
                    use_container_width=True,
                    height=min(400, len(pu_summary) * 40 + 60),
                )

    # --- Charts ---
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown(
            '<p class="section-header">📈 日別推移</p>', unsafe_allow_html=True
        )

        metric_option = st.selectbox(
            "指標を選択",
            ["友だち追加", "LP imp", "PU imp", "PU Click", "PU CTR (%)"],
            key="daily_metric",
        )

        metric_map = {
            "友だち追加": "added_friends",
            "LP imp": "lp_imp",
            "PU imp": "pu_imp",
            "PU Click": "pu_click",
            "PU CTR (%)": "pu_ctr",
        }
        metric_col = metric_map[metric_option]

        if "data_date" in df.columns:
            channel_colors = {"SEO": "#11998e", "アド": "#F2994A", "リス": "#667eea"}

            if metric_col == "pu_ctr":
                daily_ch = df.groupby(["data_date", "channel"]).agg(
                    pu_click=("pu_click", "sum"), pu_imp=("pu_imp", "sum")
                ).reset_index()
                daily_ch["pu_ctr"] = daily_ch.apply(
                    lambda r: r["pu_click"] / r["pu_imp"] * 100
                    if r["pu_imp"] > 0 else 0, axis=1,
                )
            else:
                daily_ch = df.groupby(["data_date", "channel"])[metric_col].sum().reset_index()

            daily_ch = daily_ch.sort_values("data_date")

            fig_daily = px.line(
                daily_ch,
                x="data_date",
                y=metric_col,
                color="channel",
                title=f"{metric_option} の日別推移（チャネル別）",
                labels={"data_date": "日付", metric_col: metric_option, "channel": "チャネル"},
                color_discrete_map=channel_colors,
            )
            fig_daily.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            fig_daily.update_traces(line=dict(width=2.5))
            st.plotly_chart(fig_daily, use_container_width=True)

    with chart_col2:
        st.markdown(
            '<p class="section-header">🏆 記事別ランキング</p>',
            unsafe_allow_html=True,
        )

        rank_metric = st.selectbox(
            "ランキング指標",
            ["友だち追加", "LP imp", "PU Click", "PU CTR (%)"],
            key="rank_metric",
        )

        rank_col = metric_map[rank_metric]

        if rank_col == "pu_ctr":
            tag_rank = df.groupby("tag_name").agg(
                pu_click=("pu_click", "sum"), pu_imp=("pu_imp", "sum")
            ).reset_index()
            tag_rank["pu_ctr"] = tag_rank.apply(
                lambda r: r["pu_click"] / r["pu_imp"] * 100 if r["pu_imp"] > 0 else 0,
                axis=1,
            )
        else:
            tag_rank = (
                df.groupby("tag_name")[rank_col].sum().reset_index()
            )

        tag_rank = tag_rank.sort_values(rank_col, ascending=True).tail(15)

        # タグ名を短く表示
        tag_rank["tag_short"] = tag_rank["tag_name"].apply(
            lambda x: x if len(str(x)) <= 30 else "..." + str(x)[-27:]
        )

        fig_rank = px.bar(
            tag_rank,
            x=rank_col,
            y="tag_short",
            orientation="h",
            title=f"記事別 {rank_metric} TOP15",
            labels={rank_col: rank_metric, "tag_short": "記事"},
            color=rank_col,
            color_continuous_scale="Viridis",
        )
        fig_rank.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    # --- AI Assist ---
    st.markdown(
        '<p class="section-header">🤖 AIアシスト</p>', unsafe_allow_html=True
    )

    if st.button("🧠 AI分析を実行", type="primary"):
        with st.spinner("分析中..."):
            insights = get_ai_insights(df)
        st.markdown(
            f"""<div class="ai-box">
            <h4>🤖 AI マーケティングアシスタント</h4>
            {insights}
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        # デフォルトでルールベースの洞察を表示
        insights = get_rule_based_insights(df)
        st.markdown(
            f"""<div class="ai-box">
            <h4>🤖 自動分析レポート</h4>
            {insights}
            </div>""",
            unsafe_allow_html=True,
        )

    # --- Channel breakdown ---
    st.markdown("")
    ch_col1, ch_col2, ch_col3 = st.columns(3)

    channel_colors = {"SEO": "#11998e", "アド": "#F2994A", "リス": "#667eea"}

    if "channel" in df.columns:
        ch_summary = df.groupby("channel").agg(
            lp_imp=("lp_imp", "sum"),
            pu_imp=("pu_imp", "sum"),
            pu_click=("pu_click", "sum"),
            friends=("added_friends", "sum"),
        ).reset_index()
        ch_summary["ctr"] = ch_summary.apply(
            lambda r: r["pu_click"] / r["pu_imp"] * 100 if r["pu_imp"] > 0 else 0, axis=1
        )
        ch_summary["display_rate"] = ch_summary.apply(
            lambda r: r["pu_imp"] / r["lp_imp"] * 100 if r["lp_imp"] > 0 else 0, axis=1
        )

        with ch_col1:
            st.markdown(
                '<p class="section-header">📊 チャネル別 友だち追加</p>',
                unsafe_allow_html=True,
            )
            if ch_summary["friends"].sum() > 0:
                fig_ch_pie = px.pie(
                    ch_summary, values="friends", names="channel", hole=0.4,
                    color="channel", color_discrete_map=channel_colors,
                )
                fig_ch_pie.update_layout(height=320, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_ch_pie, use_container_width=True)
            else:
                st.info("友だち追加データがありません")

        with ch_col2:
            st.markdown(
                '<p class="section-header">📊 チャネル別 KPI比較</p>',
                unsafe_allow_html=True,
            )
            fig_ch_bar = go.Figure()
            for _, row in ch_summary.iterrows():
                fig_ch_bar.add_trace(go.Bar(
                    name=row["channel"],
                    x=["LP imp", "PU imp", "PU Click", "友だち追加"],
                    y=[row["lp_imp"], row["pu_imp"], row["pu_click"], row["friends"]],
                    marker_color=channel_colors.get(row["channel"], "#999"),
                ))
            fig_ch_bar.update_layout(
                barmode="group", height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_ch_bar, use_container_width=True)

        with ch_col3:
            st.markdown(
                '<p class="section-header">📊 チャネル別 CTR・表示率</p>',
                unsafe_allow_html=True,
            )
            fig_ch_ctr = go.Figure()
            fig_ch_ctr.add_trace(go.Bar(
                name="PU CTR (%)",
                x=ch_summary["channel"], y=ch_summary["ctr"],
                marker_color="#F2994A",
            ))
            fig_ch_ctr.add_trace(go.Bar(
                name="PU表示率 (%)",
                x=ch_summary["channel"], y=ch_summary["display_rate"],
                marker_color="#11998e",
            ))
            fig_ch_ctr.update_layout(
                barmode="group", height=320,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_ch_ctr, use_container_width=True)

    # --- Detail Table ---
    st.markdown(
        '<p class="section-header">📋 詳細データテーブル</p>', unsafe_allow_html=True
    )

    # 表示カラム選択
    display_cols = [
        "data_date",
        "channel",
        "account",
        "site",
        "article",
        "pu_label",
        "lp_imp",
        "pu_imp",
        "pu_click",
        "pu_rate",
        "added_friends",
        "pu_ctr",
        "pu_display_rate",
        "tag_id",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    col_labels = {
        "data_date": "日付",
        "channel": "チャネル",
        "account": "ジャンル",
        "site": "サイト",
        "article": "記事",
        "pu_label": "PU",
        "lp_imp": "LP imp",
        "pu_imp": "PU imp",
        "pu_click": "PU Click",
        "pu_rate": "PU Rate",
        "added_friends": "友だち追加",
        "pu_ctr": "PU CTR(%)",
        "pu_display_rate": "PU表示率(%)",
        "tag_id": "Tag ID",
    }

    df_display = df[available_cols].copy()
    df_display = df_display.rename(
        columns={k: v for k, v in col_labels.items() if k in available_cols}
    )

    st.dataframe(
        df_display.sort_values("日付" if "日付" in df_display.columns else available_cols[0], ascending=False),
        use_container_width=True,
        height=400,
    )

    # CSV download
    csv = df_display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 CSVダウンロード",
        csv,
        "line_dashboard_export.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
