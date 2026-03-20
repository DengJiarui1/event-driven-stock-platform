from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import DATA_INTERIM, DATA_RAW, STOCK_CODE


LOCAL_EVENT_FILE_CANDIDATES = [
    DATA_RAW / f"news_events_{STOCK_CODE}.csv",
    DATA_RAW / f"news_events_{STOCK_CODE}.xlsx",
    DATA_RAW / f"notice_events_{STOCK_CODE}.csv",
    DATA_RAW / f"notice_events_{STOCK_CODE}.xlsx",
    DATA_RAW / "news_events.csv",
    DATA_RAW / "news_events.xlsx",
    DATA_RAW / "notice_events.csv",
    DATA_RAW / "notice_events.xlsx",
]

NEGATIVE_KEYWORDS = [
    "风险",
    "处罚",
    "问询",
    "监管",
    "减持",
    "质押",
    "诉讼",
    "立案",
    "停牌",
    "亏损",
    "违约",
    "下滑",
    "暴跌",
    "异常波动",
    "澄清",
    "退市",
    "警示",
    "调查",
    "冻结",
]

POSITIVE_KEYWORDS = [
    "增持",
    "回购",
    "中标",
    "签约",
    "合作",
    "获批",
    "收购",
    "分红",
    "预增",
    "扭亏",
    "增长",
    "新高",
    "战略",
    "订单",
    "突破",
]

MANDATORY_OUTPUT_COLUMNS = [
    "event_date",
    "event_type",
    "event_source",
    "event_title",
    "event_count",
    "raw_event_date",
    "url",
]


def pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for name in candidates:
        key = str(name).strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def read_event_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        encodings = ["utf-8", "utf-8-sig", "gb18030", "gbk", "cp936"]
        errors: list[str] = []
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError as exc:
                errors.append(f"{enc}: {exc}")
            except Exception as exc:
                errors.append(f"{enc}: {type(exc).__name__}: {exc}")

        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception as exc:
            errors.append(f"latin1: {type(exc).__name__}: {exc}")
            joined = "\n".join(errors)
            raise ValueError(
                f"CSV 文件读取失败，已尝试多种常见编码，但仍无法解析: {path}\n"
                "请优先将文件另存为 UTF-8 编码 CSV，或直接使用 .xlsx。\n"
                f"已尝试编码如下:\n{joined}"
            )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"暂不支持的事件文件格式: {path.suffix}")


def normalize_stock_code(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .fillna("")
        .str.zfill(6)
    )


def load_price_calendar() -> pd.DataFrame:
    price_path = DATA_RAW / f"stock_price_{STOCK_CODE}.csv"
    if not price_path.exists():
        raise FileNotFoundError(
            f"未找到股价文件: {price_path}\n"
            "请先运行 src.data.fetch_stock_price。"
        )

    price = pd.read_csv(price_path, parse_dates=["date"])
    price = price.sort_values("date").reset_index(drop=True)
    price["date"] = pd.to_datetime(price["date"]).dt.normalize()
    return price


def resolve_local_event_file() -> Path | None:
    for path in LOCAL_EVENT_FILE_CANDIDATES:
        if path.exists():
            return path
    return None


def classify_event_type(title: str) -> int:
    title = str(title)
    if any(keyword in title for keyword in NEGATIVE_KEYWORDS):
        return 0
    if any(keyword in title for keyword in POSITIVE_KEYWORDS):
        return 1
    return 1


def align_to_next_trade_date(event_dates: pd.Series, trade_dates: pd.Series) -> pd.Series:
    trade_index = pd.Index(pd.to_datetime(trade_dates).sort_values().unique())
    normalized = pd.to_datetime(event_dates).dt.normalize()
    positions = trade_index.searchsorted(normalized, side="left")

    aligned = []
    for pos in positions:
        if pos >= len(trade_index):
            aligned.append(pd.NaT)
        else:
            aligned.append(trade_index[pos])
    return pd.Series(aligned, index=event_dates.index)


def build_events_from_local_file(event_path: Path, price: pd.DataFrame) -> pd.DataFrame:
    raw = read_event_file(event_path)
    if raw.empty:
        raise ValueError(f"事件文件为空: {event_path}")

    # 避免源文件本身有重复列名，导致后续 df["col"] 取到 DataFrame 而不是 Series
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()

    raw_date_col = pick_column(
        raw,
        [
            "raw_event_date",
            "event_date",
            "date",
            "datetime",
            "publish_time",
            "发布时间",
            "公告日期",
            "日期",
            "时间",
        ],
    )
    title_col = pick_column(
        raw,
        ["event_title", "title", "headline", "news_title", "新闻标题", "公告标题", "标题"],
    )
    source_col = pick_column(
        raw,
        ["event_source", "source", "文章来源", "公告类型", "来源", "新闻来源"],
    )
    url_col = pick_column(raw, ["url", "news_link", "link", "网址", "新闻链接", "链接"])
    stock_col = pick_column(raw, ["stock_code", "symbol", "代码", "股票代码", "关键词"])
    event_type_col = pick_column(raw, ["event_type"])

    if raw_date_col is None or title_col is None:
        raise ValueError(
            f"事件文件缺少必要字段。\n"
            f"当前文件: {event_path}\n"
            "至少需要以下两列之一:\n"
            "- 日期列: raw_event_date / event_date / date / 发布时间 / 公告日期 / 日期\n"
            "- 标题列: event_title / title / 新闻标题 / 公告标题 / 标题"
        )

    # 不直接 rename，避免原文件已经带有 raw_event_date / event_title 时产生重名列
    df = pd.DataFrame(index=raw.index)
    df["raw_event_date"] = raw[raw_date_col]
    df["event_title"] = raw[title_col]
    df["event_source"] = raw[source_col] if source_col is not None else "manual_real_news"
    df["url"] = raw[url_col] if url_col is not None else ""

    if stock_col is not None:
        df["stock_code"] = normalize_stock_code(raw[stock_col])
        df = df[df["stock_code"] == STOCK_CODE].copy()

    df["raw_event_date"] = pd.to_datetime(df["raw_event_date"], errors="coerce")
    df["event_title"] = df["event_title"].astype(str).str.strip()
    df["event_source"] = (
        df["event_source"].astype(str).str.strip().replace({"": "manual_real_news"})
    )
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df = df.dropna(subset=["raw_event_date"])
    df = df[df["event_title"].ne("")].copy()

    if df.empty:
        raise ValueError(f"清洗后没有可用事件，请检查文件内容: {event_path}")

    min_trade_date = price["date"].min()
    max_trade_date = price["date"].max()

    df["raw_event_date"] = df["raw_event_date"].dt.normalize()
    df = df[
        (df["raw_event_date"] >= min_trade_date - pd.Timedelta(days=7))
        & (df["raw_event_date"] <= max_trade_date)
    ].copy()

    if df.empty:
        raise ValueError(
            "事件日期与股价区间没有交集。\n"
            f"股价区间: {min_trade_date.date()} ~ {max_trade_date.date()}\n"
            f"事件文件: {event_path}"
        )

    df["event_date"] = align_to_next_trade_date(df["raw_event_date"], price["date"])
    df = df.dropna(subset=["event_date"]).copy()

    if df.empty:
        raise ValueError("事件日期无法对齐到任何交易日，请检查事件原始日期格式。")

    if event_type_col is not None:
        event_type_series = pd.to_numeric(raw.loc[df.index, event_type_col], errors="coerce")
        fallback = df["event_title"].map(classify_event_type)
        df["event_type"] = event_type_series.fillna(fallback).astype(int)
    else:
        df["event_type"] = df["event_title"].map(classify_event_type)

    aggregated = (
        df.sort_values(["event_date", "raw_event_date"])
        .groupby("event_date", as_index=False)
        .agg(
            event_type=("event_type", "min"),
            event_source=("event_source", lambda s: "|".join(pd.unique(s.astype(str)))[:200]),
            event_title=("event_title", lambda s: "；".join(pd.unique(s.astype(str))[:3])),
            event_count=("event_title", "size"),
            raw_event_date=("raw_event_date", "min"),
            url=("url", lambda s: " | ".join([x for x in pd.unique(s.astype(str)) if x][:3])),
        )
    )

    aggregated = aggregated.sort_values("event_date").reset_index(drop=True)
    for col in MANDATORY_OUTPUT_COLUMNS:
        if col not in aggregated.columns:
            aggregated[col] = ""
    return aggregated[MANDATORY_OUTPUT_COLUMNS]


def save_events(events: pd.DataFrame) -> Path:
    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    output_path = DATA_INTERIM / "events.csv"
    events.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    price = load_price_calendar()
    event_path = resolve_local_event_file()

    if event_path is None:
        template_path = DATA_RAW / f"news_events_{STOCK_CODE}.csv"
        raise FileNotFoundError(
            "未找到真实新闻/公告原始文件。\n\n"
            "请在 data/raw 中准备下列文件之一：\n"
            f"- {template_path.name}\n"
            f"- notice_events_{STOCK_CODE}.csv\n"
            "- news_events.csv\n"
            "- news_events.xlsx\n\n"
            "建议至少包含两列：\n"
            "1. event_date / date / 发布时间 / 公告日期 / 日期\n"
            "2. event_title / title / 新闻标题 / 公告标题 / 标题\n\n"
            "可选列：event_source、url、stock_code。\n"
            "说明：本版本已不再使用固定步长伪事件，而是读取你准备的真实新闻/公告数据。"
        )

    events = build_events_from_local_file(event_path, price)
    output_path = save_events(events)

    print(f"已保存事件表: {output_path}")
    print(f"事件源文件: {event_path}")
    print(f"事件数: {len(events)}")
    print(events.head())
    print("\n说明：event_date 已自动对齐到最近可用交易日；同一交易日的多条新闻/公告已聚合。")


if __name__ == "__main__":
    main()
