from pathlib import Path
import os
import akshare as ak
import pandas as pd

from src.config import DATA_RAW, STOCK_CODE, START_DATE, END_DATE


PROXY_ENV_KEYS = [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "NO_PROXY", "no_proxy",
]


def _clear_proxy_env() -> None:
    for key in PROXY_ENV_KEYS:
        os.environ.pop(key, None)


def main() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    output_path = DATA_RAW / f"stock_price_{STOCK_CODE}.csv"

    use_cache = os.getenv("USE_LOCAL_STOCK_CACHE", "1") == "1"
    force_download = os.getenv("FORCE_DOWNLOAD_STOCK", "0") == "1"
    ignore_proxy = os.getenv("IGNORE_PROXY_FOR_AKSHARE", "1") == "1"

    if output_path.exists() and use_cache and not force_download:
        df = pd.read_csv(output_path)
        print(f"检测到本地缓存，直接复用: {output_path}")
        print(f"记录数: {len(df)}")
        print(df.head())
        print("如需重新联网抓取，请先执行: $env:FORCE_DOWNLOAD_STOCK='1'")
        return

    if ignore_proxy:
        _clear_proxy_env()
        print("已清理当前进程中的代理环境变量，准备重新抓取股价数据")

    df = ak.stock_zh_a_hist(
        symbol=STOCK_CODE,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq",
    )

    df = df.rename(
        columns={
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
    )

    df = df[["date", "open", "close", "high", "low", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"已保存股票数据: {output_path}")
    print(f"记录数: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
