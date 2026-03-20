import pandas as pd

from src.config import DATA_RAW, DATA_INTERIM, STOCK_CODE


def main() -> None:
    price_path = DATA_RAW / f"stock_price_{STOCK_CODE}.csv"
    events_path = DATA_INTERIM / "events.csv"

    price = pd.read_csv(price_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    events = pd.read_csv(events_path, parse_dates=["event_date"])

    windows = []
    for _, row in events.iterrows():
        t0 = row["event_date"]
        matched_index = price.index[price["date"] == t0]
        if len(matched_index) == 0:
            continue

        idx = matched_index[0]
        if idx < 1 or idx + 3 >= len(price):
            continue

        window = price.iloc[idx - 1 : idx + 4].copy()
        window["event_date"] = t0
        window["event_type"] = row.get("event_type", 1)
        windows.append(window)

    if not windows:
        raise ValueError("没有构造出任何事件窗口，请检查事件日期与股价数据是否对齐。")

    event_windows = pd.concat(windows, ignore_index=True)
    output_path = DATA_INTERIM / "event_windows.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    event_windows.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"已保存事件窗口: {output_path}")
    print(event_windows.head())


if __name__ == "__main__":
    main()
