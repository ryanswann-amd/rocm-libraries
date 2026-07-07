"""Quick validation: predict vs measured for rccl_master_sweep.csv."""

import csv
import sys
from pathlib import Path

from model.hardware import MI300X, MI300X_COMM
from model.collective import predict_row


def main():
    data_path = Path("data/rccl_master_sweep.csv")
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    rows = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows")
    print()

    supported = {"all_reduce", "all_gather", "reduce_scatter", "all_to_all", "broadcast"}

    errors = []
    by_collective = {}

    for i, row in enumerate(rows):
        primitive = row["primitive"]
        if primitive not in supported:
            continue

        msg_bytes = int(row["msg_bytes"])
        world_size = int(row["world_size"])
        nchannels = int(row["nchannels"])
        measured_us = float(row["latency_us"])

        if measured_us <= 0 or msg_bytes == 0:
            continue

        try:
            predicted_us = predict_row(
                primitive, msg_bytes, world_size, nchannels,
                MI300X, MI300X_COMM,
            )
        except Exception as e:
            if i < 5:
                print(f"Error on row {i}: {e}")
            continue

        error_pct = (predicted_us - measured_us) / measured_us * 100

        errors.append({
            "primitive": primitive,
            "msg_bytes": msg_bytes,
            "world_size": world_size,
            "nchannels": nchannels,
            "measured_us": measured_us,
            "predicted_us": predicted_us,
            "error_pct": error_pct,
        })

        if primitive not in by_collective:
            by_collective[primitive] = []
        by_collective[primitive].append(abs(error_pct))

    print(f"Predicted {len(errors)} rows")
    print()

    # Summary by collective
    print(f"{'Collective':<20} {'Count':>6} {'MdAPE':>8} {'Mean APE':>10} {'Max APE':>10}")
    print("-" * 60)
    for coll in sorted(by_collective.keys()):
        errs = sorted(by_collective[coll])
        n = len(errs)
        median = errs[n // 2]
        mean = sum(errs) / n
        mx = errs[-1]
        print(f"{coll:<20} {n:>6} {median:>8.1f}% {mean:>9.1f}% {mx:>9.1f}%")

    # Show a few example rows
    print()
    print("Sample predictions (first 10 bandwidth-bound rows per collective):")
    print(f"{'Collective':<16} {'MsgB':>10} {'W':>3} {'NCH':>4} {'Meas(us)':>10} {'Pred(us)':>10} {'Err%':>8}")
    print("-" * 70)
    shown = {}
    for e in errors:
        coll = e["primitive"]
        if e["msg_bytes"] < 1_000_000:
            continue
        if shown.get(coll, 0) >= 5:
            continue
        shown[coll] = shown.get(coll, 0) + 1
        print(f"{coll:<16} {e['msg_bytes']:>10} {e['world_size']:>3} {e['nchannels']:>4} "
              f"{e['measured_us']:>10.1f} {e['predicted_us']:>10.1f} {e['error_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
