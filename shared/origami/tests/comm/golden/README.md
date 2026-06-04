# Golden values

Reference outputs captured from the Python `origami_comms` model.
The C++ port asserts byte-identity against these.

## `layouts_grid.csv`

`(layout, num_gpus, pid, timestep, my_rank) → ScheduleEntry` for every
sampled tuple across 9 layouts × 3 world-sizes (2/4/8). 5,936 rows.

Columns:

| column     | meaning                                                |
|------------|--------------------------------------------------------|
| `layout`   | layout name (e.g. `RingAllGather`)                     |
| `num_gpus` | world size                                             |
| `pid`      | logical workgroup id (0..7 sample)                     |
| `timestep` | algorithm tick (0..num_timesteps-1)                    |
| `my_rank`  | this GPU's rank                                        |
| `link_id`  | xGMI link the WG uses (-1 = `SELF_LINK`)               |
| `peer_rank`| the peer GPU rank                                      |
| `direction`| `pull` or `push`                                       |
| `is_self`  | 0/1 — whether this timestep is local                   |
| `wg_sig`   | work-graph opcode chain, e.g. `L\|X3\|R`               |

`wg_sig` opcodes (matches `op_sig()` in test_layouts.cpp):

```
L    Load
S    Store
Sw   Store(write_through=true)
R    Reduce
P<peer>   Pull(peer)
X<peer>   Push(peer)
G<peer>   Signal(peer)
W<peer>   Wait(peer)
```

## Regenerating

```bash
# from origami_comms/ root (Python model on PYTHONPATH)
python3 origami_comms_cpp/scripts/dump_golden.py
```

Or — the inline one-shot used during M4 development — `python3` with
the dump script embedded; see git log for `M4: layouts` commit.
