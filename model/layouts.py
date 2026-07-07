"""
Collective layouts: (pid, timestep) → link.

Each layout is a closed-form function derived from actual Iris CCL kernel
loop structures. Contention (wgs_on_link) is the preimage size — computable
in O(1) for all standard patterns.

Self-timestep distinction: when a timestep maps to my_rank (reading from self),
the work graph uses Load (local HBM) instead of Pull (xGMI). Self-timesteps
don't consume xGMI bandwidth or hit MSHR limits.

Naming:
    timestep                one tick of the algorithm's inner loop — one peer
                            interaction per WG. Index range: 0..num_timesteps-1.
    chunks_per_timestep     how many timesteps the gpu_tile is divided across.
                            gpu_timestep_tile = gpu_tile / chunks_per_timestep.
                            Layouts default to 1 (whole gpu_tile per timestep);
                            chunked algorithms (ring, two-shot, a2a) override
                            with N to reflect per-step chunking.
"""

from dataclasses import dataclass
from typing import Dict, List
from .primitives import Op, Load, Store, Pull, Push, Reduce, Signal, Wait


SELF_LINK = -1  # sentinel: this timestep is local, no xGMI


@dataclass
class ScheduleEntry:
    link_id: int
    peer_rank: int
    direction: str
    work_graph: List[Op]
    is_self: bool = False


class CollectiveLayout:
    """Base class for collective layouts."""

    def link_of(self, pid: int, timestep: int, my_rank: int, num_gpus: int) -> ScheduleEntry:
        raise NotImplementedError

    def wgs_on_link(self, timestep: int, num_wgs: int, num_gpus: int) -> int:
        raise NotImplementedError

    def active_links(self, timestep: int, num_wgs: int, num_gpus: int) -> Dict[int, int]:
        raise NotImplementedError

    @property
    def num_timesteps(self) -> int:
        raise NotImplementedError

    @property
    def chunks_per_timestep(self) -> int:
        """How many timesteps the gpu_tile is divided across.

        Default 1 (= "each timestep moves the whole gpu_tile"), which preserves
        legacy behavior. Subclasses representing chunked algorithms (ring,
        two-shot, a2a) should override this to N (= num_gpus).
        """
        return 1


class AllToSameLayout(CollectiveLayout):
    """
    Pattern 1: All WGs → same link each timestep.

    Used by: one_shot AR, persistent AG, A2A (sequential).
    Every pid hits the same link at the same timestep. Different timesteps
    target different peers (sequential round-robin).

    Visits N-1 remote peers (skips self).
    """

    def __init__(self, num_gpus: int, work_graph_fn=None):
        self._num_gpus = num_gpus
        self._work_graph_fn = work_graph_fn or self._default_work_graph

    @staticmethod
    def _default_work_graph(peer, my_rank, num_gpus, is_self):
        if is_self:
            return [Load(), Store()]
        return [Pull(peer), Store()]

    def _peer_for_timestep(self, timestep, my_rank, num_gpus):
        # Skip self: visit 0 → peer (my_rank+1)%N, visit 1 → (my_rank+2)%N, ...
        return (my_rank + timestep + 1) % num_gpus

    def link_of(self, pid, timestep, my_rank, num_gpus):
        peer = self._peer_for_timestep(timestep, my_rank, num_gpus)
        is_self = (peer == my_rank)
        work = self._work_graph_fn(peer, my_rank, num_gpus, is_self)
        link = SELF_LINK if is_self else peer
        return ScheduleEntry(link_id=link, peer_rank=peer, direction="pull",
                             work_graph=work, is_self=is_self)

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        return num_wgs

    def active_links(self, timestep, num_wgs, num_gpus):
        return {timestep % (num_gpus - 1): num_wgs}

    @property
    def num_timesteps(self):
        return self._num_gpus - 1


class PidStaggeredLayout(CollectiveLayout):
    """
    Pattern 2: pid % world_size offsets starting peer.

    Used by: two_shot AR, two_shot RS.
    At any timestep, WGs distribute uniformly across all links.
    Includes self-timestep (one of N timesteps reads from local).

    Iris kernel: start_rank_idx = pid % world_size
                 remote = (start_rank_idx + i) % world_size
    """

    def __init__(self, num_gpus: int, work_graph_fn=None):
        self._num_gpus = num_gpus
        self._work_graph_fn = work_graph_fn or self._default_work_graph

    @staticmethod
    def _default_work_graph(peer, my_rank, num_gpus, is_self):
        if is_self:
            return [Load(), Reduce()]
        return [Pull(peer), Reduce()]

    def _peer_for_timestep(self, pid, timestep, my_rank, num_gpus):
        start = pid % num_gpus
        peer_idx = (start + timestep) % num_gpus
        return (my_rank + peer_idx) % num_gpus

    def link_of(self, pid, timestep, my_rank, num_gpus):
        peer = self._peer_for_timestep(pid, timestep, my_rank, num_gpus)
        is_self = (peer == my_rank)
        work = self._work_graph_fn(peer, my_rank, num_gpus, is_self)
        link = SELF_LINK if is_self else peer
        return ScheduleEntry(link_id=link, peer_rank=peer, direction="pull",
                             work_graph=work, is_self=is_self)

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        # At any timestep, 1/N of WGs hit self (local), rest spread across N-1 links
        num_links = num_gpus - 1
        remote_wgs = num_wgs * (num_gpus - 1) // num_gpus
        return max(remote_wgs // num_links, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        num_links = num_gpus - 1
        remote_wgs = num_wgs * (num_gpus - 1) // num_gpus
        per_link = max(remote_wgs // num_links, 1)
        return {i: per_link for i in range(num_links)}

    @property
    def num_timesteps(self):
        return self._num_gpus

    @property
    def chunks_per_timestep(self):
        # Each timestep visits a different peer; the gpu_tile is sliced N ways
        # so each timestep moves gpu_tile / N. Used by two-shot RS and a2a.
        return self._num_gpus


class PidPartitionedLayout(CollectiveLayout):
    """
    Pattern 3: pid // (SMS/N) permanently assigns each WG to one link.

    Used by: partitioned AG.
    No inner loop — each WG is dedicated to one destination.
    Some WGs may be assigned to self (local store).
    """

    def __init__(self, num_gpus: int, work_graph_fn=None):
        self._num_gpus = num_gpus
        self._work_graph_fn = work_graph_fn or self._default_work_graph

    @staticmethod
    def _default_work_graph(peer, my_rank, num_gpus, is_self):
        if is_self:
            return [Load(), Store()]
        return [Load(), Push(peer)]

    def link_of(self, pid, timestep, my_rank, num_gpus):
        dest = pid % num_gpus
        peer = (my_rank + dest) % num_gpus
        is_self = (peer == my_rank)
        work = self._work_graph_fn(peer, my_rank, num_gpus, is_self)
        link = SELF_LINK if is_self else peer
        return ScheduleEntry(link_id=link, peer_rank=peer, direction="push",
                             work_graph=work, is_self=is_self)

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        return max(num_wgs // num_gpus, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        per_link = max(num_wgs // num_gpus, 1)
        return {i: per_link for i in range(num_gpus - 1)}

    @property
    def num_timesteps(self):
        return 1


class RingFixedLayout(CollectiveLayout):
    """
    Pattern 4: All hops go to same next_rank. Fixed link across all timesteps.

    Used by: ring AR.
    Ring never visits self — always sends to next and receives from prev.
    """

    def __init__(self, num_gpus: int, work_graph_fn=None):
        self._num_gpus = num_gpus
        self._work_graph_fn = work_graph_fn or self._default_work_graph

    @staticmethod
    def _default_work_graph(peer, my_rank, num_gpus, is_self):
        next_rank = (my_rank + 1) % num_gpus
        prev_rank = (my_rank - 1) % num_gpus
        return [
            Load(),
            Wait(prev_rank),
            Pull(prev_rank),
            Reduce(),
            Store(),
            Signal(next_rank),
        ]

    def link_of(self, pid, timestep, my_rank, num_gpus):
        next_rank = (my_rank + 1) % num_gpus
        work = self._work_graph_fn(next_rank, my_rank, num_gpus, False)
        return ScheduleEntry(link_id=next_rank, peer_rank=next_rank,
                             direction="push", work_graph=work, is_self=False)

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        # Channels distribute across min(NCH, N-1) concurrent rings, each on
        # a different link. WGs per ring = NCH / nrings.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        return max(num_wgs // nrings, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        # Same conservation fix as RingAllGatherLayout: distribute num_wgs
        # exactly across min(num_wgs, N-1) rings — the first `extra` rings
        # carry one more WG. Previous implementation dropped WGs at high nch
        # and fabricated them at low nch; neither matches RCCL's mapping.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        base = num_wgs // nrings
        extra = num_wgs - base * nrings
        return {i: base + (1 if i < extra else 0) for i in range(nrings)}

    @property
    def num_timesteps(self):
        return self._num_gpus - 1

    @property
    def chunks_per_timestep(self):
        # Ring breaks the gpu_tile into N chunks; each step moves one chunk.
        return self._num_gpus


# ─── Standard collective constructors ───

class RingAllGatherLayout(CollectiveLayout):
    """
    Ring AllGather: N-1 steps. NCCL/RCCL assigns one ring per channel, with
    rings mapped onto the (N-1) xGMI links available from each GPU. With
    `nchannels >= N-1` every link gets at least one ring; with `nchannels <
    N-1` only `nchannels` links are used. Per-link evidence from
    rccl_master_sweep.csv (MI300X, W=8): at large messages each link
    saturates at ~9-35 GB/s depending on WG concentration, well below the
    42.86 GB/s payload cap when only one channel is on the link — see
    wiki/pages/ag-outlier-investigation.md.

    From RCCL all_gather.h: each step calls directRecvCopyDirectSend(), which:
      1. Reads from LOCAL staging buffer (prev_rank wrote via DirectWrite)
      2. Copies to local output buffer
      3. Writes to next_rank's staging buffer via xGMI (DirectWrite, Push)

    The xGMI operation from this GPU's perspective is Push (remote write).
    Remote writes are fire-and-forget — no MSHR round-trip on the writer.
    Per-WG effective rate is concentration-aware (see
    Heuristics.xgmi_write_concentration_k).

    Recv and send are pipelined within genericOp (different threads handle
    WaitRecv and WaitSend concurrently), not serialized.
    """

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def link_of(self, pid, timestep, my_rank, num_gpus):
        next_rank = (my_rank + 1) % num_gpus
        work = [
            Load(),                 # read from local staging buffer (prev wrote via DirectWrite)
            Store(),                # copy to local output buffer
            Push(next_rank),        # forward to next_rank via xGMI (fire-and-forget write)
        ]
        return ScheduleEntry(
            link_id=next_rank, peer_rank=next_rank,
            direction="push", work_graph=work, is_self=False,
        )

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        return max(num_wgs // nrings, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        # Use min(num_wgs, N-1) rings so we never fabricate WGs (the previous
        # max(num_wgs // (N-1), 1) form did, e.g. W=4 nch=1 → 3 WGs invented).
        # Distribute num_wgs as evenly as possible across nrings; the first
        # `extra` rings get one more WG than the rest so the sum equals num_wgs.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        base = num_wgs // nrings
        extra = num_wgs - base * nrings
        return {i: base + (1 if i < extra else 0) for i in range(nrings)}

    @property
    def num_timesteps(self):
        return self._num_gpus - 1

    @property
    def chunks_per_timestep(self):
        # AG convention: msg_bytes in benchmarks is the *per-rank send size*
        # (NCCL sendcount × dtype), not a total. Each of the N-1 ring steps
        # forwards one full gpu_tile (= per-rank send), so total wire per rank
        # = (N-1) · gpu_tile. No further subdivision per timestep.
        # Verified: busbw_AG = (n-1) · msg / time (matches CSV exactly).
        return 1


def allgather_layout(num_gpus: int) -> CollectiveLayout:
    return RingAllGatherLayout(num_gpus)


class RingReduceScatterLayout(CollectiveLayout):
    """
    Ring ReduceScatter: N-1 steps on 1 link.

    From RCCL reduce_scatter.h: recvReduceSend is fused (recv + reduce + send
    overlap). The recv reads from the ring buffer (prev wrote to it — local HBM),
    reduces with local input data, and pushes result to next rank's ring buffer.

    Structurally identical to AllGather ring but with Load + Reduce added.
    Total wire per GPU: (N-1) × msg/N on 1 link.
    """

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def link_of(self, pid, timestep, my_rank, num_gpus):
        next_rank = (my_rank + 1) % num_gpus
        work = [
            Load(),                 # read from ring buffer (prev wrote it)
            Reduce(),               # reduce with local input
            Store(),                # write reduced result locally
            Push(next_rank),        # forward to next rank's ring buffer
        ]
        return ScheduleEntry(
            link_id=next_rank, peer_rank=next_rank,
            direction="push", work_graph=work, is_self=False,
        )

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        # Channels distribute across min(NCH, N-1) concurrent rings, each on
        # a different link. WGs per ring = NCH / nrings.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        return max(num_wgs // nrings, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        # Same conservation fix as RingAllGatherLayout: distribute num_wgs
        # exactly across min(num_wgs, N-1) rings — the first `extra` rings
        # carry one more WG. Previous implementation dropped WGs at high nch
        # and fabricated them at low nch; neither matches RCCL's mapping.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        base = num_wgs // nrings
        extra = num_wgs - base * nrings
        return {i: base + (1 if i < extra else 0) for i in range(nrings)}

    @property
    def num_timesteps(self):
        return self._num_gpus - 1

    @property
    def chunks_per_timestep(self):
        # Ring RS: gpu_tile is the per-rank *output* (normalized convention).
        # Each of the N-1 ring steps moves one full per-rank-output chunk
        # around the ring; no further subdivision per timestep.
        # Total wire per rank = (N-1) × gpu_tile, matching busbw_RS exactly.
        return 1


def reduce_scatter_layout(num_gpus: int) -> CollectiveLayout:
    return RingReduceScatterLayout(num_gpus)


def allreduce_one_shot_layout(num_gpus: int) -> CollectiveLayout:
    def work_graph(peer, my_rank, N, is_self):
        if is_self:
            return [Load(), Reduce()]  # read local partial, no xGMI
        return [Pull(peer), Reduce()]
    return AllToSameLayout(num_gpus, work_graph_fn=work_graph)


class TwoShotAllReduceLayout(CollectiveLayout):
    """
    Two-shot AllReduce: reduce phase (N reads) + broadcast phase (N-1 writes).

    Reduce: N timesteps, pid-staggered. One visit is self (local Load, no xGMI).
    Broadcast: N-1 timesteps (skip self), pid-staggered. All are remote Push.

    Total timesteps: N + (N-1) = 2N-1.
    Remote xGMI visits: (N-1) + (N-1) = 2(N-1).
    """

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def _is_reduce_phase(self, timestep):
        return timestep < self._num_gpus

    def _peer_for_timestep(self, pid, timestep, my_rank, num_gpus):
        N = num_gpus
        start = pid % N
        if self._is_reduce_phase(timestep):
            peer_idx = (start + timestep) % N
        else:
            # Broadcast: skip self. Visit indices 0..N-2 map to N-1 remote peers.
            bcast_idx = timestep - N
            peer_idx = (start + bcast_idx) % (N - 1)
            # Offset to skip self in the peer ordering
            if (my_rank + peer_idx) % N >= my_rank:
                peer_idx = (peer_idx + 1)
        return (my_rank + peer_idx) % N

    def link_of(self, pid, timestep, my_rank, num_gpus):
        N = num_gpus
        start = pid % N

        if self._is_reduce_phase(timestep):
            peer_idx = (start + timestep) % N
            peer = (my_rank + peer_idx) % N
            is_self = (peer == my_rank)
            if is_self:
                work = [Load(), Reduce()]
            else:
                work = [Pull(peer), Reduce()]
        else:
            bcast_idx = timestep - N
            # Skip-self mapping: N-1 broadcast visits → N-1 remote peers
            peer_offset = (start + bcast_idx) % (N - 1) + 1  # +1 to skip self
            peer = (my_rank + peer_offset) % N
            is_self = False
            work = [Load(), Push(peer)]

        link = SELF_LINK if is_self else peer
        return ScheduleEntry(link_id=link, peer_rank=peer,
                             direction="pull" if self._is_reduce_phase(timestep) else "push",
                             work_graph=work, is_self=is_self)

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        N = num_gpus
        num_links = N - 1
        if self._is_reduce_phase(timestep):
            # 1/N of WGs hit self (local), rest spread across N-1 links
            remote_wgs = num_wgs * (N - 1) // N
        else:
            # All broadcast visits are remote
            remote_wgs = num_wgs
        return max(remote_wgs // num_links, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        N = num_gpus
        num_links = N - 1
        if self._is_reduce_phase(timestep):
            remote_wgs = num_wgs * (N - 1) // N
        else:
            remote_wgs = num_wgs
        per_link = max(remote_wgs // num_links, 1)
        return {i: per_link for i in range(num_links)}

    @property
    def num_timesteps(self):
        return 2 * self._num_gpus - 1

    @property
    def chunks_per_timestep(self):
        # Two-shot AR: gpu_tile is split N ways. Reduce phase: each rank owns
        # 1/N of the tile and reads that slice from each peer. Broadcast phase:
        # each rank pushes its 1/N slice to every other peer. Either way, each
        # timestep moves gpu_tile / N.
        return self._num_gpus


def allreduce_two_shot_layout(num_gpus: int) -> CollectiveLayout:
    return TwoShotAllReduceLayout(num_gpus)


class RingAllReduceLayout(CollectiveLayout):
    """
    Ring AllReduce = ReduceScatter phase (N-1 steps) + AllGather phase (N-1 steps).
    Ring never visits self — always communicates with next/prev in ring.
    """

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def _work_graph_for_visit(self, timestep, my_rank, num_gpus):
        prev = (my_rank - 1) % num_gpus
        next_ = (my_rank + 1) % num_gpus
        rs_visits = num_gpus - 1

        if timestep < rs_visits:
            return [Load(), Wait(prev), Pull(prev), Reduce(), Store(), Signal(next_)]
        else:
            return [Wait(prev), Pull(prev), Store(), Signal(next_)]

    def link_of(self, pid, timestep, my_rank, num_gpus):
        next_rank = (my_rank + 1) % num_gpus
        work = self._work_graph_for_visit(timestep, my_rank, num_gpus)
        return ScheduleEntry(
            link_id=next_rank, peer_rank=next_rank,
            direction="push", work_graph=work, is_self=False,
        )

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        # Channels distribute across min(NCH, N-1) concurrent rings, each on
        # a different link. WGs per ring = NCH / nrings.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        return max(num_wgs // nrings, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        # Same conservation fix as RingAllGatherLayout: distribute num_wgs
        # exactly across min(num_wgs, N-1) rings — the first `extra` rings
        # carry one more WG. Previous implementation dropped WGs at high nch
        # and fabricated them at low nch; neither matches RCCL's mapping.
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        base = num_wgs // nrings
        extra = num_wgs - base * nrings
        return {i: base + (1 if i < extra else 0) for i in range(nrings)}

    @property
    def num_timesteps(self):
        return 2 * (self._num_gpus - 1)

    @property
    def chunks_per_timestep(self):
        # Ring AR (RS+AG): gpu_tile is split N ways, each step moves one chunk.
        return self._num_gpus


class RingBroadcastLayout(CollectiveLayout):
    """
    Ring Broadcast: N-1 hop pipeline on 1 link.

    From RCCL broadcast.h:
      Root: directSend (or directCopySend)
      Middle GPUs: directRecvCopyDirectSend (recv + copy + forward)
      Last GPU: directRecv

    Structurally identical to AllGather ring, but:
    - AllGather: each GPU sends msg/N, each chunk traverses N-1 hops → wire = (N-1)×msg/N
    - Broadcast: root sends msg, each GPU forwards msg once → wire = msg per GPU

    Each GPU's work: Load (from staging), Store (local output), Push (forward to next).
    The pipeline means total time ≈ msg / link_bw at steady state.
    """

    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def link_of(self, pid, timestep, my_rank, num_gpus):
        next_rank = (my_rank + 1) % num_gpus
        work = [
            Load(),
            Store(),
            Push(next_rank),
        ]
        return ScheduleEntry(
            link_id=next_rank, peer_rank=next_rank,
            direction="push", work_graph=work, is_self=False,
        )

    def wgs_on_link(self, timestep, num_wgs, num_gpus):
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        return max(num_wgs // nrings, 1)

    def active_links(self, timestep, num_wgs, num_gpus):
        # Conservation-correct distribution (see RingAllGatherLayout).
        nrings = max(min(num_wgs, num_gpus - 1), 1)
        base = num_wgs // nrings
        extra = num_wgs - base * nrings
        return {i: base + (1 if i < extra else 0) for i in range(nrings)}

    @property
    def num_timesteps(self):
        return self._num_gpus - 1  # N-1 hops around the ring

    @property
    def chunks_per_timestep(self):
        # Pipelined broadcast: the full message is divided into N chunks and
        # the pipeline keeps a chunk in flight per hop. Per-rank wire ≈ msg.
        return self._num_gpus


def broadcast_layout(num_gpus: int) -> CollectiveLayout:
    return RingBroadcastLayout(num_gpus)


def allreduce_ring_layout(num_gpus: int) -> CollectiveLayout:
    return RingAllReduceLayout(num_gpus)


def alltoall_layout(num_gpus: int) -> CollectiveLayout:
    """
    AllToAll: P2P Send/Recv decomposition — all peers simultaneously.

    From RCCL enqueue.cc: AllToAll decomposes into N Send + N Recv P2P tasks,
    all dispatched concurrently. Each GPU sends msg/N to each of N-1 peers
    and receives msg/N from each, all at once. NOT a ring.

    Uses PidStaggered layout: channels spread across N-1 links simultaneously.
    1 timestep covering all peers (all concurrent).

    Work graph: Load (read local shard) + Push (write to peer) per peer.
    Self-timestep: Load + Store (local copy, no xGMI).
    """
    def work_graph(peer, my_rank, N, is_self):
        if is_self:
            return [Load(), Store()]
        return [Load(), Push(peer)]
    return PidStaggeredLayout(num_gpus, work_graph_fn=work_graph)
