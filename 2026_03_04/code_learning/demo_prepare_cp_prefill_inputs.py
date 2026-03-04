#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个可运行的 demo，用来理解 MindIE-LLM 里的:
  prepare_cp_prefill_inputs(cp_size, input_ids, position_ids, input_lengths_cumsum, input_lengths)

目标：
  1) 把 cp_load_balance_idx / cp_o_recover_idx 的“Q/weights 重排 + 输出恢复”跑出来
  2) 把 cp_kv_recover_idx 的“AllGather 后 KV 顺序恢复”跑出来
  3) 把 actual_seq_lengths_query/key + k_gather_index_(prev/next) 的“prev/next 两次 indexer”跑出来

额外（拓展：对齐真实上游路径）：
  4) 模拟 text_generator 在开启 CP 时做的 padding + per-rank 切分（GeneratorTorch._cp_partition_data），
     然后再喂给 prepare_cp_prefill_inputs，观察每个 cp_rank 的索引与重排效果。

运行：
  python3 /mnt/d/github/learning/2026_03_04/code_learning/demo_prepare_cp_prefill_inputs.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import math

def gather_list(xs: Sequence[str], idx: Sequence[int]) -> List[str]:
    return [xs[i] for i in idx]


def cumsum_int(xs: Sequence[int]) -> List[int]:
    out: List[int] = []
    s = 0
    for v in xs:
        s += int(v)
        out.append(s)
    return out


PLACEHOLDER_TOKEN = "<PAD>"


def pad_token_count_for_cp(seq_len: int, cp_size: int) -> int:
    """
    对齐 tg_infer_context_store.prepare_cp_input:
      pad_token_count = (-seq_len) % (2 * cp_size)
    """
    return (-int(seq_len)) % (2 * int(cp_size))


def cp_partition_one_request(
    tokens: List[str],
    position_ids: List[int],
    *,
    cp_size: int,
    cp_rank: int,
) -> Tuple[List[str], List[int]]:
    """
    对齐 GeneratorTorch._cp_partition_data 的“单条 request 切分”：
      num_chunks = 2 * cp_size
      chunk_length = ceil(L / num_chunks)
      取两段：
        - former: [chunk_length*cp_rank : chunk_length*(cp_rank+1)]
        - latter: [chunk_length*(num_chunks-1-cp_rank) : chunk_length*(num_chunks-cp_rank)]
    """
    L = len(tokens)
    assert L == len(position_ids)
    num_chunks = int(cp_size) * 2
    chunk_length = int(math.ceil(L / num_chunks))
    former_st = chunk_length * int(cp_rank)
    former_ed = chunk_length * (int(cp_rank) + 1)
    latter_st = chunk_length * (num_chunks - 1 - int(cp_rank))
    latter_ed = chunk_length * (num_chunks - int(cp_rank))

    out_tok = tokens[former_st:former_ed] + tokens[latter_st:latter_ed]
    out_pos = position_ids[former_st:former_ed] + position_ids[latter_st:latter_ed]
    return out_tok, out_pos


def prepare_cp_prefill_inputs_pure(
    cp_size: int,
    input_ids_len: int,
    position_ids: Sequence[int],
    input_lengths_cumsum: Sequence[int],
    input_lengths: Sequence[int],
) -> dict:
    """
    纯 Python 版 prepare_cp_prefill_inputs。
    目的不是复刻 NPU/torch 行为，而是把索引生成逻辑“完全跑通并可视化”。
    """
    cp_input_dict: dict = {}

    # 每个 request 的 chunk_len（默认按 length//2 拆）
    chunk_lengths = [int(L) // 2 for L in input_lengths]

    # 1) cp_load_balance_idx：把每个 request 的前半拼一起 + 后半拼一起
    cp_load_balance_idx_first, cp_load_balance_idx_last = [], []
    base = 0
    for length in input_lengths:
        length_range = list(range(base, base + length))
        divider = length // 2
        cp_load_balance_idx_first.extend(length_range[:divider])
        cp_load_balance_idx_last.extend(length_range[divider:])
        base += length
    cp_load_balance_idx = cp_load_balance_idx_first + cp_load_balance_idx_last
    cp_input_dict["cp_load_balance_idx"] = cp_load_balance_idx

    # 2) cp_o_recover_idx：把 [prev_out, next_out] concat 后恢复回原 packed 顺序
    cp_o_recover_idx = []
    base = 0
    chunk_lengths_sum = sum(chunk_lengths)
    for chunk_len in chunk_lengths:
        length_range = list(range(base, base + chunk_len))
        cp_o_recover_idx.extend(length_range)
        cp_o_recover_idx.extend([idx + chunk_lengths_sum for idx in length_range])
        base += chunk_len
    cp_input_dict["cp_o_recover_idx"] = cp_o_recover_idx

    # 3) cp_kv_recover_idx：用于 AllGather 后 KV 恢复“正常”顺序
    cp_kv_recover_idx = []
    req_offset = 0
    for req_chunk_len in chunk_lengths:
        gather_idx_per_chunk = [[] for _ in range(cp_size * 2)]
        for cp_rank_id in range(cp_size):
            rank_offset = cp_rank_id * input_ids_len
            gather_idx_per_chunk[cp_rank_id] = [
                rank_offset + req_offset + idx for idx in range(req_chunk_len)
            ]
            gather_idx_per_chunk[cp_size * 2 - 1 - cp_rank_id] = [
                rank_offset + req_offset + idx for idx in range(req_chunk_len, req_chunk_len * 2)
            ]
        # 纯 Python 扁平化，避免依赖 numpy
        for chunk in gather_idx_per_chunk:
            cp_kv_recover_idx.extend(chunk)
        req_offset += req_chunk_len * 2
    cp_input_dict["cp_kv_recover_idx"] = cp_kv_recover_idx

    # 4) 为 prev/next 两次 indexer 构造 query/key 的 cu_seqlens（ragged）
    input_lengths_cumsum_cp_prev = [0] * len(input_lengths_cumsum)
    input_lengths_cumsum_cp_next = [0] * len(input_lengths_cumsum)

    offset = 0
    for i in range(len(input_lengths_cumsum)):
        input_lengths_cumsum_cp_prev[i] = offset + (int(input_lengths_cumsum[i]) - offset) // 2
        input_lengths_cumsum_cp_next[i] = int(input_lengths_cumsum[i])
        offset = int(input_lengths_cumsum[i])

    # 用 position_ids 在切分点/终点处取“真实可见 KV 长度”（防止仅用 length//2 不一致）
    actual_seq_lengths_kv_cp_prev = [position_ids[x - 1] + 1 for x in input_lengths_cumsum_cp_prev]
    actual_seq_lengths_kv_cp_next = [position_ids[x - 1] + 1 for x in input_lengths_cumsum_cp_next]

    # 5) key gather index：从扁平化 KV buffer 里抽取 prev/next 对应的可见 KV 范围
    k_gather_index_prev: List[int] = []
    k_gather_index_next: List[int] = []
    k_offset = 0
    for i in range(len(input_lengths)):
        k_gather_index_prev.extend(list(range(k_offset, int(actual_seq_lengths_kv_cp_prev[i]) + k_offset)))
        k_gather_index_next.extend(list(range(k_offset, int(actual_seq_lengths_kv_cp_next[i]) + k_offset)))
        k_offset += int(input_lengths[i]) * cp_size
    cp_input_dict["k_gather_index"] = (k_gather_index_prev, k_gather_index_next)

    # 6) cu_seqlens: key/query
    cp_input_dict["actual_seq_lengths_key"] = (
        cumsum_int(actual_seq_lengths_kv_cp_prev),
        cumsum_int(actual_seq_lengths_kv_cp_next),
    )
    half_cumsum = [int(x) // 2 for x in input_lengths_cumsum]
    cp_input_dict["actual_seq_lengths_query"] = (half_cumsum, half_cumsum)

    return cp_input_dict


@dataclass
class DemoCase:
    cp_size: int
    input_lengths: List[int]  # per request


def build_packed_position_ids(input_lengths: List[int]) -> List[int]:
    # packed: [req0 positions..., req1 positions..., ...]
    out: List[int] = []
    for L in input_lengths:
        out.extend(list(range(L)))
    return out


def main() -> None:
    # 你可以改这里的例子
    case = DemoCase(cp_size=2, input_lengths=[6, 4])

    # ============================================================
    # Demo A：只看 prepare_cp_prefill_inputs（不考虑上游 CP padding/切分）
    # 便于理解索引逻辑，但不等价于“真实开启 CP 的输入形态”。
    # ============================================================
    print("## Demo A：直接对 packed 序列调用 prepare_cp_prefill_inputs（不模拟上游 CP 切分）")

    input_lengths = [int(x) for x in case.input_lengths]
    input_lengths_cumsum = cumsum_int(input_lengths)
    total_tokens = int(sum(input_lengths))
    position_ids = build_packed_position_ids(case.input_lengths)

    print("### 基本信息")
    print(f"cp_size = {case.cp_size}")
    print(f"input_lengths (per req) = {case.input_lengths}")
    print(f"total_tokens (packed) = {total_tokens}")
    print(f"input_lengths_cumsum (cu_seqlens_query) = {input_lengths_cumsum}")
    print(f"position_ids (packed) = {position_ids}")
    print()

    cp = prepare_cp_prefill_inputs_pure(
        cp_size=case.cp_size,
        input_ids_len=total_tokens,  # 等价于 input_ids.size(0)
        position_ids=position_ids,
        input_lengths_cumsum=input_lengths_cumsum,
        input_lengths=input_lengths,
    )

    print("### 1) cp_load_balance_idx：把每个 request 的前半拼一起 + 后半拼一起")
    cp_load_balance_idx = cp["cp_load_balance_idx"]
    print(f"cp_load_balance_idx = {cp_load_balance_idx}")

    q_labels: List[str] = []
    for rid, L in enumerate(case.input_lengths):
        for t in range(L):
            q_labels.append(f"q{rid}:{t}")

    q_lb = gather_list(q_labels, cp_load_balance_idx)
    print("packed q_labels =", q_labels)
    print("after gather(load_balance) =", q_lb)
    half = (len(q_lb) + 1) // 2
    q_prev = q_lb[:half]
    q_next = q_lb[half:]
    print(f"split at half={half}: prev={q_prev}, next={q_next}")
    print()

    print("### 2) cp_o_recover_idx：把 [prev_out, next_out] 拼起来后恢复回原 packed 顺序")
    cp_o_recover_idx = cp["cp_o_recover_idx"]
    print(f"cp_o_recover_idx = {cp_o_recover_idx}")
    concat_out = [f"out(prev){x}" for x in q_prev] + [f"out(next){x}" for x in q_next]
    recovered = gather_list(concat_out, cp_o_recover_idx)
    print("concat_out(prev+next) =", concat_out)
    print("after recover(gather cp_o_recover_idx) =", recovered)
    print()

    print("### 3) cp_kv_recover_idx：AllGather 后 KV 恢复“正常”顺序（demo 版）")
    cp_kv_recover_idx = cp["cp_kv_recover_idx"]
    print(f"cp_kv_recover_idx (len={len(cp_kv_recover_idx)}) = {cp_kv_recover_idx}")
    allgather_labels: List[str] = []
    for r in range(case.cp_size):
        for t in range(total_tokens):
            allgather_labels.append(f"r{r}:t{t}")
    kv_reordered = gather_list(allgather_labels, cp_kv_recover_idx)
    print("allgather_labels (rank-major) head =", allgather_labels[: min(12, len(allgather_labels))], "...")
    print("after recover(cp_kv_recover_idx) head =", kv_reordered[: min(12, len(kv_reordered))], "...")
    print()

    print("### 4) actual_seq_lengths_query/key + k_gather_index：prev/next 两次 indexer 的长度与抽取")
    aq_prev, aq_next = cp["actual_seq_lengths_query"]
    ak_prev, ak_next = cp["actual_seq_lengths_key"]
    print("actual_seq_lengths_query (cu_seqlens) prev =", aq_prev)
    print("actual_seq_lengths_query (cu_seqlens) next =", aq_next)
    print("actual_seq_lengths_key   (cu_seqlens) prev =", ak_prev)
    print("actual_seq_lengths_key   (cu_seqlens) next =", ak_next)
    k_gather_index_prev, k_gather_index_next = cp["k_gather_index"]
    print(f"k_gather_index_prev(len={len(k_gather_index_prev)}) head =", k_gather_index_prev[:20], "...")
    print(f"k_gather_index_next(len={len(k_gather_index_next)}) head =", k_gather_index_next[:20], "...")

    past_key_labels: List[str] = []
    for rid, L in enumerate(case.input_lengths):
        for r in range(case.cp_size):
            for t in range(L):
                past_key_labels.append(f"k(req{rid},r{r},t{t})")
    k_prev = gather_list(past_key_labels, k_gather_index_prev)
    k_next = gather_list(past_key_labels, k_gather_index_next)
    print("gathered key(prev) head =", k_prev[: min(18, len(k_prev))], "...")
    print("gathered key(next) head =", k_next[: min(18, len(k_next))], "...")
    print()

    # ============================================================
    # Demo B：对齐真实上游路径（padding + per-rank 切分），然后再调用 prepare_cp_prefill_inputs
    # ============================================================
    print("## Demo B：模拟上游 CP padding + per-rank 切分（GeneratorTorch._cp_partition_data）后再调用")
    print("### 对应源码位置（方便你跳转对照）")
    print("- 上游 pad_token_count / cp_tokens: MindIE-LLM/mindie_llm/text_generator/utils/tg_infer_context_store.py:27-38")
    print("- 上游 prompt padding 到 2*cp_size 倍数: MindIE-LLM/mindie_llm/connector/common/input_metadata_builder.py:712-716, 748-750")
    print("- 上游 per-rank 切分两段 chunk: MindIE-LLM/mindie_llm/text_generator/adapter/generator_torch.py:574-606")
    print("- 下游索引生成: MindIE-LLM/mindie_llm/runtime/layers/attention/backend/sparse_attention.py:93-170")
    print()
    ori_input_lengths = [int(x) for x in case.input_lengths]
    num_chunks = case.cp_size * 2
    pad_counts = [pad_token_count_for_cp(L, case.cp_size) for L in ori_input_lengths]
    padded_lengths = [L + p for L, p in zip(ori_input_lengths, pad_counts)]
    # 对齐 tg_infer_context_store.prepare_cp_input: cp_tokens = repeat(padded_len//cp_size, cp_size).reshape(-1, cp_size)
    cp_tokens: List[List[int]] = [[pl // case.cp_size for _ in range(case.cp_size)] for pl in padded_lengths]
    per_rank_lengths_per_req = [pl // case.cp_size for pl in padded_lengths]  # 等价于 cp_tokens[:, cp_rank]

    print(f"cp_size={case.cp_size}, num_chunks=2*cp_size={num_chunks}")
    print(f"ori_input_lengths(per req) = {ori_input_lengths}")
    print(f"pad_token_count(per req)   = {pad_counts}")
    print(f"padded_lengths(per req)    = {padded_lengths} (对齐到 {num_chunks} 的倍数)")
    print(f"cp_tokens (bs x cp_size)   = {cp_tokens}")
    print(f"per_rank_lengths(per req)  = {per_rank_lengths_per_req} (每个 cp_rank 都相同)")
    for r in range(case.cp_size):
        expect_ctx = sum(row[r] for row in cp_tokens)
        print(f"expect context_length sum for cp_rank={r}: {expect_ctx}")
    print()

    req_tokens_padded: List[List[str]] = []
    req_pos_padded: List[List[int]] = []
    for rid, (L, pad_cnt, padded_len) in enumerate(zip(ori_input_lengths, pad_counts, padded_lengths)):
        toks = [f"q{rid}:{t}" for t in range(L)] + [f"{PLACEHOLDER_TOKEN}{rid}" for _ in range(pad_cnt)]
        pos = list(range(padded_len))
        assert len(toks) == padded_len
        req_tokens_padded.append(toks)
        req_pos_padded.append(pos)
        print(f"req{rid} padded tokens =", toks)
        print(f"req{rid} position_ids =", pos)
    print()

    for cp_rank in range(case.cp_size):
        print(f"### ---- CP Rank {cp_rank} ----")
        per_rank_tokens: List[str] = []
        per_rank_pos: List[int] = []
        for rid in range(len(ori_input_lengths)):
            out_tok, out_pos = cp_partition_one_request(
                req_tokens_padded[rid],
                req_pos_padded[rid],
                cp_size=case.cp_size,
                cp_rank=cp_rank,
            )
            per_rank_tokens.extend(out_tok)
            per_rank_pos.extend(out_pos)

        input_lengths_rank = per_rank_lengths_per_req[:]
        input_lengths_cumsum_rank = cumsum_int(input_lengths_rank)
        total_tokens_rank = len(per_rank_tokens)

        print("per_rank_tokens(packed) =", per_rank_tokens)
        print("per_rank_position_ids   =", per_rank_pos)
        print(f"input_lengths(per req)  = {input_lengths_rank} (sum={sum(input_lengths_rank)})")
        print(f"total_tokens(per rank)  = {total_tokens_rank}")
        print()

        cp2 = prepare_cp_prefill_inputs_pure(
            cp_size=case.cp_size,
            input_ids_len=total_tokens_rank,
            position_ids=per_rank_pos,
            input_lengths_cumsum=input_lengths_cumsum_rank,
            input_lengths=input_lengths_rank,
        )

        print("#### 1) cp_load_balance_idx / cp_o_recover_idx（在 per-rank packed 上重排与恢复）")
        cp_load_balance_idx2 = cp2["cp_load_balance_idx"]
        cp_o_recover_idx2 = cp2["cp_o_recover_idx"]
        print("cp_load_balance_idx =", cp_load_balance_idx2)
        print("cp_o_recover_idx    =", cp_o_recover_idx2)

        q_lb2 = gather_list(per_rank_tokens, cp_load_balance_idx2)
        half2 = (len(q_lb2) + 1) // 2
        q_prev2 = q_lb2[:half2]
        q_next2 = q_lb2[half2:]
        concat_out2 = [f"out(prev){x}" for x in q_prev2] + [f"out(next){x}" for x in q_next2]
        recovered2 = gather_list(concat_out2, cp_o_recover_idx2)
        print("after load_balance =", q_lb2)
        print("after recover      =", recovered2)
        print()

        print("#### 2) cp_kv_recover_idx（AllGather 后长度 = cp_size * total_tokens_per_rank）")
        cp_kv_recover_idx2 = cp2["cp_kv_recover_idx"]
        print(
            f"len(cp_kv_recover_idx) = {len(cp_kv_recover_idx2)}  "
            f"(expect {case.cp_size} * {total_tokens_rank} = {case.cp_size * total_tokens_rank})"
        )
        print("cp_kv_recover_idx head =", cp_kv_recover_idx2[: min(24, len(cp_kv_recover_idx2))], "...")
        print()

        print("#### 3) actual_seq_lengths_* + k_gather_index（prev/next 两次 indexer 的长度/抽取）")
        aq_prev2, aq_next2 = cp2["actual_seq_lengths_query"]
        ak_prev2, ak_next2 = cp2["actual_seq_lengths_key"]
        k_gather_index_prev2, k_gather_index_next2 = cp2["k_gather_index"]
        print("actual_seq_lengths_query(prev/next) =", aq_prev2, aq_next2)
        print("actual_seq_lengths_key  (prev/next) =", ak_prev2, ak_next2)
        print("k_gather_index_prev head =", k_gather_index_prev2[: min(20, len(k_gather_index_prev2))], "...")
        print("k_gather_index_next head =", k_gather_index_next2[: min(20, len(k_gather_index_next2))], "...")

        past_key_labels2: List[str] = []
        for rid, L in enumerate(input_lengths_rank):
            for r in range(case.cp_size):
                for t in range(L):
                    past_key_labels2.append(f"k(req{rid},r{r},t{t})")
        k_prev2 = gather_list(past_key_labels2, k_gather_index_prev2)
        k_next2 = gather_list(past_key_labels2, k_gather_index_next2)
        print("gathered key(prev) head =", k_prev2[: min(18, len(k_prev2))], "...")
        print("gathered key(next) head =", k_next2[: min(18, len(k_next2))], "...")
        print()

    print("## 结束")
    print("你可以改 DemoCase(cp_size, input_lengths)，观察：padding→cp切分→prepare_cp_prefill_inputs 索引的全链路变化。")


if __name__ == "__main__":
    main()

