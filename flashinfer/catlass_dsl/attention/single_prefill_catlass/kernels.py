"""Dense NHD prefill on Ascend950 with FP32 online softmax and accumulation."""

import catlass.tla as tla

Q_TILE = 32
KV_TILE = 64
HEAD_DIM = 128
VECTOR_ROWS = Q_TILE // 2
FLOAT_LANES = 64
SCALE = HEAD_DIM ** -0.5


@tla.jit
def _row_major(ptr, rows, cols):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(cols, 1))
    )


@tla.jit
def _row(ptr, length):
    return tla.make_tensor(ptr, tla.make_layout(tla.make_shape(length), tla.make_stride(1)))


@tla.jit
def _scalar_at(tensor, row):
    return tla.tile_view(tensor, tla.make_shape(1), tla.make_coord(row))


@tla.jit
def _vector_at(tensor, row, column, width):
    return tla.tile_view(tensor, tla.make_shape(1, width), tla.make_coord(row, column))


@tla.jit
def _cast_contiguous(value, dtype, mask):
    narrow = value.to(
        dtype,
        tla.params.CastParams(
            reg_slot=tla.params.RegSlot.ZERO, round_mode=tla.params.RoundMode.CAST_ROUND
        ),
        mask=mask,
    )
    packed, _ = tla.deinterleave(narrow, narrow)
    return packed


@tla.jit
def _softmax_tile(scores, probs, maxima, sums, next_maxima, rescale, dtype, first):
    with tla.vec.func(mode="simd"):
        full = tla.create_mask(dtype=tla.Float32, pattern=tla.mask.ALL)
        narrow, _ = tla.update_mask(KV_TILE, dtype=dtype)
        scalar, _ = tla.update_mask(1, dtype=tla.Float32)
        broadcast = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        for row_index in tla.range(VECTOR_ROWS):
            block = _vector_at(scores, row_index, 0, KV_TILE).load()
            block = tla.mul(block, SCALE, mask=full)
            block_peak = block.reduce(tla.ReductionOp.MAX, mask=full)
            if not first:
                prior_peak = _scalar_at(maxima, row_index).load(params=broadcast)
                block_peak = tla.max(block_peak, prior_peak, mask=scalar)
            _scalar_at(next_maxima, row_index).store(
                block_peak, tla.params.UnalignStoreParams(), mask=scalar
            )
        tla.local_mem_bar(tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD)
        for r in tla.range(VECTOR_ROWS):
            score = _vector_at(scores, r, 0, KV_TILE).load()
            score = tla.mul(score, SCALE, mask=full)
            peak = _scalar_at(next_maxima, r).load(params=broadcast)
            weights = tla.exp(tla.sub(score, peak, mask=full), mask=full)
            total = weights.reduce(tla.ReductionOp.ADD, mask=full)
            correction = tla.full(0.0, dtype=tla.Float32)
            if not first:
                old_peak = _scalar_at(maxima, r).load(params=broadcast)
                old_sum = _scalar_at(sums, r).load(params=broadcast)
                correction = tla.exp(tla.sub(old_peak, peak, mask=full), mask=full)
                total = tla.add(total, tla.mul(old_sum, correction, mask=full), mask=scalar)
            _scalar_at(maxima, r).store(peak, tla.params.UnalignStoreParams(), mask=scalar)
            _scalar_at(sums, r).store(total, tla.params.UnalignStoreParams(), mask=scalar)
            _scalar_at(rescale, r).store(
                correction, tla.params.UnalignStoreParams(), mask=scalar
            )
            _vector_at(probs, r, 0, KV_TILE).store(
                _cast_contiguous(weights, dtype, full), mask=narrow
            )


@tla.jit
def _update_output(partial, accum, sums, rescale, result, dtype, first, last):
    with tla.vec.func(mode="simd"):
        full = tla.create_mask(dtype=tla.Float32, pattern=tla.mask.ALL)
        narrow, _ = tla.update_mask(FLOAT_LANES, dtype=dtype)
        broadcast = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)
        for r in tla.range(VECTOR_ROWS):
            correction = _scalar_at(rescale, r).load(params=broadcast)
            total = _scalar_at(sums, r).load(params=broadcast)
            for col in tla.range(HEAD_DIM // FLOAT_LANES):
                update = _vector_at(partial, r, col, FLOAT_LANES).load()
                state = _vector_at(accum, r, col, FLOAT_LANES)
                if not first:
                    previous = tla.mul(state.load(), correction, mask=full)
                    update = tla.add(update, previous, mask=full)
                if last:
                    normalized = tla.div(update, total, mask=full)
                    _vector_at(result, r, col, FLOAT_LANES).store(
                        _cast_contiguous(normalized, dtype, full), mask=narrow
                    )
                else:
                    state.store(update, mask=full)


@tla.kernel
def dense_prefill(
    q: tla.Tensor,
    k: tla.Tensor,
    v: tla.Tensor,
    out: tla.Tensor,
    q_length: tla.Int32,
    kv_length: tla.Int32,
    heads: tla.Int32,
    fp16: tla.Constexpr[bool],
) -> None:
    dtype = tla.Float16 if tla.const_expr(fp16) else tla.BFloat16
    step = heads * HEAD_DIM
    tasks = (q_length // Q_TILE) * heads
    tiles = kv_length // KV_TILE
    qk_ready = tla.cross_flag("qk_ready")
    probs_ready = tla.cross_flag("probs_ready")
    pv_ready = tla.cross_flag("pv_ready")
    l1_ready = tla.flag("l1_ready", tla.arch.MTE2, tla.arch.MTE1)
    l0_ready = tla.flag("l0_ready", tla.arch.MTE1, tla.arch.CUBE)
    cube_done = tla.flag("cube_done", tla.arch.CUBE, tla.arch.FIX)
    vector_done = tla.flag("vector_done", tla.arch.VECTOR, tla.arch.MTE3)
    l1_q = tla.allocate(Q_TILE * HEAD_DIM, dtype, tla.AddressSpace.l1, 512)
    l1_k = tla.allocate(HEAD_DIM * KV_TILE, dtype, tla.AddressSpace.l1, 512)
    l1_v = tla.allocate(KV_TILE * HEAD_DIM, dtype, tla.AddressSpace.l1, 512)
    l1_p = tla.allocate(Q_TILE * KV_TILE, dtype, tla.AddressSpace.l1, 512)
    l0_q = tla.allocate(Q_TILE * HEAD_DIM, dtype, tla.AddressSpace.l0a, 512)
    l0_k = tla.allocate(HEAD_DIM * KV_TILE, dtype, tla.AddressSpace.l0b, 512)
    l0_v = tla.allocate(KV_TILE * HEAD_DIM, dtype, tla.AddressSpace.l0b, 512)
    l0_p = tla.allocate(Q_TILE * KV_TILE, dtype, tla.AddressSpace.l0a, 512)
    l0_s = tla.allocate(Q_TILE * KV_TILE, tla.Float32, tla.AddressSpace.l0c, 1024)
    l0_o = tla.allocate(Q_TILE * HEAD_DIM, tla.Float32, tla.AddressSpace.l0c, 1024)
    # The same address is local to each vector subblock. SPLIT_M writes half the rows to each.
    ub_s = tla.allocate(VECTOR_ROWS * KV_TILE, tla.Float32, tla.AddressSpace.ub, 256)
    ub_p = tla.allocate(VECTOR_ROWS * KV_TILE, dtype, tla.AddressSpace.ub, 256)
    ub_partial = tla.allocate(VECTOR_ROWS * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_accum = tla.allocate(VECTOR_ROWS * HEAD_DIM, tla.Float32, tla.AddressSpace.ub, 256)
    ub_out = tla.allocate(VECTOR_ROWS * HEAD_DIM, dtype, tla.AddressSpace.ub, 256)
    ub_max = tla.allocate(VECTOR_ROWS, tla.Float32, tla.AddressSpace.ub, 256)
    ub_next_max = tla.allocate(VECTOR_ROWS, tla.Float32, tla.AddressSpace.ub, 256)
    ub_sum = tla.allocate(VECTOR_ROWS, tla.Float32, tla.AddressSpace.ub, 256)
    ub_rescale = tla.allocate(VECTOR_ROWS, tla.Float32, tla.AddressSpace.ub, 256)
    split_rows = tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)
    l1_probs = tla.make_tensor_like(l1_p, _row_major(l1_p, Q_TILE, KV_TILE), tla.arch.zN)
    q_l1 = tla.make_tensor_like(l1_q, _row_major(l1_q, Q_TILE, HEAD_DIM), tla.arch.zN)
    q_l0a = tla.make_tensor_like(l0_q, q_l1, tla.arch.zN)

    for task in tla.range(tla.arch.block_idx(), tasks, tla.arch.block_num()):
        head_offset = (task % heads) * HEAD_DIM
        q_start = (task // heads) * Q_TILE
        with tla.cube():
            q_src = tla.make_tensor(
                q.ptr + q_start * step + head_offset,
                tla.make_layout(tla.make_shape(Q_TILE, HEAD_DIM), tla.make_stride(step, 1)),
            )
            tla.copy(q_l1, q_src)
            tla.set_flag(l1_ready)
            tla.wait_flag(l1_ready)
            tla.copy(q_l0a, q_l1)
            tla.set_flag(l0_ready)
            tla.wait_flag(l0_ready)

        for tile in tla.range(tiles):
            kv_start = tile * KV_TILE * step + head_offset
            with tla.cube():
                k_src = tla.make_tensor(
                    k.ptr + kv_start,
                    tla.make_layout(tla.make_shape(HEAD_DIM, KV_TILE), tla.make_stride(1, step),
                                    layoutTag=tla.arch.ColumnMajor),
                )
                v_src = tla.make_tensor(
                    v.ptr + kv_start,
                    tla.make_layout(tla.make_shape(KV_TILE, HEAD_DIM), tla.make_stride(step, 1)),
                )
                k_l1 = tla.make_tensor_like(l1_k, k_src, tla.arch.nZ)
                v_l1 = tla.make_tensor_like(l1_v, v_src, tla.arch.zN)
                k_l0b = tla.make_tensor_like(l0_k, k_l1, tla.arch.nZ)
                v_l0b = tla.make_tensor_like(l0_v, v_l1, tla.arch.nZ)
                p_l0a = tla.make_tensor_like(l0_p, l1_probs, tla.arch.zN)
                score = _row_major(ub_s, Q_TILE, KV_TILE)
                score_l0c = tla.make_tensor_like(l0_s, score, tla.arch.L0Clayout)
                pv = _row_major(ub_partial, Q_TILE, HEAD_DIM)
                pv_l0c = tla.make_tensor_like(l0_o, pv, tla.arch.L0Clayout)

                tla.copy(k_l1, k_src)
                tla.copy(v_l1, v_src)
                tla.set_flag(l1_ready)
                tla.wait_flag(l1_ready)
                tla.copy(k_l0b, k_l1)
                tla.copy(v_l0b, v_l1)
                tla.set_flag(l0_ready)
                tla.wait_flag(l0_ready)
                tla.mmad(score_l0c, q_l0a, k_l0b, init_c=True)
                tla.set_flag(cube_done)
                tla.wait_flag(cube_done)
                tla.copy(score, score_l0c, split_rows)
                tla.cross_core_set_flag(qk_ready, tla.arch.FIX)

                tla.cross_core_wait_flag(probs_ready, tla.arch.MTE1)
                tla.copy(p_l0a, l1_probs)
                tla.set_flag(l0_ready)
                tla.wait_flag(l0_ready)
                tla.mmad(pv_l0c, p_l0a, v_l0b, init_c=True)
                tla.set_flag(cube_done)
                tla.wait_flag(cube_done)
                tla.copy(pv, pv_l0c, split_rows)
                tla.cross_core_set_flag(pv_ready, tla.arch.FIX)
                tla.pipe_barrier(tla.pipes.ALL)

            with tla.vector():
                scores = _row_major(ub_s, VECTOR_ROWS, KV_TILE)
                probs = _row_major(ub_p, VECTOR_ROWS, KV_TILE)
                result = _row_major(ub_out, VECTOR_ROWS, HEAD_DIM)
                partial = _row_major(ub_partial, VECTOR_ROWS, HEAD_DIM)
                accum = _row_major(ub_accum, VECTOR_ROWS, HEAD_DIM)
                maxima = _row(ub_max, VECTOR_ROWS)
                next_maxima = _row(ub_next_max, VECTOR_ROWS)
                sums = _row(ub_sum, VECTOR_ROWS)
                rescale = _row(ub_rescale, VECTOR_ROWS)
                subblock = tla.arch.sub_block_idx()
                l1_slice = tla.tile_view(
                    l1_probs, tla.make_shape(VECTOR_ROWS, KV_TILE), tla.make_coord(subblock, 0)
                )
                first = tile == 0
                last = tile == tiles - 1
                tla.cross_core_wait_flag(qk_ready, tla.arch.VECTOR)
                _softmax_tile(scores, probs, maxima, sums, next_maxima, rescale, dtype, first)
                tla.set_flag(vector_done)
                tla.wait_flag(vector_done)
                tla.copy(l1_slice, probs)
                tla.cross_core_set_flag(probs_ready, tla.arch.MTE3)

                tla.cross_core_wait_flag(pv_ready, tla.arch.VECTOR)
                _update_output(partial, accum, sums, rescale, result, dtype, first, last)
                if last:
                    out_dst = tla.make_tensor(
                        out.ptr + (q_start + subblock * VECTOR_ROWS) * step + head_offset,
                        tla.make_layout(tla.make_shape(VECTOR_ROWS, HEAD_DIM), tla.make_stride(step, 1)),
                    )
                    tla.set_flag(vector_done)
                    tla.wait_flag(vector_done)
                    tla.copy(out_dst, result)
                tla.pipe_barrier(tla.pipes.ALL)
