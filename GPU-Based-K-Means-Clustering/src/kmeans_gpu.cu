#include "kmeans_gpu.h"

#include <chrono>
#include <cstdio>
#include <memory>

#include "cpu_reference.h"
#include "cuda_util.cuh"
#include "kernels.cuh"
#include "launch_config.h"

namespace kmeans {
namespace gpu {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(const Clock::time_point &start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// DEC-009: the dimension-major copy is produced once on the host from the row-major input, before
// the timed loop. Doing it here rather than in a kernel keeps the device kernel set exactly as
// CUDA-002 enumerates it; DEC-009 permits either. The transpose is blocked because the naive
// version writes with stride N and misses the cache on every store once N*D leaves L2.
void transpose_to_dimension_major(const float *x_row, int64_t N, int64_t D, float *x_col)
{
    constexpr int64_t kTile = 64;
    for (int64_t i0 = 0; i0 < N; i0 += kTile) {
        const int64_t i_end = (i0 + kTile < N) ? i0 + kTile : N;
        for (int64_t j0 = 0; j0 < D; j0 += kTile) {
            const int64_t j_end = (j0 + kTile < D) ? j0 + kTile : D;
            for (int64_t i = i0; i < i_end; ++i) {
                for (int64_t j = j0; j < j_end; ++j) {
                    x_col[static_cast<size_t>(j) * static_cast<size_t>(N) +
                          static_cast<size_t>(i)] =
                        x_row[static_cast<size_t>(i) * static_cast<size_t>(D) +
                              static_cast<size_t>(j)];
                }
            }
        }
    }
}

// Device-side state of one run. Declared as a struct so the RAII members are released in reverse
// declaration order on every exit path, including an exception out of a launcher (MEM-007).
struct DeviceState {
    CudaStream stream;
    DeviceBuffer<float> x_col;
    DeviceBuffer<float> x_row;
    DeviceBuffer<float> centroids_a;
    DeviceBuffer<float> centroids_b;
    DeviceBuffer<int32_t> assignments;
    DeviceBuffer<double> sums;
    DeviceBuffer<unsigned int> counts;
    DeviceBuffer<double> partials;              // Approach B only
    DeviceBuffer<unsigned int> partial_counts;  // Approach B only
    DeviceBuffer<double> block_max;
    DeviceBuffer<ConvergenceScalars> scalars;
    PinnedBuffer<ConvergenceScalars> host_scalars;
};

// PERF-007: boundary events around the four phases, summed across iterations.
//
// The marks are recorded on the same stream as the kernels, so each pair measures exactly the
// span between two points of the launch sequence. Elapsed times are read only at a point where
// the stream has already been synchronised -- the end of an iteration, which the 16-byte
// convergence copy forces anyway (CUDA-011) -- so phase timing adds events but no extra
// synchronisation. The seven events are created once and reused: an iteration's marks are read
// before the next iteration re-records them.
//
//   m0 -> m1  accumulate (Approach A's zeroing memsets included: they are part of its cost)
//   m1 -> m2  update
//   m2 -> m3  movement + max_reduce            (the convergence test)
//   m3 -> m4  the `changed` memset + assignment
//   m5 -> m6  the initial assignment pass, outside the loop
//
// The finalization accumulate and update (section 2.8 step 1) reuse m0..m2 after the loop.
struct PhaseTimers {
    bool enabled = false;
    std::unique_ptr<CudaEvent> marks[7];
    double assign_ms = 0.0;
    double accum_ms = 0.0;
    double update_ms = 0.0;
    double conv_ms = 0.0;
    bool initial_pending = false;  // m5/m6 recorded, not yet added

    explicit PhaseTimers(bool on) : enabled(on)
    {
        if (!on) {
            return;
        }
        for (std::unique_ptr<CudaEvent> &mark : marks) {
            mark.reset(new CudaEvent());
        }
    }

    void mark(int index, cudaStream_t stream)
    {
        if (enabled) {
            marks[index]->record(stream);
        }
    }

    // m5/m6 now bracket the initial assignment pass; its elapsed time is added at the next
    // point where the stream is known to be idle.
    void initial_recorded() { initial_pending = enabled; }

    double span(int from, int to) const
    {
        return static_cast<double>(CudaEvent::elapsed_ms(*marks[from], *marks[to]));
    }

    // Called after the stream has been synchronised at the end of an iteration.
    void add_iteration()
    {
        if (!enabled) {
            return;
        }
        accum_ms += span(0, 1);
        update_ms += span(1, 2);
        conv_ms += span(2, 3);
        assign_ms += span(3, 4);
        add_initial();
    }

    // The initial pass is added at the first point where the stream is known to be idle, which
    // is the end of the first iteration, or the end of the run when max_iter is 0 (VAL-004).
    void add_initial()
    {
        if (enabled && initial_pending) {
            assign_ms += span(5, 6);
            initial_pending = false;
        }
    }

    // The finalization accumulate and update, added after the T_alg end event.
    void add_finalization()
    {
        if (!enabled) {
            return;
        }
        accum_ms += span(0, 1);
        update_ms += span(1, 2);
        add_initial();
    }
};

}  // namespace

Result run(const float *X, const float *C0, const Options &options, RunInfo *info,
           const LockstepHook &hook)
{
    const Config &cfg = options.config;
    const int64_t N = cfg.N;
    const int64_t D = cfg.D;
    const int64_t K = cfg.K;

    RunInfo local_info;
    RunInfo &out = (info != nullptr) ? *info : local_info;

    // --- ARC-04: device query, capability gate, G selection, capacity check -------------------
    out.device = query_device(options.device);
    check_shared_memory(out.device, cfg.block_size, kAssignTileK, kAssignTileD);
    const ChunkSelection chunks = select_chunks(out.device, cfg, options.reduction_cap_bytes);
    check_capacity(out.device, cfg, chunks.G);

    const bool reduction = cfg.strategy == Strategy::kReduction;
    const int64_t G = reduction ? chunks.G : 0;
    const int sm_count = out.device.sm_count;
    const int block = cfg.block_size;
    const int movement_blocks = movement_grid_size(K, block, sm_count);

    out.reduce_chunks = G;
    out.g_memory = chunks.g_memory;
    out.g_target = chunks.g_target;
    out.reduction_bytes = reduction ? chunks.required_bytes : 0;
    out.assign_grid = grid_size(N, block, sm_count);
    out.accum_grid = reduction ? grid_size(G * D, block, sm_count)
                               : grid_size(N * D, block, sm_count);
    out.movement_grid = movement_blocks;
    out.timing_valid = options.verify != Verify::kLockstep;

    const size_t nd = static_cast<size_t>(N) * static_cast<size_t>(D);
    const size_t kd = static_cast<size_t>(K) * static_cast<size_t>(D);

    // --- allocation (T_alloc, excluded from both T_alg and T_e2e per PERF-005) ----------------
    const Clock::time_point alloc_start = Clock::now();
    DeviceState dev;
    dev.x_col.allocate(nd);
    dev.x_row.allocate(nd);
    dev.centroids_a.allocate(kd);
    dev.centroids_b.allocate(kd);
    dev.assignments.allocate(static_cast<size_t>(N));
    dev.sums.allocate(kd);
    dev.counts.allocate(static_cast<size_t>(K));
    if (reduction) {
        dev.partials.allocate(static_cast<size_t>(G) * kd);
        dev.partial_counts.allocate(static_cast<size_t>(G) * static_cast<size_t>(K));
    }
    dev.block_max.allocate(static_cast<size_t>(movement_blocks));
    dev.scalars.allocate(1);
    dev.host_scalars.allocate(1);
    out.t_alloc_ms = ms_since(alloc_start);

    // MEM-006: nothing is read before it is written. The assignment array is zeroed once because
    // the initial pass writes it without reading it, and a partial final block would otherwise
    // leave untouched elements that the accumulation then reads; the scalars are zeroed so the
    // 16-byte copy never reads an uninitialised field (verified by initcheck, TEST-S03).
    dev.assignments.fill_bytes(0, dev.stream.get());
    dev.scalars.fill_bytes(0, dev.stream.get());

    if (options.poison_buffers) {
        // TEST-U06: 0xFF bytes are a quiet NaN as binary64 and 0xFFFFFFFF as uint32, so any
        // element that Stage 1 or Stage 2 fails to overwrite poisons the result visibly.
        dev.sums.fill_bytes(0xFF, dev.stream.get());
        dev.counts.fill_bytes(0xFF, dev.stream.get());
        if (reduction) {
            dev.partials.fill_bytes(0xFF, dev.stream.get());
            dev.partial_counts.fill_bytes(0xFF, dev.stream.get());
        }
    }

    // --- T_e2e starts at the layout preparation and the input copies (PERF-005, DEC-009) -----
    const Clock::time_point e2e_start = Clock::now();
    {
        std::vector<float> x_col_host(nd);
        transpose_to_dimension_major(X, N, D, x_col_host.data());
        const Clock::time_point h2d_start = Clock::now();
        dev.x_col.copy_from_host(x_col_host.data(), nd, dev.stream.get());
        dev.x_row.copy_from_host(X, nd, dev.stream.get());
        dev.centroids_a.copy_from_host(C0, kd, dev.stream.get());
        dev.stream.synchronize();  // the host transpose buffer dies at the end of this scope
        out.t_h2d_ms = ms_since(h2d_start);
    }

    float *c_cur = dev.centroids_a.get();
    float *c_next = dev.centroids_b.get();
    ConvergenceScalars *scalars = dev.scalars.get();
    ConvergenceScalars *host_scalars = dev.host_scalars.get();

    // One assignment pass with the kernel selected by SMEM-004. The two kernels have the same
    // contract -- same assignments, same change count, in-place write -- so the choice is confined
    // to this lambda and no caller downstream can tell which ran.
    const bool use_smem = cfg.assign_kernel == AssignKernel::kSharedMemory;
    auto assign_pass = [&](const float *centroids, bool count_changes) {
        if (use_smem) {
            launch_assign_smem(dev.x_col.get(), N, D, centroids, K, dev.assignments.get(),
                               &scalars->changed, count_changes, block, sm_count,
                               dev.stream.get());
        } else {
            launch_assign(dev.x_col.get(), N, D, centroids, K, dev.assignments.get(),
                          &scalars->changed, count_changes, block, sm_count, dev.stream.get());
        }
    };

    // One accumulation pass with the strategy selected by FR-018. Only this lambda differs
    // between Approach A and Approach B.
    auto accumulate = [&]() {
        if (reduction) {
            // RED-002/RED-003: Stage 1 overwrites every element of P and Q, Stage 2 every element
            // of S and n, so neither buffer needs zeroing here (MEM-005).
            launch_reduce_stage1(dev.x_row.get(), N, D, dev.assignments.get(), K, G,
                                 dev.partials.get(), dev.partial_counts.get(), block, sm_count,
                                 dev.stream.get());
            launch_reduce_stage2(dev.partials.get(), dev.partial_counts.get(), D, K, G,
                                 dev.sums.get(), dev.counts.get(), sm_count, dev.stream.get());
        } else {
            // ATOM-001: zero the shared sum and count buffers on the same stream first.
            dev.sums.fill_bytes(0, dev.stream.get());
            dev.counts.fill_bytes(0, dev.stream.get());
            launch_accum_atomic(dev.x_col.get(), N, D, dev.assignments.get(), K, dev.sums.get(),
                                dev.counts.get(), block, sm_count, dev.stream.get());
        }
    };

    CudaEvent alg_start;
    CudaEvent alg_end;
    PhaseTimers phases(options.profile_phases);

    // --- section 2.7 ---------------------------------------------------------------------------
    alg_start.record(dev.stream.get());

    // The initial assignment pass has no previous assignment, so it counts no changes.
    phases.mark(5, dev.stream.get());
    assign_pass(c_cur, false);
    phases.mark(6, dev.stream.get());
    phases.initial_recorded();

    int64_t t = 0;
    Status status = Status::ITERATION_LIMIT;
    double last_delta = 0.0;
    bool has_last_delta = false;

    while (t < cfg.max_iter) {
        // CUDA-003 launch sequence, in order, on one stream.
        phases.mark(0, dev.stream.get());
        accumulate();
        phases.mark(1, dev.stream.get());
        launch_update(dev.sums.get(), dev.counts.get(), K, D, c_cur, c_next, block, sm_count,
                      dev.stream.get());
        phases.mark(2, dev.stream.get());
        launch_movement(c_cur, c_next, dev.counts.get(), K, D, dev.block_max.get(), block,
                        sm_count, dev.stream.get());
        launch_max_reduce(dev.block_max.get(), movement_blocks, &scalars->delta, dev.stream.get());
        phases.mark(3, dev.stream.get());

        LockstepSnapshot snapshot;
        const bool lockstep = options.verify == Verify::kLockstep && static_cast<bool>(hook);
        if (lockstep) {
            // A(t) must be read before the assignment pass overwrites it in place (FR-007).
            snapshot.iteration = t;
            snapshot.a_in.resize(static_cast<size_t>(N));
            snapshot.c_in.resize(kd);
            snapshot.c_out.resize(kd);
            snapshot.counts.resize(static_cast<size_t>(K));
            snapshot.sums.resize(kd);
            dev.assignments.copy_to_host(snapshot.a_in.data(), static_cast<size_t>(N),
                                         dev.stream.get());
            KMEANS_CUDA_CHECK(cudaMemcpyAsync(snapshot.c_in.data(), c_cur, kd * sizeof(float),
                                              cudaMemcpyDeviceToHost, dev.stream.get()));
            KMEANS_CUDA_CHECK(cudaMemcpyAsync(snapshot.c_out.data(), c_next, kd * sizeof(float),
                                              cudaMemcpyDeviceToHost, dev.stream.get()));
            dev.counts.copy_to_host(snapshot.counts.data(), static_cast<size_t>(K),
                                    dev.stream.get());
            dev.sums.copy_to_host(snapshot.sums.data(), kd, dev.stream.get());
            dev.stream.synchronize();
        }

        // The change counter is zeroed immediately before the pass that fills it (FR-007). Only
        // the 8-byte counter is cleared; Delta was just written by max_reduce and must survive.
        KMEANS_CUDA_CHECK(cudaMemsetAsync(&scalars->changed, 0, sizeof(scalars->changed),
                                          dev.stream.get()));
        // c_next holds C(t+1): the pointer swap below is bookkeeping, so the assignment pass of
        // iteration t reads the buffer the update just wrote.
        assign_pass(c_next, true);
        phases.mark(4, dev.stream.get());

        // CUDA-011: the only in-loop transfer, 16 bytes into pinned host memory.
        KMEANS_CUDA_CHECK(cudaMemcpyAsync(host_scalars, scalars, sizeof(ConvergenceScalars),
                                          cudaMemcpyDeviceToHost, dev.stream.get()));
        dev.stream.synchronize();
        // The stream is idle here, so every mark of this iteration has completed and its elapsed
        // time can be read without adding a synchronisation of its own (PERF-007).
        phases.add_iteration();

        const double delta = host_scalars->delta;
        const int64_t changed = static_cast<int64_t>(host_scalars->changed);

        ++t;  // DEC-006: the count is the number of completed centroid updates
        last_delta = delta;
        has_last_delta = true;

        // DEC-002 on the host: assignment stability first, then the inclusive movement test.
        Status decided = Status::ERROR;
        if (!cfg.fixed_iters) {
            if (changed == 0) {
                decided = Status::CONVERGED_ASSIGNMENTS;
            } else if (delta <= cfg.tol) {
                decided = Status::CONVERGED_MOVEMENT;
            }
        }

        if (lockstep) {
            snapshot.a_out.resize(static_cast<size_t>(N));
            dev.assignments.copy_to_host(snapshot.a_out.data(), static_cast<size_t>(N),
                                         dev.stream.get());
            dev.stream.synchronize();
            snapshot.delta = delta;
            snapshot.changed = changed;
            hook(snapshot);
        }

        // CUDA-003: the pointer swap closes the iteration. It happens before any break so the
        // reported pair is (A(t), C(t)) with C(t) in c_cur (section 2.8 P1).
        float *swap = c_cur;
        c_cur = c_next;
        c_next = swap;

        if (decided != Status::ERROR) {
            status = decided;
            break;
        }
    }

    // --- section 2.8 finalization -------------------------------------------------------------
    // Step 1: one accumulation pass over A_final with the selected strategy, then the update
    // kernel turns (S, n) into M = fl32(S/n) for non-empty k. C_final itself is never touched:
    // the means land in the scratch centroid buffer.
    phases.mark(0, dev.stream.get());
    accumulate();
    phases.mark(1, dev.stream.get());
    launch_update(dev.sums.get(), dev.counts.get(), K, D, c_cur, c_next, block, sm_count,
                  dev.stream.get());
    phases.mark(2, dev.stream.get());
    alg_end.record(dev.stream.get());
    out.t_alg_ms = static_cast<double>(CudaEvent::elapsed_ms(alg_start, alg_end));

    // elapsed_ms above synchronised on alg_end, so the finalization marks have completed.
    phases.add_finalization();
    out.t_assign_ms = phases.assign_ms;
    out.t_accum_ms = phases.accum_ms;
    out.t_update_ms = phases.update_ms;
    out.t_conv_ms = phases.conv_ms;
    out.phases_measured = options.profile_phases;

    Result result;
    result.assignments.resize(static_cast<size_t>(N));
    result.centroids.resize(kd);
    result.counts.resize(static_cast<size_t>(K));
    std::vector<float> means(kd);

    const Clock::time_point d2h_start = Clock::now();
    dev.assignments.copy_to_host(result.assignments.data(), static_cast<size_t>(N),
                                 dev.stream.get());
    KMEANS_CUDA_CHECK(cudaMemcpyAsync(result.centroids.data(), c_cur, kd * sizeof(float),
                                      cudaMemcpyDeviceToHost, dev.stream.get()));
    KMEANS_CUDA_CHECK(cudaMemcpyAsync(means.data(), c_next, kd * sizeof(float),
                                      cudaMemcpyDeviceToHost, dev.stream.get()));
    KMEANS_CUDA_CHECK(cudaMemcpyAsync(result.counts.data(), dev.counts.get(),
                                      static_cast<size_t>(K) * sizeof(uint32_t),
                                      cudaMemcpyDeviceToHost, dev.stream.get()));
    // TEST-R02: S as the finalization pass left it, before fl32(S/n) throws away 29 of its bits.
    // Inside the T_d2h window and outside T_alg, so a capture run's T_d2h is larger by one KD
    // copy and its T_alg is unaffected; capture is used by tests, never by a benchmark row.
    if (options.capture_sums) {
        out.final_sums.resize(kd);
        dev.sums.copy_to_host(out.final_sums.data(), kd, dev.stream.get());
    }
    dev.stream.synchronize();
    out.t_d2h_ms = ms_since(d2h_start);
    out.t_e2e_ms = ms_since(e2e_start);

    // Step 2: P2 holds when every non-empty cluster's reported centroid agrees with the mean of
    // its assigned points within NUM-006. Empty clusters are not evaluated: DEC-003 retained
    // their centroid, which is not the mean of anything.
    const double s_max = s_max_from_range(options.m_range);
    bool consistent = true;
    double worst = 0.0;
    for (int64_t k = 0; k < K; ++k) {
        if (result.counts[static_cast<size_t>(k)] == 0u) {
            continue;
        }
        for (int64_t j = 0; j < D; ++j) {
            const size_t e = static_cast<size_t>(k * D + j);
            const double diff = std::fabs(static_cast<double>(result.centroids[e]) -
                                          static_cast<double>(means[e]));
            if (diff > worst) {
                worst = diff;
            }
            if (!centroid_agrees(result.centroids[e], means[e], s_max)) {
                consistent = false;
            }
        }
    }

    result.iterations = t;
    result.status = status;
    result.last_delta = last_delta;
    result.has_last_delta = has_last_delta;
    result.centroid_consistent = consistent;
    result.assignment_consistent = true;  // P1 by construction (section 2.8)
    // FR-016: J in binary64 on the host, after the download and outside every timed region.
    result.objective = cpu::objective(X, N, D, result.assignments.data(), result.centroids.data());
    out.max_centroid_difference = worst;
    return result;
}

FinalVerification verify_final(const float *X, const Result &result, const Config &config,
                               double m_range)
{
    FinalVerification v;
    const int64_t N = config.N;
    const int64_t D = config.D;
    const int64_t K = config.K;

    // P1: the reported assignment of every point must be the nearest reported centroid under the
    // reference's binary32 distance path. A disagreement is a failure unless the point is a
    // near-tie, where FMA contraction alone can reorder two candidates (NUM-005).
    for (int64_t i = 0; i < N; ++i) {
        const float *point = X + i * D;
        const int32_t expected = cpu::nearest_centroid(point, D, result.centroids.data(), K);
        const int32_t got = result.assignments[static_cast<size_t>(i)];
        if (expected == got) {
            continue;
        }
        const cpu::TieInfo tie = cpu::tie_info(point, D, result.centroids.data(), K);
        char line[256];
        std::snprintf(line, sizeof(line),
                      "point %lld: device k=%d, reference k=%d, margin %.3e, threshold %.3e%s",
                      static_cast<long long>(i), got, expected, tie.margin,
                      near_tie_threshold(D), tie.near_tie ? " (near-tie)" : " (HARD)");
        v.reports.push_back(line);
        if (tie.near_tie) {
            ++v.p1_near_tie_mismatches;
        } else {
            ++v.p1_hard_mismatches;
        }
    }

    // P2: the reference's own finalization pass must reach the same verdict as the device's flag.
    const cpu::Finalization fin =
        cpu::finalize(X, N, D, result.assignments.data(), result.centroids.data(), K, m_range);
    v.reference_p2 = fin.centroid_consistent;
    v.p2_flag_correct = (fin.centroid_consistent == result.centroid_consistent);
    v.max_centroid_difference = fin.max_centroid_difference;
    return v;
}

}  // namespace gpu
}  // namespace kmeans
