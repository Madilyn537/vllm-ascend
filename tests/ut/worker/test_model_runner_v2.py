from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import BatchReqState, GPUModelRunner

from vllm_ascend.worker.v2 import model_runner as runner_module
from vllm_ascend.worker.v2.input_batch import AscendInputBuffers
from vllm_ascend.worker.v2.model_runner import NPUModelRunner
from vllm_ascend.worker.v2.pcp_manager import AscendPCPManager


def _make_runner(need_timing: bool = True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(profiling_chunk_config=SimpleNamespace(need_timing=need_timing))
    )
    runner.vllm_config = SimpleNamespace()
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


@pytest.mark.parametrize("is_vllm_0_27_1", [True, False], ids=["v0.27.1", "newer"])
def test_execute_model_records_profiling_time(is_vllm_0_27_1):
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ) as mock_execute_model,
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_version_is",
            return_value=is_vllm_0_27_1,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch(
            "vllm_ascend.core.profiling_chunk_predictor.time.perf_counter",
            side_effect=[10.0, 10.125],
        ),
    ):
        output = runner.execute_model(scheduler_output)

    assert output is None
    assert runner._cpp_execution_time_ms == pytest.approx(125.0)
    assert mock_synchronize.call_count == 2
    expected_kwargs: dict[str, object] = {
        "intermediate_tensors": None,
        "dummy_run": False,
        "skip_attn_for_dummy_run": False,
        "is_profile": False,
    }
    if not is_vllm_0_27_1:
        expected_kwargs["context_len"] = 0
    mock_execute_model.assert_called_once_with(scheduler_output, **expected_kwargs)


def test_execute_model_disables_profiling_timer_and_clears_stale_time():
    runner = _make_runner()
    runner._cpp_execution_time_ms = 123.0
    scheduler_output = SimpleNamespace(disable_profiling_timing=True)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch("vllm_ascend.core.profiling_chunk_predictor.time.perf_counter") as mock_perf_counter,
    ):
        runner.execute_model(scheduler_output)

    profiling_config = runner.ascend_config.scheduler_config.profiling_chunk_config
    assert not profiling_config.need_timing
    assert runner._cpp_execution_time_ms is None
    mock_synchronize.assert_not_called()
    mock_perf_counter.assert_not_called()


def test_full_decode_only_keeps_graph_descriptor_request_count():
    runner = _make_runner()
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.decode_query_len = 1
    query_start_loc_np = np.array([0, 1, 2, 2, 2, 2], dtype=np.int32)

    actual, num_reqs_padded = runner._pad_query_start_loc_for_fia(
        num_tokens_padded=4,
        num_reqs_padded=4,
        num_reqs=2,
        query_start_loc_np=query_start_loc_np,
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_desc_num_reqs=4,
    )

    assert num_reqs_padded == 4
    np.testing.assert_array_equal(actual[:5], np.array([0, 1, 2, 3, 4], dtype=np.int32))


def test_sample_tokens_restores_replicated_draft_hidden_states():
    runner = _make_runner(need_timing=False)
    runner.is_last_pp_rank = True
    runner.speculator = SimpleNamespace(replicated_pcp=True)
    runner.use_spec_pp = False

    aux_hidden_states = [
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.float32).reshape(2, 2),
    ]
    state = Mock(aux_hidden_states=aux_hidden_states)
    restored_state = object()
    state._replace.return_value = restored_state
    runner.execute_model_state = state

    target_hidden_states = object()
    restored_aux_hidden_states = torch.ones(4, 5)
    runner.pcp_manager = SimpleNamespace(
        restore_hidden_state_buffer=Mock(),
        restore_hidden_states=Mock(
            return_value=restored_aux_hidden_states,
        ),
    )
    runner.model = SimpleNamespace(
        get_mtp_target_hidden_states=lambda: target_hidden_states,
    )
    grammar_output = object()
    expected_output = object()

    with patch.object(
        GPUModelRunner,
        "sample_tokens",
        return_value=expected_output,
    ) as parent_sample_tokens:
        actual = runner.sample_tokens(grammar_output)

    assert actual is expected_output
    parent_sample_tokens.assert_called_once_with(grammar_output)
    runner.pcp_manager.restore_hidden_state_buffer.assert_called_once_with(target_hidden_states)
    restored_input = runner.pcp_manager.restore_hidden_states.call_args.args[0]
    torch.testing.assert_close(
        restored_input,
        torch.cat(aux_hidden_states, dim=-1),
    )
    state._replace.assert_called_once_with(aux_hidden_states=[restored_aux_hidden_states])
    assert runner.execute_model_state is restored_state


@pytest.mark.parametrize(
    "dp_size,use_pcp,scheduled,prefilling,dispatch_tokens,padded_tokens",
    [
        (2, True, [80, 48], [True, True], 64, 64),
        (2, True, [18, 1], [True, False], 11, 11),
        (2, True, [1, 1], [False, False], 2, 4),
        (1, True, [18, 1], [True, False], 19, 19),
        (2, False, [18, 1], [True, False], 19, 19),
    ],
)
def test_pcp_dp_dispatch_preserves_full_inputs_before_partition(
    dp_size, use_pcp, scheduled, prefilling, dispatch_tokens, padded_tokens
):
    runner = _make_runner()
    runner.dp_size = dp_size
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 2
    runner.use_dcp = runner.use_pp = False
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.model_state = SimpleNamespace(num_new_sampled_tokens_per_step=1)
    runner.eplb = Mock()
    runner._update_seq_lens_cpu = Mock()
    runner.input_buffers = AscendInputBuffers(2, 128, runner.device)
    runner.input_buffers.input_ids.copy_(torch.arange(128, dtype=torch.int32))
    runner.input_buffers.positions.copy_(torch.arange(128))
    runner.req_states = SimpleNamespace(
        num_computed_tokens_np=np.array([64, 0], dtype=np.int32),
        num_computed_tokens=SimpleNamespace(gpu=torch.tensor([64, 0], dtype=torch.int32)),
        prefill_len=SimpleNamespace(gpu=torch.tensor([128, 128], dtype=torch.int32)),
        all_token_ids=SimpleNamespace(gpu=torch.zeros((2, 128), dtype=torch.int32)),
        next_prefill_tokens=torch.zeros(2, dtype=torch.int32),
        last_sampled_tokens=torch.zeros(2, dtype=torch.int32),
        draft_tokens=torch.empty((2, 0), dtype=torch.int32),
    )
    runner.pcp_manager = AscendPCPManager(2, 0, runner.device) if use_pcp else None
    state = BatchReqState(
        req_ids=["a", "b"],
        num_scheduled_tokens=np.array(scheduled, dtype=np.int32),
        num_tokens=sum(scheduled),
        idx_mapping_np=np.array([1, 0], dtype=np.intp),
        prefill_len_np=np.array([128, 128], dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([0, 64], dtype=np.int32),
        is_prefilling_np=np.array(prefilling),
        has_prefill=any(prefilling),
    )
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={}, has_structured_output_requests=False)
    uniform_count = None if state.has_prefill else 1
    with patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(state, uniform_count)):
        dispatched, actual_uniform_count = runner.gather_batch_req_state(scheduler_output, dummy_run=False)
    assert dispatched.num_tokens == dispatch_tokens
    assert actual_uniform_count == uniform_count
    assert state.num_tokens == sum(scheduled)
    np.testing.assert_array_equal(dispatched.num_scheduled_tokens, scheduled)

    def copy_to_cpu(value, out=None, device=None):
        value = torch.as_tensor(value)
        if out is not None:
            out.copy_(value)
            return out
        return value

    # Stub device kernels, but execute the actual prepare_inputs all the way
    # to its PCP boundary. Token views and request offsets must agree there.
    with (
        patch.object(runner_module, "async_copy_to_gpu", side_effect=copy_to_cpu),
        patch.object(runner_module, "build_attn_state", return_value=None),
        patch.object(runner_module, "prepare_prefill_inputs"),
        patch.object(runner_module, "prepare_pos_seq_lens"),
        patch.object(runner_module, "combine_sampled_and_draft_tokens", return_value=torch.tensor([0, 1])),
        patch.object(runner_module, "update_cos_sin"),
        patch.object(
            runner_module.vllm_model_runner.pcp, "maybe_partition_pcp_batch", side_effect=lambda manager, batch: batch
        ) as partition,
    ):
        batch = runner.prepare_inputs(
            scheduler_output,
            dispatched,
            SimpleNamespace(num_tokens=padded_tokens, num_reqs=2, cg_mode=CUDAGraphMode.NONE),
        )

    partition.assert_called_once_with(runner.pcp_manager, batch)
    expected_size = sum(scheduled) if state.has_prefill else padded_tokens
    assert batch.num_tokens == sum(scheduled)
    assert batch.num_tokens_after_padding == expected_size
    assert batch.input_ids.tolist() == list(range(expected_size))
    assert batch.positions.tolist() == list(range(expected_size))
    assert batch.is_padding.shape == (expected_size,)
    np.testing.assert_array_equal(batch.query_start_loc_np, [0, scheduled[0], sum(scheduled)])
    assert dispatched.num_tokens == dispatch_tokens


def test_pcp_dp_gather_dummy_does_not_require_request_state():
    runner = _make_runner()
    runner.dp_size = 2
    runner.pcp_manager = AscendPCPManager(2, 0, torch.device("cpu"))
    with patch.object(GPUModelRunner, "gather_batch_req_state", return_value=(None, 1)):
        assert runner.gather_batch_req_state(object(), dummy_run=True) == (None, 1)
