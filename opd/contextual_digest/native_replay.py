import torch


class FirstTokenReplay:
    def __init__(self, token):
        self.token = token

    def __call__(self, output_ids, logits):
        if not output_ids:
            logits.fill_(-torch.inf)
            logits[self.token] = 0
        return logits


def install_prefix_replay(replays, strip_request_id, reset_private_state, runner_class=None):


    if runner_class is None:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        runner_class = GPUModelRunner
    original = runner_class._sample

    def sample(self, logits, spec_decode_metadata):
        first = []
        for row, request_id in enumerate(self.input_batch.req_ids):
            request_id = strip_request_id(request_id)
            token = replays.get(request_id)
            generated = (int(self.input_batch.num_tokens_no_spec[row])
                         - int(self.input_batch.num_prompt_tokens[row]))
            if token is not None and generated == 0:
                first.append((row, request_id, token))
                if spec_decode_metadata is None:
                    FirstTokenReplay(token)([], logits[row])
        output = original(self, logits, spec_decode_metadata)
        for row, request_id, token in first:


            output.sampled_token_ids[row].fill_(-1)
            output.sampled_token_ids[row, 0] = token
            reset_private_state(request_id)
            if spec_decode_metadata is not None:
                if self._verl_opd_pending_mask_rows is None:
                    self._verl_opd_pending_mask_rows = [None] * len(self.input_batch.req_ids)
                self._verl_opd_pending_mask_rows[row] = {
                    'tokens': [token], 'mask': [False], 'actions': [], 'teacher_action_logps': []}
        return output

    runner_class._sample = sample
