from dataclasses import dataclass
import hashlib
import json
import random
from functools import lru_cache


@lru_cache(maxsize=32)
def relay_handoff_order(count, batch_size, seed, epochs):

    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler
    indices = range(count)
    sampler = RandomSampler(indices, generator=torch.Generator().manual_seed(seed))
    loader = StatefulDataLoader(indices, batch_size=batch_size, sampler=sampler,
                                num_workers=8, drop_last=True, collate_fn=list,
                                multiprocessing_context='spawn' if torch.cuda.is_initialized() else None)


    return tuple(index for _ in range(epochs) for batch in loader for index in batch)


@dataclass
class TrainingQuestion:
    id: str
    text: str
    prompt_ids: list[int]
    ground_truth: str
    data_source: str


def load_questions(path, tokenizer, max_prompt_tokens):
    import pyarrow.parquet as pq

    questions, skipped = [], 0
    for row in pq.read_table(path).to_pylist():
        messages = row["prompt"]
        identity = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        prompt_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False,
        )
        if len(prompt_ids) > max_prompt_tokens:
            skipped += 1
            continue
        questions.append(TrainingQuestion(
            hashlib.sha256(identity.encode()).hexdigest(),


            "\n".join(m["content"] for m in messages if m["role"] == "user"),
            list(prompt_ids), row["reward_model"]["ground_truth"], row["data_source"],
        ))
    if not questions:
        raise ValueError("No eligible training questions")
    return questions, skipped


class OnlineTrainingStream:
    def __init__(self, questions, handoff_batch_size, task_batch_size, seed=42, epochs=1):
        self.questions = questions
        if handoff_batch_size <= 0 or task_batch_size < 0 or len(questions) < handoff_batch_size:
            raise ValueError("The training corpus must contain at least one full handoff batch")
        self.fingerprint = hashlib.sha256(json.dumps([
            (q.id, q.text, q.prompt_ids, q.ground_truth, q.data_source) for q in questions
        ], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.handoff_batch_size, self.task_batch_size = handoff_batch_size, task_batch_size
        self.seed, self.epochs = seed, epochs
        self.step = 0
        self.task_rng = random.Random(seed + 1)
        self.task_order = list(range(len(questions)))
        self.task_rng.shuffle(self.task_order)
        self.task_cursor = 0
        self.order = list(relay_handoff_order(len(questions), handoff_batch_size, seed, epochs))

    def __iter__(self):
        return self

    def __next__(self):
        start = self.step * self.handoff_batch_size
        if start >= len(self.order):
            raise StopIteration
        handoff = [self.questions[i] for i in self.order[start:start + self.handoff_batch_size]]
        excluded = {x.id for x in handoff}
        eligible = {q.id for q in self.questions} - excluded
        if len(eligible) < self.task_batch_size:
            raise ValueError("Not enough distinct training questions for a disjoint online task batch")
        task = []
        while len(task) < self.task_batch_size:
            if self.task_cursor == len(self.task_order):
                self.task_rng.shuffle(self.task_order)
                self.task_cursor = 0
            item = self.questions[self.task_order[self.task_cursor]]
            self.task_cursor += 1
            if item.id not in excluded:
                task.append(item)
                excluded.add(item.id)
        self.step += 1
        assert not {q.id for q in handoff} & {q.id for q in task}
        return handoff, task

    def state_dict(self):
        return {"fingerprint": self.fingerprint, "handoff_sampler": "relay_stateful_loader_8_workers_v1",
                "step": self.step, "task_order": self.task_order, "task_cursor": self.task_cursor,
                "task_rng": self.task_rng.getstate()}

    def load_state_dict(self, state):
        if state.get("handoff_sampler") != "relay_stateful_loader_8_workers_v1":
            raise ValueError("Checkpoint uses the old Python handoff shuffle; start a new aligned run")
        if state["fingerprint"] != self.fingerprint:
            raise ValueError("The training corpus or tokenization changed since this checkpoint")
        if sorted(state["task_order"]) != list(range(len(self.questions))):
            raise ValueError("Checkpoint task stream does not match the training corpus")
        self.step, self.task_cursor = state["step"], state["task_cursor"]
        self.task_order = list(state["task_order"])
        self.task_rng.setstate(state["task_rng"])
