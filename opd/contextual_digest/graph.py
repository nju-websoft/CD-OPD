from dataclasses import dataclass

class GraphError(ValueError):
    pass

class SummaryBudgetExceeded(GraphError):
    pass

@dataclass(frozen=True)
class SummarySpan:
    node_id: str
    start: int
    end: int
    text: str

@dataclass(frozen=True)
class CompiledSummary:
    token_ids: tuple[int, ...]
    spans: tuple[SummarySpan, ...]
    selected_nodes: tuple[str, ...]
    rendering: str = "contextual_digest"
    references: tuple[dict, ...] = ()
