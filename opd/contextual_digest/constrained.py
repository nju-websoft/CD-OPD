import json
import torch

STRUCTURAL_PROTOCOLS = {'cd_opd_contextual_digest_v1'}


def compile_graph_schema(compiler, schema):
    import xgrammar as xgr
    def syntax_schema(value):
        if isinstance(value, list):
            return [syntax_schema(x) for x in value]
        if isinstance(value, dict):
            return {key: syntax_schema(item) for key, item in value.items() if key not in {'minLength', 'maxLength'}}
        return value
    grammar = xgr.Grammar.from_json_schema(syntax_schema(schema), max_whitespace_cnt=2)
    return compiler.compile_grammar(grammar)


class GraphLogitsProcessor:
    def __init__(self, compiler, schema, tokenizer, *, check_source_states=False):
        import xgrammar as xgr
        compiled = compile_graph_schema(compiler, schema)
        self.xgr = xgr
        self.matcher = xgr.GrammarMatcher(compiled)
        self.bitmask = xgr.allocate_token_bitmask(1, compiled.tokenizer_info.vocab_size)
        self.prompt_length = None
        self.output_tokens = []

    def __call__(self, input_ids, scores):
        if input_ids.shape[0] != 1:
            raise ValueError('Graph processing expects one request')
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        new = input_ids[0, self.prompt_length + len(self.output_tokens):].tolist()
        for token in new:
            if not self.matcher.accept_token(token):
                raise RuntimeError('Graph token violates its JSON grammar')
        self.output_tokens.extend(new)
        if not self.matcher.is_terminated():
            self.matcher.fill_next_token_bitmask(self.bitmask, 0)
        self.xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
        return scores
