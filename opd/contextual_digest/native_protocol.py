from collections import defaultdict


class NativeTraceStore:
    def __init__(self):
        self.tokens = defaultdict(dict)
        self.stops = defaultdict(list)

    def record(self, request_id, mask, tokens, positions, *unused):
        if not (len(mask) == len(tokens) == len(positions)):
            raise ValueError('Incomplete native Relay token provenance')
        rows = self.tokens[request_id]
        for position, token, teacher in zip(positions, tokens, mask, strict=True):
            if position < 0:
                raise ValueError('Negative native response position')
            value = (int(token), bool(teacher))
            if position in rows and rows[position] != value:
                raise ValueError('Conflicting native token provenance')
            rows[position] = value

    def stop(self, request_id, event):
        if event not in self.stops[request_id]:
            self.stops[request_id].append(dict(event))

    def consume(self, request_id, token_ids, *, limit, replay_token=None, finish_reason=None):
        if limit < 1:
            raise ValueError('Native response budget must be positive')
        tokens = list(token_ids)
        rows = self.tokens.pop(request_id, {})
        stops = self.stops.pop(request_id, [])
        if len(stops) > 1:
            raise ValueError('One native call must stop at its first completed guidance')
        for position, token in enumerate(tokens):
            if position not in rows or rows[position][0] != token:
                raise ValueError(f'Native token provenance mismatch at {position}')
        offset = int(replay_token is not None)
        if offset and (not tokens or tokens[0] != replay_token or rows[0][1]):
            raise ValueError('Private prefix replay did not reproduce its existing token')
        stop_position = int(stops[0]['response_pos']) if stops else None
        if stop_position is not None and not offset <= stop_position <= len(tokens):
            raise ValueError('Native handoff control position outside response')
        end = min(len(tokens), offset + limit)
        returned = stop_position is not None and stop_position <= end
        if returned:
            end = stop_position
        public = tokens[offset:end]
        mask = [rows[position][1] for position in range(offset, end)]
        if not public:
            raise ValueError('Native Relay emitted no public tokens')
        teacher = [i for i, value in enumerate(mask) if value]
        if teacher and teacher != list(range(teacher[0], len(public))):
            raise ValueError('Native stop-at-handoff call has a noncontiguous guidance segment')
        if returned and not teacher:
            raise ValueError('Native handoff lacks Teacher guidance')
        return {'token_ids': public, 'sources': ['teacher' if x else 'student' for x in mask],
                'returned': returned, 'guidance_start': teacher[0] if returned else None,
                'finish_reason': finish_reason, 'decoded_tokens': len(tokens),
                'private_replay_tokens': offset,
                'private_stop_position': stop_position}
