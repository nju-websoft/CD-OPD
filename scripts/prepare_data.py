import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SYSTEM = r'Please reason step by step, and put your final answer within \boxed{}.'


def convert(rows, problem_column, answer_column, data_source):
    output, seen = [], set()
    for row in rows:
        if 'prompt' in row and isinstance(row['prompt'], list):
            prompt = row['prompt']
        else:
            question = row.get(problem_column)
            if not isinstance(question, str) or not question.strip():
                raise ValueError('Every row needs a nonempty problem statement')
            prompt = [dict(role='system', content=SYSTEM), dict(role='user', content=question)]
        if any(not isinstance(m, dict) or m.get('role') not in {'system', 'user', 'assistant'}
               or not isinstance(m.get('content'), str) for m in prompt):
            raise ValueError('Invalid chat prompt')
        question = '\n'.join(m['content'] for m in prompt if m['role'] == 'user')
        if not question.strip():
            raise ValueError('Prompt must contain a user problem')
        if question in seen:
            continue
        reward = row.get('reward_model')
        answer = reward.get('ground_truth') if isinstance(reward, dict) else row.get(answer_column)
        if answer is None:
            raise ValueError('Every problem needs a reference answer')
        if isinstance(answer, (list, tuple)):
            if len(answer) != 1:
                raise ValueError('Expected one reference answer per problem')
            answer = answer[0]
        seen.add(question)
        output.append(dict(prompt=prompt,
            reward_model=dict(style='rule', ground_truth=str(answer)),
            data_source=row.get('data_source') or data_source))
    if not output:
        raise ValueError('No problems were found')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--problem-column', default='problem')
    parser.add_argument('--answer-column', default='answer')
    parser.add_argument('--data-source', default='dapo')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.input.suffix == '.parquet':
        rows = pq.read_table(args.input).to_pylist()
    elif args.input.suffix == '.jsonl':
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    else:
        raise ValueError('Input must be a Parquet or JSONL file')
    converted = convert(rows, args.problem_column, args.answer_column, args.data_source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(converted), args.output)
    print(json.dumps(dict(problems=len(converted), output=str(args.output))))


if __name__ == '__main__':
    main()
