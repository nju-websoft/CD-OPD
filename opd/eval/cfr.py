import argparse
import json
from pathlib import Path


def response_key(row):
    if 'question_id' in row and 'sample_id' in row:
        return str(row['question_id']), row['sample_id']
    if 'problem_idx' in row and 'req_idx' in row:
        return row['problem_idx'], row['req_idx']
    raise ValueError('Each response needs question_id/sample_id or problem_idx/req_idx')


def index_responses(rows):
    result = {}
    for row in rows:
        key = response_key(row)
        score = row.get('correct')
        if not isinstance(score, (bool, int, float)) or score not in (0, 1):
            raise ValueError('Correctness must be binary')
        if key in result:
            raise ValueError('Duplicate paired-response key')
        result[key] = bool(score)
    return result


def collaboration_failure_rate(student, teacher, takeover):
    student, teacher, takeover = map(index_responses, (student, teacher, takeover))
    if student.keys() != teacher.keys() or student.keys() != takeover.keys():
        raise ValueError('CFR requires identical paired response keys in all three conditions')
    eligible = [key for key in student if student[key] and teacher[key]]
    failures = sum(not takeover[key] for key in eligible)
    return dict(paired_responses=len(student), both_correct=len(eligible), failures=failures,
                cfr=failures / len(eligible) if eligible else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('student', 'teacher', 'takeover'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rows = [[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            for path in (args.student, args.teacher, args.takeover)]
    print(json.dumps(collaboration_failure_rate(*rows), indent=2))


if __name__ == '__main__':
    main()
