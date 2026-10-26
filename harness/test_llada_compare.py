"""Offline tests: saved-run checks, exact-ID joins, the AND rule and LLaDA parity."""
import unittest

from .data import read_jsonl
from .run import digest
from .diffusion.interface import t1_message
from .prompts import TEMPLATES
from .llada_compare import DEFAULT_AR, DEFAULT_LLADA, compare, load_baseline, load_llada, parity

EAGER_AR = 'results/bf16/ar-sanity-bf16-20261020-015535'


class LLaDATests(unittest.TestCase):
    def test_saved_runs_load_and_inputs_exclude_labels(self):
        records, bank_hash, _, scores = load_baseline('bank/sanity/sanity_set.jsonl', DEFAULT_AR)
        self.assertEqual(len(records), 50)
        self.assertEqual(set(scores['A']), {r['id'] for r in records})
        self.assertTrue(all('label' not in r and 'note' not in r for r in records))
        lm, llada = load_llada(DEFAULT_LLADA, bank_hash)
        self.assertEqual(set(llada), {r['id'] for r in records})
        self.assertEqual(lm['answer_slots'], 1)

    def test_eager_attention_run_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'SDPA'):
            load_baseline('bank/sanity/sanity_set.jsonl', EAGER_AR)

    def test_exact_id_join_and_pair_acceptance(self):
        records = [dict(id=str(i), domain=d, problem='p', candidate_answer='a')
                   for i, d in enumerate(['math', 'math', 'code', 'code'])]
        labels = {'0': 1, '1': 0, '2': 1, '3': 0}
        values = {'Qwen': [2, 2, 2, -2], 'Mistral': [2, -2, 2, 2], 'LLaDA': [2, -2, -2, -2]}
        scores = {n: {str(i): dict(status='completed', score=v) for i, v in enumerate(vs)} for n, vs in values.items()}
        _, pairs, review = compare(records, labels, scores)
        ql = next(r for r in pairs if r['pair'] == 'Qwen+LLaDA' and r['domain'] == 'all')
        self.assertEqual((ql['tp'], ql['fp'], ql['fn']), (1, 0, 1))
        self.assertEqual(len(review), 4)
        scores['LLaDA'].pop('3')
        with self.assertRaisesRegex(ValueError, 'ID mismatch'):
            compare(records, labels, scores)


class ParityTests(unittest.TestCase):
    def setUp(self):
        _, self.bank_hash, _, _ = load_baseline('bank/sanity/sanity_set.jsonl', DEFAULT_AR)
        self.lm, self.llada = load_llada(DEFAULT_LLADA, self.bank_hash)

    def rows(self, shift=0.0, **override):
        # harness/diffusion's score-row format (harness/diffusion/interface.py).
        return [dict(candidate_id=cid, bank_hash=self.bank_hash, model='llada', revision=self.lm['model_revision'], template='T1',
                     n_slots=self.lm['answer_slots'], score=row['score'] + shift, status='ok', **override)
                for cid, row in self.llada.items()]

    def test_identical_scores_pass_and_shifted_scores_fail(self):
        self.assertTrue(parity(self.rows(), self.lm, self.llada, self.bank_hash)['passed'])
        shifted = parity(self.rows(shift=0.2), self.lm, self.llada, self.bank_hash)
        self.assertFalse(shifted['passed'])
        self.assertAlmostEqual(shifted['max_abs_score_delta'], 0.2)
        self.assertAlmostEqual(shifted['spearman'], 1.0)  # same ranking, but the scores moved

    def test_wrong_settings_or_missing_items_are_refused(self):
        rows = self.rows()
        rows[0]['n_slots'] = 8
        with self.assertRaisesRegex(ValueError, 'n_slots'):
            parity(rows, self.lm, self.llada, self.bank_hash)
        with self.assertRaisesRegex(ValueError, 'ID mismatch'):
            parity(self.rows()[1:], self.lm, self.llada, self.bank_hash)
        with self.assertRaisesRegex(ValueError, 'bank_hash'):
            parity(self.rows(), self.lm, self.llada, 'another-bank')
        with self.assertRaisesRegex(ValueError, 'answer_prefix'):
            parity(self.rows(answer_prefix='think_nl'), self.lm, self.llada, self.bank_hash)

    def test_saved_scores_match_their_manifest(self):
        raw = open(f'{DEFAULT_LLADA}/scores_llada.jsonl', 'rb').read()
        self.assertEqual(digest(raw), self.lm['outputs_sha256']['scores_llada.jsonl'])
        self.assertEqual(len(read_jsonl(f'{DEFAULT_LLADA}/scores_llada.jsonl')[0]), 50)


class TemplateTests(unittest.TestCase):
    def test_diffusion_t1_is_the_ar_t1(self):
        problem, candidate = 'What is 2 + 2?', 'It is 4.\nThe final answer is \\boxed{4}.'
        self.assertEqual(t1_message(problem, candidate),
                         TEMPLATES['T1'].format(problem=problem, candidate_answer=candidate))


if __name__ == '__main__':
    unittest.main()
