"""Offline integrity tests: exact-ID joins, first-mask scoring and AND rule."""
import unittest
from unittest.mock import patch
from .llada_compare import DEFAULT_AR, MASK_ID, choose_slots, compare, load_baseline, masked_ids, score_one

class LLaDATests(unittest.TestCase):
    def test_existing_baseline_is_complete_and_inputs_exclude_labels(self):
        records,_,_,scores=load_baseline('bank/sanity/sanity_set.jsonl',DEFAULT_AR)
        self.assertEqual(len(records),50)
        self.assertEqual(set(scores['A']),{r['id'] for r in records})
        self.assertTrue(all('label' not in r and 'note' not in r for r in records))

    def test_slot_ties_prefer_fewer_and_invalid_mass_is_rejected(self):
        self.assertEqual(choose_slots({1:.4,8:.8,32:.8}),8)
        with self.assertRaises(ValueError): choose_slots({1:float('nan'),8:.8,32:.9})

    def test_mask_position_follows_prompt(self):
        ids,pos=masked_ids([7,8,9],8)
        self.assertEqual(pos,3)
        self.assertEqual(ids,[7,8,9]+[MASK_ID]*8)
        with self.assertRaises(ValueError): masked_ids([MASK_ID],1)
        with self.assertRaises(ValueError): masked_ids([1]*1761,1)

    def test_first_mask_is_scored_instead_of_last_mask(self):
        try: import torch
        except ImportError: self.skipTest('optional CPU torch not installed')
        from types import SimpleNamespace
        class Toy:
            def __call__(self,x):
                logits=torch.zeros((1,x.shape[1],4))
                logits[0,2,1]=9
                logits[0,-1,2]=9
                return SimpleNamespace(logits=logits)
        with patch('torch.cuda.synchronize'):
            row=score_one(Toy(),[7,8],8,{'yes':{'ids':[1]},'no':{'ids':[2]}},torch.device('cpu'))
        self.assertAlmostEqual(row['score'],9,places=5)
        self.assertEqual(row['first_mask_position'],2)

    def test_exact_id_join_and_pair_acceptance(self):
        records=[dict(id=str(i),domain=d,problem='p',candidate_answer='a') for i,d in enumerate(['math','math','code','code'])]
        labels={'0':1,'1':0,'2':1,'3':0}
        values={'Qwen':[2,2,2,-2],'Mistral':[2,-2,2,2],'LLaDA':[2,-2,-2,-2]}
        scores={n:{str(i):dict(status='completed',score=v) for i,v in enumerate(vs)} for n,vs in values.items()}
        _,pairs,review=compare(records,labels,scores)
        ql=next(r for r in pairs if r['pair']=='Qwen+LLaDA' and r['domain']=='all')
        self.assertEqual((ql['tp'],ql['fp'],ql['fn']),(1,0,1))
        self.assertEqual(len(review),4)
        scores['LLaDA'].pop('3')
        with self.assertRaisesRegex(ValueError,'ID mismatch'): compare(records,labels,scores)

if __name__=='__main__': unittest.main()
