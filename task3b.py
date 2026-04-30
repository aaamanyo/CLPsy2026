import json
import re
from collections import Counter
from typing import List, Dict
from config import TASK3_PRED_PATH, TASK3B_PRED_PATH, OUTPUT_DIR
import os

class SignatureExtractor:
    def __init__(self, task31_path: str, task3_train_path: str):
        with open(task31_path, "r", encoding="utf-8") as f:
            self.preds = json.load(f)
        with open(task3_train_path, "r", encoding="utf-8") as f:
            self.train = json.load(f)
        self.seq_change_type = {item["sequence_id"]: item.get("change_type", "Switch") for item in self.train}

    def extract_patterns(self, texts: List[str]) -> Counter:
        patterns = Counter()
        for text in texts:
            elems = re.findall(r'\(([A-Z]-[SO]|A|D)\)', text)
            dynamics = re.findall(r'(mutually reinforcing|reflective dialogue|overshadow|suppress|co-activation|dominance)', text, re.I)
            for e in elems:
                patterns[e] += 1
            for d in dynamics:
                patterns[d.lower()] += 1
        return patterns

    def generate_signature(self, texts: List[str], sig_type: str) -> str:
        if not texts:
            return f"Many sequences show recurrent patterns of {sig_type}."
        pat = self.extract_patterns(texts)
        top_elems = [k for k, _ in pat.most_common(6) if re.match(r'^\([A-Z]', k)]
        top_dyn = [k for k, _ in pat.most_common(6) if k in 
                   ["mutually reinforcing", "reflective dialogue", "overshadow", "suppress", "co-activation", "dominance"]]
        elems_str = ", ".join(top_elems[:4]) if top_elems else "ABCD elements"
        dyn_str = top_dyn[0] if top_dyn else "intensifying interactions"
        template = (
            f"Many sequences of {sig_type} show a progression where {elems_str} engage in {dyn_str}, "
            f"reflecting shifting dominance between self-states and culminating in significant {sig_type}."
        )
        words = template.split()
        if len(words) > 90:
            template = " ".join(words[:90])
        return template

    def run(self):
        det_texts, imp_texts = [], []
        det_evid, imp_evid = [], []

        for pred in self.preds:
            sid, tid, summ = pred["sequence_id"], pred["timeline_id"], pred["summary"].lower()
            is_det = any(w in summ for w in ["deterioration", "deteriorate", "worsening", "severe", "suicidal", "hopeless", "dominant maladaptive"])
            is_imp = any(w in summ for w in ["improvement", "improve", "recovery", "contentment", "dominant adaptive", "self-compassion"])

            if is_det and not is_imp:
                det_texts.append(pred["summary"]); det_evid.append({"timeline_id": tid, "sequence_id": sid})
            elif is_imp and not is_det:
                imp_texts.append(pred["summary"]); imp_evid.append({"timeline_id": tid, "sequence_id": sid})
            else:
                train_item = next((x for x in self.train if x["sequence_id"] == sid), None)
                if train_item and "deterioration" in train_item.get("summary", "").lower():
                    det_texts.append(pred["summary"]); det_evid.append({"timeline_id": tid, "sequence_id": sid})
                elif train_item and "improvement" in train_item.get("summary", "").lower():
                    imp_texts.append(pred["summary"]); imp_evid.append({"timeline_id": tid, "sequence_id": sid})

        if len(det_evid) < 2:
            for item in self.train:
                if len(det_evid) >= 2: break
                if "deterioration" in item.get("summary", "").lower():
                    det_evid.append({"timeline_id": item["timeline_id"], "sequence_id": item["sequence_id"]})
                    det_texts.append(item["summary"])

        if len(imp_evid) < 2:
            for item in self.train:
                if len(imp_evid) >= 2: break
                if "improvement" in item.get("summary", "").lower():
                    imp_evid.append({"timeline_id": item["timeline_id"], "sequence_id": item["sequence_id"]})
                    imp_texts.append(item["summary"])

        det_sig = self.generate_signature(det_texts, "deterioration")
        imp_sig = self.generate_signature(imp_texts, "improvement")

        def build_evidence(ev_list):
            return {str(i+1): ev for i, ev in enumerate(ev_list[:5])}

        output = [
            {"signature_type": "deterioration", "signature": det_sig, "evidence": build_evidence(det_evid)},
            {"signature_type": "improvement", "signature": imp_sig, "evidence": build_evidence(imp_evid)}
        ]

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(TASK3B_PRED_PATH, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        print(f">>> Task 3.2 saved to {TASK3B_PRED_PATH}")

if __name__ == "__main__":
    extractor = SignatureExtractor(TASK3_PRED_PATH, "train_task3.json")
    extractor.run()