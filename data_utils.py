import os
import json
from typing import List, Dict, Optional

ABCD_MAP = {
    "A": {
        "adaptive": {
            1: "(1) Calm/ laid back", 3: "(3) Sad, Emotional pain, grieving",
            5: "(5) Content, happy, joy, hopeful", 7: "(7) Vigor / energetic",
            9: "(9) Justifiable anger/ assertive anger, justifiable outrage",
            11: "(11) Proud", 13: "(13) Feel loved, belong"
        },
        "maladaptive": {
            2: "(2) Anxious/ fearful/ tense", 4: "(4) Depressed, despair, hopeless",
            6: "(6) Mania", 8: "(8) Apathic, don't care, blunted",
            10: "(10) Angry (aggression), disgust, contempt",
            12: "(12) Ashamed, guilty", 14: "(14) Feel lonely"
        }
    },
    "B-O": {
        "adaptive": {1: "(1) Relating behavior", 3: "(3) Autonomous or adaptive control behavior"},
        "maladaptive": {2: "(2) Fight or flight behavior", 4: "(4) Over controlled or controlling behavior"}
    },
    "B-S": {
        "adaptive": {1: "(1) Self care and improvement"},
        "maladaptive": {2: "(2) Self harm, neglect and avoidance"}
    },
    "C-O": {
        "adaptive": {1: "(1) Perception of the other as related", 3: "(3) Perception of the other as facilitating autonomy needs"},
        "maladaptive": {2: "(2) Perception of the other as detached or over attached", 4: "(4) Perception of the other as blocking autonomy needs"}
    },
    "C-S": {
        "adaptive": {1: "(1) Self-acceptance and compassion"},
        "maladaptive": {2: "(2) Self criticism"}
    },
    "D": {
        "adaptive": {1: "(1) Relatedness", 3: "(3) Autonomy and adaptive control", 5: "(5) Competence, self esteem, self-care"},
        "maladaptive": {2: "(2) Expectation that relatedness needs will not be met", 4: "(4) Expectation that autonomy needs will not be met", 6: "(6) Expectation that competence needs will not be met"}
    }
}


class Task12DataLoader:
    def __init__(self, task12_dir: str,
                 task1_pred_path: Optional[str] = None,
                 task2_pred_path: Optional[str] = None):
        self.posts: Dict[tuple, Dict] = {}
        self._load_raw_dir(task12_dir)
        if task1_pred_path:
            self._load_task1_pred(task1_pred_path)
        if task2_pred_path:
            self._load_task2_pred(task2_pred_path)

    def _load_raw_dir(self, directory: str):
        if not os.path.isdir(directory):
            return
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(directory, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tid = data.get("timeline_id")
                if not tid:
                    continue
                for post in data.get("posts", []):
                    pid = post.get("post_id")
                    if pid:
                        self.posts[(tid, pid)] = {
                            "timeline_id": tid, "post_id": pid,
                            "post_index": post.get("post_index"),
                            "date": post.get("date"), "post": post.get("post", ""),
                            "Well-being": post.get("Well-being"),
                            "Switch": post.get("Switch"), "Escalation": post.get("Escalation"),
                            "evidence": post.get("evidence", {})
                        }
            except Exception as e:
                print(f"[WARN] Skip {fname}: {e}")

    def _load_task1_pred(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        for p in preds:
            tid, pid = p.get("timeline_id"), p.get("post_id")
            if not tid or not pid:
                continue
            key = (tid, pid)
            if key not in self.posts:
                self.posts[key] = {"timeline_id": tid, "post_id": pid}
            evidence = {}
            for state_key in ["adaptive-state", "maladaptive-state"]:
                if state_key not in p:
                    continue
                src = p[state_key]
                valence = "adaptive" if state_key == "adaptive-state" else "maladaptive"
                dst = {"Presence": src.get("Presence")}
                for elem in ["A", "B-S", "B-O", "C-S", "C-O", "D"]:
                    if elem in src and isinstance(src[elem], dict) and "subelement" in src[elem]:
                        num = src[elem]["subelement"]
                        text = ABCD_MAP.get(elem, {}).get(valence, {}).get(num, f"({num}) Unknown")
                        dst[elem] = {"Category": text}
                evidence[state_key] = dst
            self.posts[key]["evidence"] = evidence

    def _load_task2_pred(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        for p in preds:
            key = (p.get("timeline_id"), p.get("post_id"))
            if key in self.posts:
                self.posts[key]["Switch"] = p.get("Switch")
                self.posts[key]["Escalation"] = p.get("Escalation")

    def get_post(self, tid: str, pid: str) -> Optional[Dict]:
        return self.posts.get((tid, pid))


class CLPsychDataMerger:
    def __init__(self, task3_path: str, loader: Task12DataLoader):
        with open(task3_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        try:
            self.task3 = json.loads(content)
            if not isinstance(self.task3, list):
                raise ValueError("must be list")
        except Exception:
            objects = []
            depth, start = 0, -1
            for i, ch in enumerate(content):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start != -1:
                        try:
                            obj = json.loads(content[start:i + 1])
                            if isinstance(obj, dict) and 'timeline_id' in obj:
                                objects.append(obj)
                        except Exception:
                            pass
            self.task3 = objects
            print(f"[INFO] Fallback parse: {len(self.task3)} sequences")
        self.loader = loader

    def get_posts_for_sequence(self, seq: Dict) -> List[Dict]:
        tid = seq["timeline_id"]
        pids = seq.get("postids", [])
        posts = []
        for pid in pids:
            post = self.loader.get_post(tid, pid)
            if post:
                posts.append(post)
        posts.sort(key=lambda x: x.get("post_index") or 0)
        return posts

    @staticmethod
    def format_post_block(post: Dict, include_text: bool = True) -> str:
        lines = []
        lines.append(f"--- Post #{post.get('post_index', '?')} (ID: {post.get('post_id', '?')}) ---")
        wb = post.get("Well-being")
        if wb is not None:
            lines.append(f"Well-being: {wb}/10")
        lines.append(f"Change: Switch={post.get('Switch', '?')}, Escalation={post.get('Escalation', '?')}")
        for state_key, label in [("adaptive-state", "Adaptive"), ("maladaptive-state", "Maladaptive")]:
            state = post.get("evidence", {}).get(state_key, {})
            if not state:
                continue
            has_elem = any(k != "Presence" for k in state.keys())
            if not has_elem:
                continue
            lines.append(f"[{label}] Presence: {state.get('Presence', '?')}/5")
            for elem in ["A", "B-S", "B-O", "C-S", "C-O", "D"]:
                val = state.get(elem)
                if isinstance(val, dict):
                    lines.append(f"  ({elem}): {val.get('Category', '')}")
        if include_text:
            txt = post.get("post", "").replace("\n", " ")
            lines.append(f"Text: {txt}")
        return "\n".join(lines)

    def build_prompt(self, seq: Dict, posts: List[Dict],
                     few_shots: Optional[List[Dict]] = None,
                     change_type: Optional[str] = None) -> str:
        system = (
            "You are an expert clinical psychologist using the MIND framework. "
            "Analyze the given social media sequence and generate a structured summary (max 350 words). "
            "The summary MUST: "
            "1) Describe the central recurring theme using ABCD subelements (A, B-S, B-O, C-S, C-O, D); "
            "2) Describe dynamics WITHIN adaptive/maladaptive self-states (e.g., mutually reinforcing, co-activation); "
            "3) Describe dynamics BETWEEN self-states (e.g., dominance, suppression, reflective dialogue); "
            "4) Explicitly state the change direction (improvement/deterioration) and type (Switch/Escalation); "
            "5) Use exact abbreviations: (A), (B-S), (B-O), (C-S), (C-O), (D). "
            "Do NOT include post text in the final JSON submission."
        )
        user_parts = []
        user_parts.append(f"Timeline: {seq['timeline_id']} | Sequence: {seq['sequence_id']}")
        user_parts.append(f"Change Type: {change_type if change_type else 'Unknown (infer from sequence)'}")

        if few_shots:
            user_parts.append("\n=== Reference Examples ===")
            for i, ex in enumerate(few_shots, 1):
                ct = ex.get("change_type", "Unknown")
                summ = ex.get("summary", "")
                user_parts.append(f"[Example {i} - {ct}]\n{summ[:400]}...")

        user_parts.append("\n=== Sequence Posts ===")
        for p in posts:
            user_parts.append(self.format_post_block(p, include_text=True))
        user_parts.append("\nGenerate the structured summary:")

        user_text = "\n".join(user_parts)
        prompt = (
            "<|im_start|>system\n" + system + "<|im_end|>\n"
            "<|im_start|>user\n" + user_text + "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return prompt