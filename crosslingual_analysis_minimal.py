#!/usr/bin/env python3
"""
最小可运行版：分析层跨语言关联分析脚本

功能：
1. 跨语言句级对齐（优先尝试 Bertalign，失败则回退到长度比动态规划对齐）
2. 译法差异量化（编辑距离 + LCS）
3. 核心术语英译变迁统计
4. 译者关系与版本演化轨迹挖掘

运行方式：
    python crosslingual_analysis_minimal.py --demo
    python crosslingual_analysis_minimal.py --input demo_confucius_data.json --output report.json
    python crosslingual_analysis_minimal.py --input demo_confucius_data.json --use-bertalign

输入 JSON 格式：
{
  "source_title": "论语",
  "source_sentences": ["学而时习之，不亦说乎？", "..."],
  "translations": [
    {
      "id": "legge_1861",
      "translator": "James Legge",
      "year": 1861,
      "sentences": ["Is it not pleasant to learn with a constant perseverance and application?", "..."],
      "mentor": "",
      "influenced_by": []
    }
  ],
  "core_terms": ["仁", "义"]
}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "these", "those", "as", "at", "from", "it", "its", "his", "her", "their",
    "our", "your", "you", "we", "they", "he", "she", "i", "not", "no", "do",
    "does", "did", "done", "have", "has", "had", "into", "than", "then", "so",
    "if", "but", "also", "such", "may", "can", "shall", "will", "would", "should"
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize_en(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for ca in a:
        curr = [0]
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[j - 1]))
        prev = curr
    return prev[-1]


def normalized_edit_similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return 1.0 - edit_distance(a, b) / denom


def normalized_lcs_similarity(a: str, b: str) -> float:
    denom = max(len(a), len(b), 1)
    return lcs_length(a, b) / denom


def chunk_join(parts: Sequence[str]) -> str:
    return " ".join(normalize_text(p) for p in parts if normalize_text(p))


def fallback_mn_align(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
    max_group: int = 2,
) -> List[Dict]:
    """
    简单 m:n 句对齐：允许 1:1 / 1:2 / 2:1 / 2:2，基于长度比例做动态规划。
    这不是 Bertalign，但可以作为无依赖的最小回退实现。
    """
    n = len(source_sentences)
    m = len(target_sentences)
    if n == 0 or m == 0:
        return []

    src_lens = [max(len(s), 1) for s in source_sentences]
    tgt_lens = [max(len(s), 1) for s in target_sentences]
    ratio = (sum(src_lens) / max(sum(tgt_lens), 1)) if sum(tgt_lens) else 1.0

    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    prev: List[List[Tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] == inf:
                continue
            for di in range(1, max_group + 1):
                for dj in range(1, max_group + 1):
                    ni = i + di
                    nj = j + dj
                    if ni > n or nj > m:
                        continue
                    src_len = sum(src_lens[i:ni])
                    tgt_len = sum(tgt_lens[j:nj])
                    cost = abs(src_len - tgt_len * ratio)
                    if dp[i][j] + cost < dp[ni][nj]:
                        dp[ni][nj] = dp[i][j] + cost
                        prev[ni][nj] = (i, j)

    if prev[n][m] is None:
        return []

    pairs = []
    i, j = n, m
    while (i, j) != (0, 0):
        pi, pj = prev[i][j]
        src_idx = list(range(pi, i))
        tgt_idx = list(range(pj, j))
        pairs.append({
            "source_indices": src_idx,
            "target_indices": tgt_idx,
            "source_text": chunk_join(source_sentences[k] for k in src_idx),
            "target_text": chunk_join(target_sentences[k] for k in tgt_idx),
            "method": "fallback_dp_length_ratio",
        })
        i, j = pi, pj
    pairs.reverse()
    return pairs


def try_bertalign(source_sentences: Sequence[str], target_sentences: Sequence[str]) -> List[Dict]:
    """
    可选 Bertalign 接口。
    由于不同安装版本 API 可能不同，这里只做安全尝试；失败则由上层回退。
    """
    candidates = [
        ("bertalign", "Bertalign"),
        ("bertalign.aligner", "Bertalign"),
    ]

    last_error = None
    for module_name, class_name in candidates:
        try:
            module = __import__(module_name, fromlist=[class_name])
            aligner_cls = getattr(module, class_name)
            aligner = aligner_cls(list(source_sentences), list(target_sentences))

            if hasattr(aligner, "align_sents"):
                raw = aligner.align_sents()
            elif hasattr(aligner, "align"):
                raw = aligner.align()
            else:
                raise AttributeError("Bertalign object has no align/align_sents method")

            results = []
            for item in raw:
                if isinstance(item, dict):
                    src_idx = item.get("source_indices") or item.get("src") or []
                    tgt_idx = item.get("target_indices") or item.get("tgt") or []
                else:
                    src_idx, tgt_idx = item

                src_idx = list(src_idx)
                tgt_idx = list(tgt_idx)
                results.append({
                    "source_indices": src_idx,
                    "target_indices": tgt_idx,
                    "source_text": chunk_join(source_sentences[k] for k in src_idx),
                    "target_text": chunk_join(target_sentences[k] for k in tgt_idx),
                    "method": "bertalign",
                })
            return results
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(f"Bertalign unavailable or API mismatch: {last_error}")


def align_sentences(
    source_sentences: Sequence[str],
    target_sentences: Sequence[str],
    use_bertalign: bool = False,
) -> Tuple[List[Dict], str]:
    if use_bertalign:
        try:
            return try_bertalign(source_sentences, target_sentences), "bertalign"
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Bertalign 调用失败，自动回退到最小对齐器: {exc}")
    return fallback_mn_align(source_sentences, target_sentences), "fallback_dp_length_ratio"


def build_term_evolution(
    source_sentences: Sequence[str],
    translations: Sequence[Dict],
    alignments_by_version: Dict[str, List[Dict]],
    core_terms: Sequence[str],
) -> Dict[str, List[Dict]]:
    result: Dict[str, List[Dict]] = {term: [] for term in core_terms}

    sentence_ids_by_term = defaultdict(list)
    for idx, sent in enumerate(source_sentences):
        for term in core_terms:
            if term in sent:
                sentence_ids_by_term[term].append(idx)

    for version in translations:
        version_id = version["id"]
        aligned_pairs = alignments_by_version[version_id]
        target_lookup = {}
        for pair in aligned_pairs:
            for src_idx in pair["source_indices"]:
                target_lookup.setdefault(src_idx, []).append(pair["target_text"])

        for term in core_terms:
            counter = Counter()
            for src_idx in sentence_ids_by_term.get(term, []):
                for target_text in target_lookup.get(src_idx, []):
                    counter.update(tokenize_en(target_text))
            result[term].append({
                "translator": version["translator"],
                "year": version["year"],
                "top_candidates": counter.most_common(8),
            })

    for term in result:
        result[term].sort(key=lambda x: x["year"])
    return result


def compare_translations(translations: Sequence[Dict]) -> List[Dict]:
    reports = []
    ordered = sorted(translations, key=lambda x: x["year"])
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            left = ordered[i]
            right = ordered[j]
            left_text = chunk_join(left["sentences"])
            right_text = chunk_join(right["sentences"])
            reports.append({
                "left": left["id"],
                "right": right["id"],
                "left_translator": left["translator"],
                "right_translator": right["translator"],
                "edit_similarity": round(normalized_edit_similarity(left_text, right_text), 4),
                "lcs_similarity": round(normalized_lcs_similarity(left_text, right_text), 4),
                "year_gap": right["year"] - left["year"],
            })
    return reports


def build_relation_graph(translations: Sequence[Dict]) -> Dict:
    ordered = sorted(translations, key=lambda x: x["year"])
    nodes = []
    edges = []

    for item in ordered:
        nodes.append({
            "id": item["id"],
            "translator": item["translator"],
            "year": item["year"],
        })

    for item in ordered:
        mentor = normalize_text(item.get("mentor", ""))
        if mentor:
            edges.append({
                "source": mentor,
                "target": item["translator"],
                "type": "mentor_of",
                "weight": 1.0,
            })
        for ref in item.get("influenced_by", []):
            edges.append({
                "source": ref,
                "target": item["translator"],
                "type": "influenced",
                "weight": 1.0,
            })

    for idx in range(1, len(ordered)):
        prev_item = ordered[idx - 1]
        curr_item = ordered[idx]
        prev_text = chunk_join(prev_item["sentences"])
        curr_text = chunk_join(curr_item["sentences"])
        lcs_score = normalized_lcs_similarity(prev_text, curr_text)
        edit_score = normalized_edit_similarity(prev_text, curr_text)
        edges.append({
            "source": prev_item["translator"],
            "target": curr_item["translator"],
            "type": "evolution",
            "weight": round((lcs_score + edit_score) / 2, 4),
            "year_gap": curr_item["year"] - prev_item["year"],
        })

    return {"nodes": nodes, "edges": edges}


def build_report(data: Dict, use_bertalign: bool = False) -> Dict:
    source_sentences = [normalize_text(s) for s in data["source_sentences"]]
    translations = data["translations"]
    core_terms = data.get("core_terms", ["仁", "义"])

    alignments_by_version = {}
    alignment_meta = {}

    for version in translations:
        aligned, method = align_sentences(
            source_sentences,
            version["sentences"],
            use_bertalign=use_bertalign,
        )
        alignments_by_version[version["id"]] = aligned
        alignment_meta[version["id"]] = {
            "translator": version["translator"],
            "year": version["year"],
            "method": method,
            "pair_count": len(aligned),
        }

    return {
        "source_title": data.get("source_title", "未命名文本"),
        "alignment_summary": alignment_meta,
        "alignments": alignments_by_version,
        "translation_comparisons": compare_translations(translations),
        "term_evolution": build_term_evolution(
            source_sentences,
            translations,
            alignments_by_version,
            core_terms,
        ),
        "relation_graph": build_relation_graph(translations),
    }


def demo_data() -> Dict:
    return {
        "source_title": "论语（示例）",
        "source_sentences": [
            "学而时习之，不亦说乎？",
            "有朋自远方来，不亦乐乎？",
            "人不知而不愠，不亦君子乎？",
            "君子喻于义，小人喻于利。",
            "仁者安仁，知者利仁。",
        ],
        "translations": [
            {
                "id": "legge_1861",
                "translator": "James Legge",
                "year": 1861,
                "mentor": "",
                "influenced_by": [],
                "sentences": [
                    "Is it not pleasant to learn with constant practice?",
                    "Is it not delightful to have friends coming from distant quarters?",
                    "Is he not a superior man, who feels no discomposure though men may take no note of him?",
                    "The superior man understands righteousness; the mean man understands profit.",
                    "The man of perfect virtue rests in benevolence, and the wise value benevolence.",
                ],
            },
            {
                "id": "ku_huang_1998",
                "translator": "Ku Hung-ming",
                "year": 1898,
                "mentor": "",
                "influenced_by": ["James Legge"],
                "sentences": [
                    "To learn and at due times repeat what one has learnt, is that not after all a pleasure?",
                    "To have friends come from afar, is that not a joy?",
                    "To remain unvexed when others do not know you, is that not the mark of a gentleman?",
                    "A gentleman understands duty; a small man understands gain.",
                    "The humane man is at home in humanity, and the wise profit by humanity.",
                ],
            },
            {
                "id": "lau_1979",
                "translator": "D. C. Lau",
                "year": 1979,
                "mentor": "",
                "influenced_by": ["James Legge", "Ku Hung-ming"],
                "sentences": [
                    "To learn and to practise what is learned from time to time, is this not a pleasure?",
                    "To have friends come from afar, is this not a joy?",
                    "To remain unsoured even though others do not recognize you, is this not to be a gentleman?",
                    "The gentleman understands what is right; the petty man understands profit.",
                    "The benevolent man finds peace in benevolence; the wise make use of benevolence.",
                ],
            },
        ],
        "core_terms": ["仁", "义"],
    }


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跨语言关联分析最小脚本")
    parser.add_argument("--input", type=Path, help="输入 JSON 文件路径")
    parser.add_argument("--output", type=Path, default=Path("analysis_report.json"), help="输出 JSON 文件路径")
    parser.add_argument("--demo", action="store_true", help="使用内置《论语》示例数据")
    parser.add_argument("--use-bertalign", action="store_true", help="优先尝试调用 Bertalign")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        data = demo_data()
    elif args.input:
        data = load_json(args.input)
    else:
        raise SystemExit("请提供 --demo 或 --input data.json")

    report = build_report(data, use_bertalign=args.use_bertalign)
    save_json(args.output, report)

    print(f"[OK] 分析完成，结果已写入: {args.output}")
    print("[INFO] 句对齐摘要:")
    for version_id, meta in report["alignment_summary"].items():
        print(
            f"  - {version_id}: translator={meta['translator']}, year={meta['year']}, "
            f"method={meta['method']}, pair_count={meta['pair_count']}"
        )


if __name__ == "__main__":
    main()
