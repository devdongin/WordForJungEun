"""N5 단어 수집기.

공개 JLPT 단어 목록(open-anki-jlpt-decks, MIT)에서 아직 저장소에 없는 단어를
목록 순서대로 가져오고, Claude로 한국어 뜻, 품사, 예문을 만들어 words/n5.json에 추가한다.

ANTHROPIC_API_KEY가 없으면 후보 단어만 보여주고 파일은 바꾸지 않는다.

사용법:
    python scripts/collect_words.py --count 20
    python scripts/collect_words.py --count 20 --dry-run
"""

import argparse
import csv
import io
import json
import os
import sys
import urllib.request
from datetime import date
from pathlib import Path

SOURCE_URL = "https://raw.githubusercontent.com/jamsinclair/open-anki-jlpt-decks/main/src/n5.csv"
SOURCE_NAME = "open-anki-jlpt-decks (MIT)"
ROOT = Path(__file__).resolve().parent.parent
WORDS_FILE = ROOT / "words" / "n5.json"
INDEX_FILE = ROOT / "words" / "index.json"
POS_CHOICES = ["명사", "대명사", "동사", "い형용사", "な형용사", "부사", "명사·する동사", "접속사", "감동사", "조사", "접두사", "접미사", "연체사", "수사", "표현"]
MODEL = "claude-opus-5"


def key(kanji: str, reading: str) -> str:
    return f"{kanji or reading}-{reading}"


def summary(text: str) -> None:
    """콘솔과 GitHub Actions 실행 요약에 같이 쓴다."""
    print(text)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text + "\n")


def fetch_candidates(existing: set[str], count: int) -> list[dict]:
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as res:
        text = res.read().decode("utf-8")
    picked, seen = [], set()
    for row in csv.DictReader(io.StringIO(text)):
        expression = row["expression"].strip()
        reading = row["reading"].strip()
        # 여러 표기가 섞인 줄이나 물결표가 붙은 접사는 건너뜀
        if not expression or not reading or any(c in expression + reading for c in ";；、,～~"):
            continue
        kanji = "" if expression == reading else expression
        k = key(kanji, reading)
        if k in existing or k in seen:
            continue
        seen.add(k)
        picked.append({"kanji": kanji, "reading": reading, "english": row["meaning"].strip()})
        if len(picked) == count:
            break
    return picked


SCHEMA = {
    "type": "object",
    "properties": {
        "words": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "meanings": {"type": "array", "items": {"type": "string"}},
                    "pos": {"type": "string", "enum": POS_CHOICES},
                    "example_jp": {"type": "string"},
                    "example_ko": {"type": "string"},
                },
                "required": ["index", "meanings", "pos", "example_jp", "example_ko"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["words"],
    "additionalProperties": False,
}

PROMPT = """한국인이 JLPT N5 단어를 외우는 단어장에 넣을 정보를 만들어 주세요.

각 단어마다:
- meanings: 한국어 뜻 1~3개. 퀴즈 정답으로 쓰이니 짧고 자연스러운 기본형으로 씁니다 (예: "먹다", "학교", "조용하다"). 영어 뜻은 참고만 하고, 일본어 단어의 실제 뜻을 기준으로 합니다.
- pos: 품사. 주어진 목록에서 고릅니다.
- example_jp: 이 단어가 들어간 N5 수준의 짧은 일본어 예문 한 문장.
- example_ko: 그 예문의 자연스러운 한국어 번역.
- index: 입력의 index를 그대로 돌려줍니다.

단어 목록:
{words}"""


def enrich(candidates: list[dict]) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    listing = "\n".join(
        f'{i}. {c["kanji"] or c["reading"]} ({c["reading"]}) - {c["english"]}' for i, c in enumerate(candidates)
    )
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
            messages=[{"role": "user", "content": PROMPT.format(words=listing)}],
        )
    except anthropic.AuthenticationError:
        sys.exit("ANTHROPIC_API_KEY가 올바르지 않습니다.")
    except anthropic.RateLimitError:
        sys.exit("Claude API 사용 한도에 걸렸습니다. 잠시 후 다시 실행하세요.")
    except anthropic.APIStatusError as e:
        sys.exit(f"Claude API 오류 ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Claude API에 연결하지 못했습니다.")

    if response.stop_reason == "refusal":
        sys.exit("Claude가 요청을 처리하지 않았습니다.")
    if response.stop_reason == "max_tokens":
        sys.exit("응답이 잘렸습니다. --count를 줄여서 다시 실행하세요.")
    text = next(b.text for b in response.content if b.type == "text")
    results = {w["index"]: w for w in json.loads(text)["words"]}

    enriched = []
    for i, c in enumerate(candidates):
        r = results.get(i)
        meanings = [m.strip() for m in (r or {}).get("meanings", []) if m.strip()]
        if not r or not meanings:
            print(f"건너뜀: {c['kanji'] or c['reading']} (뜻을 받지 못함)")
            continue
        enriched.append({**c, "meanings": meanings, "pos": r["pos"], "example_jp": r["example_jp"].strip(), "example_ko": r["example_ko"].strip()})
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="후보만 보여주고 파일을 바꾸지 않음")
    args = parser.parse_args()

    words = json.loads(WORDS_FILE.read_text(encoding="utf-8"))
    custom_file = ROOT / "words" / "custom.json"
    custom = json.loads(custom_file.read_text(encoding="utf-8")) if custom_file.exists() else []
    existing = {key(w.get("kanji", ""), w["reading"]) for w in words + custom}

    candidates = fetch_candidates(existing, args.count)
    if not candidates:
        summary("새로 추가할 N5 단어가 없습니다.")
        return

    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    if args.dry_run or not has_key:
        summary(f"## 수집 후보 {len(candidates)}개\n")
        for c in candidates:
            summary(f"- {c['kanji'] or c['reading']} ({c['reading']}): {c['english']}")
        if not has_key and not args.dry_run:
            summary("\nANTHROPIC_API_KEY Secret이 없어 한국어 뜻을 만들지 못했습니다. 단어는 추가하지 않았습니다.")
        return

    enriched = enrich(candidates)
    next_rank = max((w.get("rank", 0) for w in words), default=0) + 1
    added = []
    for offset, e in enumerate(enriched):
        added.append({
            "id": f"jp-{key(e['kanji'], e['reading'])}",
            "kanji": e["kanji"],
            "reading": e["reading"],
            "meanings": e["meanings"],
            "pos": e["pos"],
            "example": {"jp": e["example_jp"], "ko": e["example_ko"]},
            "tags": ["N5"],
            "rank": next_rank + offset,
            "source": SOURCE_NAME,
        })
    if not added:
        summary("추가된 단어가 없습니다.")
        return

    words.extend(added)
    WORDS_FILE.write_text(json.dumps(words, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    for f in index["files"]:
        if f["path"] == "n5.json":
            f["version"] += 1
            f["count"] = len(words)
    index["updatedAt"] = date.today().isoformat()
    INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    summary(f"## N5 단어 {len(added)}개 추가\n")
    for w in added:
        summary(f"- {w['kanji'] or w['reading']} ({w['reading']}): {', '.join(w['meanings'])}")


if __name__ == "__main__":
    main()
