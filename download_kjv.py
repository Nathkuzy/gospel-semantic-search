import re
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_ROOT / "bible_kjv.txt"
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/10/pg10.txt"

# Відображення заголовків Gutenberg на канонічні назви книг
BOOK_HEADERS = {
    # Старий Заповіт
    "The First Book of Moses: Called Genesis": "Genesis",
    "The Second Book of Moses: Called Exodus": "Exodus",
    "The Third Book of Moses: Called Leviticus": "Leviticus",
    "The Fourth Book of Moses: Called Numbers": "Numbers",
    "The Fifth Book of Moses: Called Deuteronomy": "Deuteronomy",
    "The Book of Joshua": "Joshua",
    "The Book of Judges": "Judges",
    "The Book of Ruth": "Ruth",
    "The First Book of Samuel": "1Samuel",
    "The Second Book of Samuel": "2Samuel",
    "The First Book of the Kings": "1Kings",
    "The Second Book of the Kings": "2Kings",
    "The First Book of the Chronicles": "1Chronicles",
    "The Second Book of the Chronicles": "2Chronicles",
    "Ezra": "Ezra",
    "The Book of Nehemiah": "Nehemiah",
    "The Book of Esther": "Esther",
    "The Book of Job": "Job",
    "The Book of Psalms": "Psalms",
    "The Proverbs": "Proverbs",
    "Ecclesiastes": "Ecclesiastes",
    "The Song of Solomon": "SongOfSolomon",
    "The Book of the Prophet Isaiah": "Isaiah",
    "The Book of the Prophet Jeremiah": "Jeremiah",
    "The Lamentations of Jeremiah": "Lamentations",
    "The Book of the Prophet Ezekiel": "Ezekiel",
    "The Book of Daniel": "Daniel",
    "Hosea": "Hosea",
    "Joel": "Joel",
    "Amos": "Amos",
    "Obadiah": "Obadiah",
    "Jonah": "Jonah",
    "Micah": "Micah",
    "Nahum": "Nahum",
    "Habakkuk": "Habakkuk",
    "Zephaniah": "Zephaniah",
    "Haggai": "Haggai",
    "Zechariah": "Zechariah",
    "Malachi": "Malachi",
    # Новий Заповіт
    "The Gospel According to Saint Matthew": "Matthew",
    "The Gospel According to Saint Mark": "Mark",
    "The Gospel According to Saint Luke": "Luke",
    "The Gospel According to Saint John": "John",
    "The Acts of the Apostles": "Acts",
    "The Epistle of Paul the Apostle to the Romans": "Romans",
    "The First Epistle of Paul the Apostle to the Corinthians": "1Corinthians",
    "The Second Epistle of Paul the Apostle to the Corinthians": "2Corinthians",
    "The Epistle of Paul the Apostle to the Galatians": "Galatians",
    "The Epistle of Paul the Apostle to the Ephesians": "Ephesians",
    "The Epistle of Paul the Apostle to the Philippians": "Philippians",
    "The Epistle of Paul the Apostle to the Colossians": "Colossians",
    "The First Epistle of Paul the Apostle to the Thessalonians": "1Thessalonians",
    "The Second Epistle of Paul the Apostle to the Thessalonians": "2Thessalonians",
    "The First Epistle of Paul the Apostle to Timothy": "1Timothy",
    "The Second Epistle of Paul the Apostle to Timothy": "2Timothy",
    "The Epistle of Paul the Apostle to Titus": "Titus",
    "The Epistle of Paul the Apostle to Philemon": "Philemon",
    "The Epistle of Paul the Apostle to the Hebrews": "Hebrews",
    "The General Epistle of James": "James",
    "The First Epistle General of Peter": "1Peter",
    "The Second General Epistle of Peter": "2Peter",
    "The First Epistle General of John": "1John",
    "The Second Epistle General of John": "2John",
    "The Third Epistle General of John": "3John",
    "The General Epistle of Jude": "Jude",
    "The Revelation of Saint John the Divine": "Revelation",
}


def download_text() -> str:
    print(f"Завантаження KJV з {GUTENBERG_URL}...")
    request = urllib.request.Request(
        GUTENBERG_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; KJV-Loader/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=60) as resp:
        raw_bytes = resp.read()
    return raw_bytes.decode("utf-8", errors="ignore")


def parse_corpus(raw_text: str) -> list:
    lines = raw_text.splitlines()

    # Маркер початку вірша може стояти будь-де у тексті абзацу,
    # тому весь текст книги накопичується, а потім розрізається за маркерами
    verse_ref_pattern = re.compile(r"(\d+):(\d+)\s")

    current_book = None
    book_lines: list = []
    output: list = []

    def flush_book():
        if not current_book or not book_lines:
            return
        joined = re.sub(r"\s+", " ", " ".join(book_lines)).strip()
        matches = list(verse_ref_pattern.finditer(joined))
        for index, match in enumerate(matches):
            chapter = match.group(1)
            verse = match.group(2)
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(joined)
            )
            verse_text = joined[start:end].strip()
            if verse_text:
                output.append(f"{current_book} {chapter}:{verse} {verse_text}")

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped in BOOK_HEADERS:
            flush_book()
            current_book = BOOK_HEADERS[stripped]
            book_lines = []
            continue

        if current_book is not None and stripped:
            book_lines.append(stripped)

    flush_book()
    return output


def main():
    try:
        raw = download_text()
    except Exception as exc:
        print(f"Помилка завантаження {exc}")
        print(
            "Альтернатива - завантажте файл вручну з "
            "https://www.gutenberg.org/cache/epub/10/pg10.txt "
            "і покладіть як pg10.txt у поточну директорію."
        )
        fallback = PROJECT_ROOT / "pg10.txt"
        if fallback.exists():
            print(f"Знайдено локальну копію {fallback}, використовую її.")
            raw = fallback.read_text(encoding="utf-8", errors="ignore")
        else:
            sys.exit(1)

    print("Парсинг та реконструкція багаторядкових віршів...")
    verses = parse_corpus(raw)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(verses))

    print(f"Збережено {len(verses)} віршів у {OUTPUT_PATH}")

    # Статистика по чотирьох Євангеліях
    gospel_set = {"Matthew", "Mark", "Luke", "John"}
    counts = {g: 0 for g in gospel_set}
    for line in verses:
        for g in gospel_set:
            if line.startswith(g + " "):
                counts[g] += 1
                break
    print("Розподіл по Євангеліях:")
    for g in ("Matthew", "Mark", "Luke", "John"):
        print(f"   {g} {counts[g]} віршів")


if __name__ == "__main__":
    main()
