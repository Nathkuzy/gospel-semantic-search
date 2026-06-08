import re
from typing import Dict, Pattern


class DataLoader:
    TARGET_BOOKS = {"Matthew", "Mark", "Luke", "John"}

    def __init__(self, buffer_size: int = 8192, gospels_only: bool = True):
        self._buffer_size: int = buffer_size
        self._gospels_only: bool = gospels_only

        # формат "Book Chapter:Verse text", напр. "Matthew 1:1 The book of..."
        self._regex_pattern: Pattern = re.compile(
            r"^([1-3]?\s?[A-Za-z]+(?:\s[A-Za-z]+)?)\s+(\d+):(\d+)\s+(.+)$"
        )

        # fallback: окремий заголовок + рядки "1:1 text"
        self._verse_only_pattern: Pattern = re.compile(r"^(\d+):(\d+)\s+(.+)$")
        self._book_header_pattern: Pattern = re.compile(
            r"^(The\s+Gospel\s+According\s+to\s+(\w+)|Book\s+of\s+(\w+)|(\w+))\s*$"
        )

        self.parsed_corpus: Dict = {}

    def load_file(self, file_path: str, encoding: str = "utf-8") -> Dict:
        try:
            current_book = None

            with open(file_path, "r", encoding=encoding, buffering=self._buffer_size) as fh:
                for raw_line in fh:
                    line = raw_line.strip()
                    if not line:
                        continue

                    match = self._regex_pattern.match(line)
                    if match:
                        book_name = match.group(1).strip()
                        chapter = int(match.group(2))
                        verse_num = int(match.group(3))
                        text = match.group(4).strip()

                        if self._gospels_only and book_name not in self.TARGET_BOOKS:
                            continue

                        self._insert_verse(book_name, chapter, verse_num, text)
                        current_book = book_name
                        continue

                    header_match = self._book_header_pattern.match(line)
                    if header_match:
                        candidate = (
                            header_match.group(2)
                            or header_match.group(3)
                            or header_match.group(4)
                        )
                        if candidate:
                            current_book = candidate

                    verse_match = self._verse_only_pattern.match(line)
                    if verse_match and current_book is not None:
                        if self._gospels_only and current_book not in self.TARGET_BOOKS:
                            continue
                        chapter = int(verse_match.group(1))
                        verse_num = int(verse_match.group(2))
                        text = verse_match.group(3).strip()
                        self._insert_verse(current_book, chapter, verse_num, text)

            return self.parsed_corpus

        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Файл корпусу не знайдено: {file_path}") from exc
        except PermissionError as exc:
            raise PermissionError(f"Немає прав для читання: {file_path}") from exc
        except UnicodeDecodeError as exc:
            raise UnicodeDecodeError(
                exc.encoding, exc.object, exc.start, exc.end,
                f"Помилка декодування у кодуванні {encoding}",
            )

    def _insert_verse(self, book: str, chapter: int, verse: int, text: str) -> None:
        if book not in self.parsed_corpus:
            self.parsed_corpus[book] = {"name": book, "chapters": {}}

        if chapter not in self.parsed_corpus[book]["chapters"]:
            self.parsed_corpus[book]["chapters"][chapter] = {"number": chapter, "verses": []}

        uid = f"{book}_{chapter}_{verse}"
        self.parsed_corpus[book]["chapters"][chapter]["verses"].append(
            {"book": book, "chapter": chapter, "verse": verse, "uid": uid, "text": text}
        )

    def flatten_verses(self) -> list:
        flat = []
        for book_data in self.parsed_corpus.values():
            for chapter_data in sorted(book_data["chapters"].values(), key=lambda c: c["number"]):
                for verse_data in sorted(chapter_data["verses"], key=lambda v: v["verse"]):
                    flat.append(verse_data)
        return flat
