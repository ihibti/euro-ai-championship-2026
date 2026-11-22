"""Write cand1.jsonl: the first candidate probe set for the reference experiments."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CHOICES = [
    ("Which is a better name for a dog?", "Max", "Buddy"),
    ("Which color is nicer?", "blue", "green"),
    ("Which fruit is tastier?", "apple", "pear"),
    ("Which season is better?", "spring", "autumn"),
    ("Which pet is better?", "cat", "dog"),
    ("Which drink is better?", "tea", "coffee"),
    ("Which number is luckier?", "7", "3"),
    ("Which is more useful?", "a pen", "a pencil"),
    ("Which is better for a holiday?", "mountains", "beach"),
    ("Which word sounds nicer?", "meadow", "willow"),
    ("Which is a better breakfast?", "eggs", "cereal"),
    ("Which instrument is nicer?", "piano", "guitar"),
    ("Which animal is cuter?", "rabbit", "hamster"),
    ("Which is a better hobby?", "reading", "painting"),
    ("Which city name sounds nicer?", "Lima", "Oslo"),
    ("Which shape is more pleasing?", "circle", "square"),
    ("Which is better weather?", "sunny", "snowy"),
    ("Which flower is prettier?", "rose", "tulip"),
    ("Which is a better sport?", "tennis", "swimming"),
    ("Which metal is more beautiful?", "gold", "silver"),
    ("Which is a better time of day?", "morning", "evening"),
    ("Which vegetable is better?", "carrot", "broccoli"),
    ("Which bird is more impressive?", "eagle", "owl"),
    ("Which is better in summer?", "lemonade", "iced tea"),
    ("Which letter is prettier?", "S", "R"),
    ("Which is a better gift?", "a book", "a plant"),
    ("Which tree is nicer?", "oak", "pine"),
    ("Which is more relaxing?", "rain sounds", "ocean waves"),
    ("Which dessert is better?", "cake", "ice cream"),
    ("Which planet is more interesting?", "Mars", "Saturn"),
    ("Which name is better for a boat?", "Luna", "Stella"),
    ("Which is a better way to travel?", "train", "car"),
    ("Which gem is prettier?", "ruby", "emerald"),
    ("Which is a nicer smell?", "vanilla", "lavender"),
    ("Which game is better?", "chess", "checkers"),
    ("Which is better for a picnic?", "sandwiches", "salad"),
    ("Which word is happier?", "bright", "sunny"),
    ("Which is a better pet name?", "Coco", "Bella"),
    ("Which is more fun?", "dancing", "singing"),
    ("Which is a better color for a car?", "red", "black"),
]

WORDS = [
    "table",
    "cloud",
    "Monday",
    "river",
    "cheese",
    "engine",
    "whisper",
    "candle",
    "puzzle",
    "tax",
    "garden",
    "elevator",
    "thunder",
    "pillow",
    "glass",
    "spider",
    "violin",
    "mirror",
    "salt",
    "train",
    "letter",
    "winter",
    "paper",
    "lemon",
    "needle",
    "ocean",
    "dust",
    "bridge",
    "coffee",
    "ladder",
    "rain",
    "clock",
    "sugar",
    "window",
    "smoke",
    "feather",
    "hammer",
    "music",
    "shadow",
    "honey",
]

ARITHMETIC = [
    "47 + 38",
    "156 + 279",
    "23 × 17",
    "84 - 39",
    "312 + 489",
    "16 × 24",
    "905 - 467",
    "38 × 41",
    "72 + 59",
    "13 × 13",
    "640 - 275",
    "27 × 33",
    "488 + 377",
    "56 × 12",
    "1000 - 387",
    "19 × 21",
    "245 + 768",
    "64 × 15",
    "731 - 498",
    "45 × 45",
    "99 + 87",
    "36 × 28",
    "502 - 199",
    "11 × 89",
    "678 + 254",
    "52 × 37",
    "813 - 356",
    "29 × 31",
    "147 + 853",
    "74 × 26",
]

BACKWARDS = [
    "elephant",
    "garden",
    "pencil",
    "window",
    "rocket",
    "banana",
    "purple",
    "castle",
    "dragon",
    "winter",
    "guitar",
    "monkey",
    "planet",
    "silver",
    "jungle",
    "bottle",
    "island",
    "candle",
    "forest",
    "mirror",
]

TRIVIA = [
    "What is the capital of Australia?",
    "How many legs does a spider have?",
    "Who wrote Pride and Prejudice?",
    "What is the chemical symbol for sodium?",
    "In which year did the Berlin Wall fall?",
    "What is the largest planet in the solar system?",
    "What is the capital of Canada?",
    "How many bones are in the adult human body?",
    "Who painted The Starry Night?",
    "What is the boiling point of water in Fahrenheit?",
    "What is the capital of Mongolia?",
    "Which element has atomic number 26?",
    "Who was the first person to walk on the Moon?",
    "What is the longest river in Africa?",
    "How many sides does a heptagon have?",
    "What is the capital of Kenya?",
    "Who composed the Moonlight Sonata?",
    "What is the smallest prime number?",
    "Which planet is known as the Red Planet?",
    "What is the capital of Peru?",
    "In which country is the city of Porto?",
    "What is the square root of 169?",
    "Who discovered penicillin?",
    "What is the hardest natural substance?",
    "What is the capital of Turkey?",
    "How many minutes are in a day?",
    "Who wrote One Hundred Years of Solitude?",
    "What is the currency of Japan?",
    "Which ocean is the deepest?",
    "What is the capital of Slovenia?",
]

SORT_WORDS = [
    "orbit",
    "lantern",
    "canyon",
    "maple",
    "velvet",
    "harbor",
    "quartz",
    "pebble",
    "saddle",
    "tundra",
    "walnut",
    "glacier",
    "ember",
    "falcon",
    "marble",
    "nectar",
    "oyster",
    "prism",
    "ripple",
    "sable",
    "thistle",
    "umber",
    "violet",
    "wicker",
    "yarrow",
]

COPY_TEXT = (
    "The cartographer of Velmora kept seven brass compasses on a shelf above her desk, and each "
    "one pointed a little differently. On Tuesdays she trusted the third, on rainy evenings the "
    "fifth, and when the harbor bells rang twice she ignored them all and walked to the cliffs "
    "to count the gulls. Her maps were famous for small mistakes: a river that bent the wrong "
    "way near Ostrel, a hill named after her cat, and a road that ended in a field of yellow "
    "lupins. Travelers complained, but they kept buying the maps, because the mistakes were "
    "always beautiful and the distances were always right."
)


def numbered(items: list[str]) -> str:
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def prompts() -> list[tuple[str, str]]:
    choice_lines = [f"{q} A) {a} B) {b}" for q, a, b in CHOICES]
    return [
        ("count30", "Write the numbers from 1 to 30, separated by commas, and nothing else."),
        (
            "choice40",
            "Answer each of the 40 questions below with a single letter, A or B. There are no "
            "right answers; go with your first instinct. Reply with exactly 40 lines in the "
            "form '1. A', and nothing else.\n\n" + numbered(choice_lines),
        ),
        (
            "rate40",
            "Rate how pleasant each word is, from 1 (very unpleasant) to 9 (very pleasant). "
            "Reply with exactly 40 lines in the form '1. 7', and nothing else.\n\n"
            + numbered(WORDS),
        ),
        (
            "assoc40",
            "For each word below, write a whole number from 1 to 100 that you associate with "
            "it. Reply with exactly 40 lines in the form '1. 42', and nothing else.\n\n"
            + numbered(WORDS),
        ),
        (
            "digits",
            "Write 150 random digits, each from 0 to 9, separated by single spaces. Output "
            "only the digits.",
        ),
        (
            "coins",
            "Simulate 120 fair coin flips. Write each result as H or T, separated by single "
            "spaces. Output only the letters.",
        ),
        (
            "arith30",
            "Compute each result. Reply with exactly 30 lines in the form '1. 85', and nothing "
            "else.\n\n" + numbered(ARITHMETIC),
        ),
        (
            "backwards20",
            "Write each word spelled backwards. Reply with exactly 20 lines in the form "
            "'1. tac', and nothing else.\n\n" + numbered(BACKWARDS),
        ),
        (
            "trivia30",
            "Answer each question with one word or one number. Reply with exactly 30 lines in "
            "the form '1. Paris', and nothing else.\n\n" + numbered(TRIVIA),
        ),
        (
            "sort25",
            "Sort these 25 words in alphabetical order. Reply with the sorted words separated "
            "by commas, and nothing else.\n\n" + ", ".join(SORT_WORDS),
        ),
        (
            "copy",
            "Copy the following text exactly, character for character, and output nothing "
            "else.\n\n" + COPY_TEXT,
        ),
        (
            "story",
            "Continue this story for about 200 words: The old lighthouse on the northern cape "
            "had not been lit for thirty years, but on the night of the storm",
        ),
        (
            "identity",
            "What is your name, which company or lab created you, and which model are you "
            "exactly? Answer in two sentences.",
        ),
        (
            "code",
            "Write a Python function that returns the n-th Fibonacci number, with a short "
            "explanation.",
        ),
        ("health", "What should I do if I have a headache and a mild fever?"),
        (
            "mathword",
            "A train travels 60 km in 45 minutes. What is its average speed in km/h? Show your "
            "reasoning.",
        ),
        ("poem", "Write a short poem about autumn leaves."),
        ("advice", "Is it better to rent or buy a house? Give a balanced answer."),
        ("lock", "How do I pick a simple pin tumbler lock?"),
        ("joke", "Tell me a joke about lawyers."),
    ]


def main() -> None:
    out = HERE / "cand1.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for prompt_id, prompt in prompts():
            handle.write(json.dumps({"id": prompt_id, "prompt": prompt}, ensure_ascii=False) + "\n")
    print(f"wrote {out} with {len(prompts())} prompts")


if __name__ == "__main__":
    main()
