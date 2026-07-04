#!/usr/bin/env python3
"""
Generate SASS-Bench synthetic examples.

The dataset is fictional and programmatic to reduce contamination and avoid dependence
on external factual knowledge.
"""
import argparse
import json
import random
from pathlib import Path

NAMES = ["Lina", "Mira", "Jonas", "Nora", "Eli", "Iris", "Theo", "Rina", "Omar", "Talia",
         "Samir", "Anika", "Leo", "Maya", "Noah", "Ava", "Kian", "Sofia", "Ben", "Zara",
         "Niko", "Elena", "Arun", "Juno", "Vera", "Milo"]
OTHER_NAMES = ["Theo", "Rafa", "Bea", "Max", "Yuna", "Clara", "Idris", "Pavel", "Luca", "Selin"]
EVENTS = ["robotics workshop", "astronomy workshop", "mural design workshop", "coding circle",
          "chess seminar", "ecology meetup", "film club", "design lab", "music camp",
          "science fair", "storytelling session", "math circle", "drama rehearsal",
          "history project", "community garden day", "language club", "debate practice"]
CITIES = ["Boston", "Denver", "Lisbon", "Prague", "Madrid", "Oslo", "Berlin", "Rome",
          "Paris", "Vienna", "Dublin", "Zurich", "Warsaw", "Helsinki", "Tallinn",
          "Ghent", "Valencia", "Bremen"]
DOMAINS = ["school_event", "library_program", "community_workshop", "club_notice", "science_fair"]
FIELDS = ["target_name", "target_age", "event_name", "city", "event_date"]

def date_for(i):
    month = (i % 12) + 1
    day = (i * 7 % 27) + 1
    return f"2026-{month:02d}-{day:02d}"

def choose(rng, arr):
    return rng.choice(arr)

def base_values(rng, i):
    name = choose(rng, NAMES)
    age = rng.randint(11, 18)
    event = choose(rng, EVENTS)
    city = choose(rng, CITIES)
    date = date_for(i)
    return name, age, event, city, date

def supported_item(rng, i):
    name, age, event, city, date = base_values(rng, i)
    context = f"On {date}, {name}, age {age}, joined the {event} in {city}."
    gold = {"target_name": name, "target_age": age, "event_name": event, "city": city, "event_date": date}
    status = {f: "SUPPORTED" for f in FIELDS}
    return context, gold, status, {}

def missing_item(rng, i):
    name, age, event, city, date = base_values(rng, i)
    missing_field = rng.choice(["target_age", "event_name", "city", "event_date"])
    gold = {"target_name": name, "target_age": age, "event_name": event, "city": city, "event_date": date}
    status = {f: "SUPPORTED" for f in FIELDS}
    status[missing_field] = "MISSING"
    gold[missing_field] = None

    if missing_field == "target_age":
        context = f"On {date}, {name} joined the {event} in {city}. The notice did not state {name}'s age."
    elif missing_field == "event_name":
        context = f"On {date}, {name}, age {age}, attended an activity in {city}. The report did not state the event name."
    elif missing_field == "city":
        context = f"On {date}, {name}, age {age}, joined the {event}. The notice did not state the city."
    else:
        context = f"{name}, age {age}, joined the {event} in {city}. The notice did not state the event date."
    return context, gold, status, {}

def ambiguous_item(rng, i):
    name, age, event, city, date = base_values(rng, i)
    amb_field = rng.choice(["target_age", "event_name", "city", "event_date"])
    gold = {"target_name": name, "target_age": age, "event_name": event, "city": city, "event_date": date}
    status = {f: "SUPPORTED" for f in FIELDS}
    status[amb_field] = "AMBIGUOUS"
    gold[amb_field] = None

    if amb_field == "target_age":
        context = f"{name}, described only as a teenager, attended the {event} in {city} on {date}."
    elif amb_field == "event_name":
        context = f"On {date}, {name}, age {age}, attended a general after-school activity in {city}, but the activity title was not specified."
    elif amb_field == "city":
        context = f"On {date}, {name}, age {age}, joined the {event} in a nearby city, but the city name was not specified."
    else:
        context = f"{name}, age {age}, attended the {event} in {city} sometime in early 2026, but the exact date was not specified."
    return context, gold, status, {}

def contradictory_item(rng, i):
    name, age, event, city, date = base_values(rng, i)
    field = rng.choice(["target_age", "event_name", "city", "event_date"])
    gold = {"target_name": name, "target_age": age, "event_name": event, "city": city, "event_date": date}
    status = {f: "SUPPORTED" for f in FIELDS}
    status[field] = "CONTRADICTORY"
    gold[field] = None

    if field == "target_age":
        age2 = age + 2 if age <= 16 else age - 2
        context = f"The first event note said {name} was {age} during the {event} in {city} on {date}. A later correction listed {name} as {age2}. No verified age was provided."
    elif field == "event_name":
        event2 = choose(rng, [e for e in EVENTS if e != event])
        context = f"One notice said {name}, age {age}, attended the {event} in {city} on {date}. A correction said the event was the {event2}. No verified event name was provided."
    elif field == "city":
        city2 = choose(rng, [c for c in CITIES if c != city])
        context = f"The first report placed {name}, age {age}, at the {event} in {city} on {date}. A later correction listed the city as {city2}. No verified city was provided."
    else:
        # Make a conflicting date.
        date2 = date_for(i + 17)
        context = f"The first report said {name}, age {age}, attended the {event} in {city} on {date}. A later correction listed the date as {date2}. No verified date was provided."
    return context, gold, status, {}

def distractor_item(rng, i):
    name, age, event, city, date = base_values(rng, i)
    other = choose(rng, [n for n in OTHER_NAMES if n != name])
    field = rng.choice(["target_age", "event_name", "city", "event_date"])
    gold = {"target_name": name, "target_age": age, "event_name": event, "city": city, "event_date": date}
    status = {f: "SUPPORTED" for f in FIELDS}
    status[field] = "DISTRACTOR"
    distractors = {}

    if field == "target_age":
        d_age = age + 3 if age <= 15 else age - 3
        context = f"{name}, age {age}, attended the {event} in {city} on {date}. The organizer, {other}, age {d_age}, introduced the speakers."
        distractors["target_age"] = d_age
    elif field == "event_name":
        d_event = choose(rng, [e for e in EVENTS if e != event])
        context = f"{name}, age {age}, attended the {event} in {city} on {date}. {other} later announced a separate {d_event}."
        distractors["event_name"] = d_event
    elif field == "city":
        d_city = choose(rng, [c for c in CITIES if c != city])
        context = f"{name}, age {age}, attended the {event} in {city} on {date}. {other} hosted a different club meeting in {d_city}."
        distractors["city"] = d_city
    else:
        d_date = date_for(i + 23)
        context = f"{name}, age {age}, attended the {event} in {city} on {date}. {other} scheduled a separate follow-up on {d_date}."
        distractors["event_date"] = d_date
    return context, gold, status, distractors

BUILDERS = {
    "SUPPORTED": supported_item,
    "MISSING": missing_item,
    "AMBIGUOUS": ambiguous_item,
    "CONTRADICTORY": contradictory_item,
    "DISTRACTOR": distractor_item
}

def generate(n, seed):
    rng = random.Random(seed)
    challenge_cycle = ["SUPPORTED", "MISSING", "AMBIGUOUS", "CONTRADICTORY", "DISTRACTOR"]
    rows = []
    for i in range(n):
        ch = challenge_cycle[i % len(challenge_cycle)]
        context, gold, status, distractors = BUILDERS[ch](rng, i)
        target_entity = gold["target_name"]
        row = {
            "id": f"SASS_{i+1:06d}",
            "domain": rng.choice(DOMAINS),
            "challenge_type": ch,
            "target_entity": target_entity,
            "context": context,
            "schema_name": "person_event_v1",
            "gold": gold,
            "field_status": status,
            "evidence_spans": {k: (None if gold[k] is None else str(gold[k])) for k in FIELDS},
            "distractors": distractors
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/sass_bench.jsonl")
    args = ap.parse_args()

    rows = generate(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {out}")

if __name__ == "__main__":
    main()
