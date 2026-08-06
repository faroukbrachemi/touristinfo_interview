import json
import re
import sys
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

URL = ("https://www.valgardena.it/en/addresses/detail/base/company/"
       "selva-di-val-gardena/bar-oswald/4206CF9CFA75D1664ECC5822E183F31B/")

DAYS = {"mo": "Monday", "tu": "Tuesday", "we": "Wednesday", "th": "Thursday",
        "fr": "Friday", "sa": "Saturday", "su": "Sunday"}
TIMES = re.compile(r"(\d{1,2})[:.](\d{2})\s*[-–]\s*(\d{1,2})[:.](\d{2})")
DATES = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})\s*[-–]\s*(\d{1,2})\.(\d{1,2})\.(\d{4})")
CAP = re.compile(r"^(\d{5})\s+(.+)$")          # 39048 Selva di Val Gardena - South Tyrol


def in_footer(node):
    """The site footer carries the tourist board's own phone/email — skip it."""
    return any(p.name == "footer" or "footer" in " ".join(p.get("class") or [])
               for p in node.parents if p.name)


def cf_decode(hex_str):
    """Cloudflare hides emails as hex where byte 0 is an XOR key."""
    key = int(hex_str[:2], 16)
    return "".join(chr(int(hex_str[i:i + 2], 16) ^ key)
                   for i in range(2, len(hex_str), 2))


def get_contact(soup):
    contact = {"phone": None, "email": None, "address": None}

    tel = next((a for a in soup.select('a[href^="tel:"]') if not in_footer(a)), None)
    if tel:
        contact["phone"] = re.sub(r"[^\d+]", "", unquote(tel["href"])[4:])

    mail = next((a for a in soup.select('a[href^="mailto:"]') if not in_footer(a)), None)
    if mail:
        contact["email"] = unquote(mail["href"])[7:].split("?")[0]
    else:
        node = next((n for n in soup.select("a[href*='email-protection'], [data-cfemail]")
                     if not in_footer(n)), None)
        if node:
            raw = node.get("data-cfemail") or node["href"].split("#")[-1]
            contact["email"] = unquote(cf_decode(raw)).split("?")[0]

    # walk up from the phone link until the block also holds a postal code
    block = tel
    while block and not contact["address"]:
        lines = [l.strip() for l in block.get_text("\n").split("\n") if l.strip()]
        for i, line in enumerate(lines):
            m = CAP.match(line)
            if m and i:
                city, _, region = m.group(2).partition(" - ")
                contact["address"] = {"street": lines[i - 1],
                                      "postal_code": m.group(1),
                                      "city": city.strip(),
                                      "region": region.strip() or None,
                                      "country": "Italy"}
                break
        block = block.parent

    return contact


def get_hours(soup):
    """Date-range x weekday grid -> [{valid_from, valid_until, days, opens, closes}]."""
    table = next((t for t in soup.find_all("table") if TIMES.search(t.get_text(" "))), None)
    if not table:
        return []

    rows = table.find_all("tr")
    head = [c.get_text(strip=True) for c in rows[0].find_all(["th", "td"])]
    names = [DAYS.get(c[:2].lower(), c) for c in head[1:]] or list(DAYS.values())

    valid = DATES.search(head[0] if head else "")
    span = {"valid_from": f"{valid[3]}-{int(valid[2]):02d}-{int(valid[1]):02d}",
            "valid_until": f"{valid[6]}-{int(valid[5]):02d}-{int(valid[4]):02d}"} \
        if valid else {"valid_from": None, "valid_until": None}

    out = []
    for row in rows[1:]:
        cells = row.find_all(["th", "td"])
        if not cells or not TIMES.search(cells[0].get_text(" ")):
            continue
        # open days are marked with an icon, not text — check classes and glyphs
        flags = [bool(re.search(r"check|open|✓|✔", " ".join(
            [c.get_text(" ", strip=True)] + (c.get("class") or []) +
            [k for ch in c.find_all(True) for k in (ch.get("class") or [])]), re.I))
            for c in cells[1:]]
        days = [d for d, f in zip(names, flags) if f] or names  # no marker -> all days
        for h1, m1, h2, m2 in TIMES.findall(cells[0].get_text(" ")):
            out.append({**span, "days": days,
                        "opens": f"{int(h1):02d}:{m1}", "closes": f"{int(h2):02d}:{m2}"})
    return out


def get_description(soup, name, city, hours):
    """The page lists features, not sentences — turn them into a 2-3 sentence summary."""
    box = soup.find(id=re.compile("description", re.I))
    raw = box.get_text(" ", strip=True) if box else ""
    raw = re.sub(r"^\s*description\s*", "", raw, flags=re.I).strip()
    if not raw:
        meta = soup.find("meta", attrs={"name": "description"})
        raw = meta["content"].strip() if meta and meta.get("content") else ""
    if not raw:
        return None

    feats = [f.strip() for f in re.split(r"[,;]", raw) if 2 < len(f.strip()) < 40][:6]
    text = f"{name} is a bar and café{' in ' + city if city else ''}, in the Dolomites of South Tyrol."
    if feats:
        text += f" It serves {', '.join(feats[:-1])} and {feats[-1]}." if len(feats) > 1 \
            else f" It serves {feats[0]}."
    if hours:
        h = hours[0]
        text += (f" It is open daily from {h['opens']} to {h['closes']}."
                 if len(h["days"]) == 7 else
                 f" Open {', '.join(h['days'])}, {h['opens']}–{h['closes']}.")
    return text


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (tourism-extractor)",
                                      "Accept-Language": "en"}, timeout=20)
    html.raise_for_status()
    soup = BeautifulSoup(html.text, "lxml")

    name = soup.h1.get_text(strip=True) if soup.h1 else None
    contact = get_contact(soup)
    hours = get_hours(soup)
    city = (contact["address"] or {}).get("city")

    print(json.dumps({
        "name": name,
        "opening_hours": hours,
        "contact": contact,
        "description": get_description(soup, name, city, hours),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()