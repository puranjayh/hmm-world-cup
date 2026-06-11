"""
update_elo.py — Append current Elo snapshot for all WC 2026 teams to eloratings.csv.

Source: https://www.eloratings.net/{Team_Name}.tsv
TSV format (per row = one match involving the team):
  col3=home_code  col4=away_code
  col9=home_elo_change  col10=home_elo_after  col11=away_elo_after  col12=away_elo_change

Usage:
    cd hmm-world-cup/data/raw
    python update_elo.py
    python data_filter.py
"""
import io, sys, time
import requests
import pandas as pd
from datetime import date

sys.stdout.reconfigure(encoding="utf-8")

# simulator name -> (url_slug, site_code)
TEAMS = {
    "Mexico":       ("Mexico",        "MX"),
    "USA":          ("United_States", "US"),
    "Canada":       ("Canada",        "CA"),
    "Colombia":     ("Colombia",      "CO"),
    "Ecuador":      ("Ecuador",       "EC"),
    "Uruguay":      ("Uruguay",       "UY"),
    "Panama":       ("Panama",        "PA"),
    "Bolivia":      ("Bolivia",       "BO"),
    "Argentina":    ("Argentina",     "AR"),
    "Chile":        ("Chile",         "CL"),
    "Peru":         ("Peru",          "PE"),
    "Venezuela":    ("Venezuela",     "VE"),
    "Brazil":       ("Brazil",        "BR"),
    "Paraguay":     ("Paraguay",      "PY"),
    "Costa Rica":   ("Costa_Rica",    "CR"),
    "Guatemala":    ("Guatemala",     "GT"),
    "France":       ("France",        "FR"),
    "Belgium":      ("Belgium",       "BE"),
    "England":      ("England",       "EN"),
    "Serbia":       ("Serbia",        "RS"),
    "Portugal":     ("Portugal",      "PT"),
    "Spain":        ("Spain",         "ES"),
    "Turkey":       ("Turkey",        "TR"),
    "Ukraine":      ("Ukraine",       "UA"),
    "Germany":      ("Germany",       "DE"),
    "Netherlands":  ("Netherlands",   "NL"),
    "Denmark":      ("Denmark",       "DK"),
    "Cameroon":     ("Cameroon",      "CM"),
    "Italy":        ("Italy",         "IT"),
    "Croatia":      ("Croatia",       "HR"),
    "Switzerland":  ("Switzerland",   "CH"),
    "Albania":      ("Albania",       "SQ"),
    "Morocco":      ("Morocco",       "MA"),
    "Senegal":      ("Senegal",       "SN"),
    "South Africa": ("South_Africa",  "ZA"),
    "Tunisia":      ("Tunisia",       "TN"),
    "Nigeria":      ("Nigeria",       "NG"),
    "Egypt":        ("Egypt",         "EG"),
    "Algeria":      ("Algeria",       "DZ"),
    "Ivory Coast":  ("Ivory_Coast",   "CI"),
    "Japan":        ("Japan",         "JP"),
    "South Korea":  ("South_Korea",   "KR"),
    "Australia":    ("Australia",     "AU"),
    "Saudi Arabia": ("Saudi_Arabia",  "SA"),
    "Iran":         ("Iran",          "IR"),
    "Uzbekistan":   ("Uzbekistan",    "UZ"),
    "Qatar":        ("Qatar",         "QA"),
    "Kyrgyzstan":   ("Kyrgyzstan",    "KG"),
}

TODAY = date.today().strftime("%Y-%m-%d")

def _clean(val: str) -> float:
    import re
    # Strip any non-ASCII minus-sign variants, keep digits and standard minus/dot
    s = re.sub(r"[^\d.\-]", "", str(val).encode("ascii", "ignore").decode())
    return float(s) if s and s != "-" else 0.0


def fetch_current_elo(url_slug: str, code: str, session: requests.Session):
    url = f"https://www.eloratings.net/{url_slug}.tsv"
    try:
        r = session.get(url, timeout=15)
        if r.status_code != 200:
            return None
        df = pd.read_csv(io.StringIO(r.text), sep="\t", header=None, dtype=str)
    except Exception as e:
        print(f"error: {e}")
        return None

    for _, row in df.iloc[::-1].iterrows():
        if str(row[3]) == code:          # team was home
            return float(row[10]), _clean(row[9])
        if str(row[4]) == code:          # team was away
            return float(row[11]), _clean(row[12])
    return None


def main():
    elo_path = "eloratings.csv"
    elo_df = pd.read_csv(elo_path)
    elo_df["team"] = elo_df["team"].str.replace("\xa0", " ", regex=False)

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"

    new_rows = []
    for sim_name, (url_slug, code) in TEAMS.items():
        print(f"  [{sim_name}] ...", end=" ", flush=True)
        result = fetch_current_elo(url_slug, code, session)
        if result is None:
            print("skipped")
            time.sleep(0.2)
            continue
        rating, change = result
        print(f"rating={rating:.0f}  change={change:+.0f}")
        new_rows.append({"date": TODAY, "team": sim_name, "rating": rating, "change": change})
        time.sleep(0.3)

    if new_rows:
        combined = pd.concat([elo_df, pd.DataFrame(new_rows)], ignore_index=True)
        combined.to_csv(elo_path, index=False)
        print(f"\nAppended {len(new_rows)} rows to {elo_path}")
        print("Now run: python data_filter.py")
    else:
        print("No rows added.")


if __name__ == "__main__":
    main()
