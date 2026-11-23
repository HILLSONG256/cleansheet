"""
CleanSheet auto-updater ("the robot").
Runs daily in the cloud (GitHub Actions). Pure numpy — no sklearn, no pandas,
no big data file. Steps:
  1. fetch upcoming EPL fixtures + bookmaker odds from the-odds-api
  2. compute each team's features from team_state.json
  3. predict H/D/A with model.json (a tiny exported logistic model)
  4. inject fresh predictions into index.html
If the fetch fails or returns nothing, it changes NOTHING (site keeps last-good data).

Local test:  python robot.py --mock sample_odds.json
Live run:    THE_ODDS_API_KEY=xxxx python robot.py
"""
import json, re, os, sys, math, datetime, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SPORT = "soccer_epl"
LEAGUE = "E0"
REGIONS = "uk"          # UK bookmakers cover the EPL well (1 API credit/run)

TEAM_MAP = {
  "Nottingham Forest": "Nott'm Forest", "Manchester United": "Man United",
  "Manchester City": "Man City", "Leeds United": "Leeds", "Coventry City": "Coventry",
  "Ipswich Town": "Ipswich", "Hull City": "Hull", "Brighton and Hove Albion": "Brighton",
  "Newcastle United": "Newcastle", "Tottenham Hotspur": "Tottenham",
  "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
  "Sheffield United": "Sheffield United",
}
def mn(x): return TEAM_MAP.get(x, x)
def load(f): return json.load(open(os.path.join(HERE, f)))


def fetch_events():
    """Return the list of fixture events, or [] if unavailable."""
    if "--mock" in sys.argv:
        path = sys.argv[sys.argv.index("--mock") + 1]
        print(f"[mock] reading fixtures from {path}")
        return json.load(open(path))
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not key:
        print("[skip] THE_ODDS_API_KEY not set — leaving the site unchanged.")
        return []
    url = (f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
           f"?apiKey={key}&regions={REGIONS}&markets=h2h&oddsFormat=decimal")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            remaining = r.headers.get("x-requests-remaining")
            data = json.load(r)
        print(f"[ok] fetched {len(data)} events  (API credits left: {remaining})")
        return data
    except Exception as e:
        print(f"[skip] fetch failed ({e}) — leaving the site unchanged.")
        return []


def parse_fixtures(events):
    """events -> [{date, home, away, oddH, oddD, oddA}] using consensus (avg) odds."""
    out = []
    for ev in events:
        if ev.get("sport_key") != SPORT:
            continue
        hn, an = ev["home_team"], ev["away_team"]
        acc = {hn: [], an: [], "Draw": []}
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk["key"] != "h2h":
                    continue
                for o in mk["outcomes"]:
                    if o["name"] in acc:
                        acc[o["name"]].append(o["price"])
        if not (acc[hn] and acc[an] and acc["Draw"]):
            continue
        avg = lambda L: sum(L) / len(L)
        out.append({"date": ev["commence_time"][:10], "home": mn(hn), "away": mn(an),
                    "oddH": avg(acc[hn]), "oddD": avg(acc["Draw"]), "oddA": avg(acc[an])})
    return out


def softmax(v):
    m = max(v); e = [math.exp(x - m) for x in v]; s = sum(e)
    return [x / s for x in e]


def predict(fixtures, model, state):
    feats, K = model["feats"], model["K"]
    mean, scale = model["mean"], model["scale"]
    coef, inter, classes = model["coef"], model["intercept"], model["classes"]
    rows = []
    for fx in fixtures:
        H, A = fx["home"], fx["away"]
        # market implied probabilities (de-margined)
        ip = [1 / fx["oddH"], 1 / fx["oddD"], 1 / fx["oddA"]]
        s = sum(ip); mkt = [round(x / s, 3) for x in ip]
        row = {"date": fmt_date(fx["date"]), "home": H, "away": A, "market": mkt}
        sh, sa = state.get(H), state.get(A)
        if sh and sa and sh.get("elo") is not None and sa.get("elo") is not None \
           and sh.get("form") is not None and sa.get("form") is not None:
            rh, ra = sh["roll"], sa["roll"]      # [sf, sa, gf, ga]
            vals = {"elo_diff": sh["elo"] - sa["elo"], "form_diff": sh["form"] - sa["form"],
                    "sot_att": rh[0] - ra[1], "sot_def": ra[0] - rh[1],
                    "gf_att": rh[2] - ra[3], "gf_def": ra[2] - rh[3],
                    "h_sf": rh[0], "h_sa": rh[1], "a_sf": ra[0], "a_sa": ra[1]}
            z = [(vals[f] - mean[i]) / scale[i] for i, f in enumerate(feats)]
            logits = [sum(z[j] * coef[c][j] for j in range(len(feats))) + inter[c]
                      for c in range(len(classes))]
            p = dict(zip(classes, softmax(logits)))
            row["model"] = [round(p.get("H", 0), 3), round(p.get("D", 0), 3), round(p.get("A", 0), 3)]
        else:
            row["model"] = mkt; row["nomodel"] = True   # unknown team -> show market only
        rows.append(row)
    return rows


def fmt_date(iso):
    try:
        return datetime.datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b")
    except Exception:
        return iso


def main():
    events = fetch_events()
    fixtures = parse_fixtures(events)
    if not fixtures:
        print("No fixtures to write. Done (no changes).")
        return
    model, state = load("model.json"), load("team_state.json")
    upcoming = predict(fixtures, model, state)

    data = load("base_data.json")
    data["leagues"][LEAGUE]["upcoming"] = upcoming
    data["generated"] = datetime.date.today().strftime("%d %b %Y")

    html = open(os.path.join(HERE, "index.html")).read()
    html = re.sub(r"const DATA = .*?;\nconst RES",
                  "const DATA = " + json.dumps(data) + ";\nconst RES", html, count=1, flags=re.S)
    open(os.path.join(HERE, "index.html"), "w").write(html)

    modeled = sum(1 for u in upcoming if not u.get("nomodel"))
    print(f"[done] wrote {len(upcoming)} fixtures ({modeled} with a model prediction):")
    for u in upcoming:
        tag = "  (market-only)" if u.get("nomodel") else ""
        print(f"   {u['date']}  {u['home']:16} v {u['away']:16}  model {u['model']}  mkt {u['market']}{tag}")


if __name__ == "__main__":
    main()
