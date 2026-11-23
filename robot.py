"""
CleanSheet auto-updater ("the robot") - multi-league.
Pure standard library. Fetches fixtures+odds for many leagues from the-odds-api,
predicts each with the right model, and rewrites index.html.
If a league fetch fails it's skipped (others still update). If nothing updates at
all, the site is left unchanged.

Local test:  python robot.py --mock soccer_epl=/path/to/epl.json
Live run:    THE_ODDS_API_KEY=xxxx python robot.py
"""
import json, re, os, sys, math, datetime, unicodedata, difflib, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

# key, site-code, display name, model, API region, candidate divisions ([] = all European)
EU = ['E0','E1','E2','SP1','SP2','I1','I2','D1','D2','F1','F2','P1','B1','N1','SC0','SC1','T1','G1']
LEAGUES = [
  ("soccer_epl","E0","Premier League","eu","uk",['E0','E1']),
  ("soccer_spain_la_liga","SP1","La Liga","eu","uk",['SP1','SP2']),
  ("soccer_italy_serie_a","I1","Serie A","eu","uk",['I1','I2']),
  ("soccer_germany_bundesliga","D1","Bundesliga","eu","uk",['D1','D2']),
  ("soccer_france_ligue_one","F1","Ligue 1","eu","uk",['F1','F2']),
  ("soccer_england_efl_champ","E1","Championship","eu","uk",['E1','E0','E2']),
  ("soccer_portugal_primeira_liga","P1","Primeira Liga","eu","uk",['P1']),
  ("soccer_netherlands_eredivisie","N1","Eredivisie","eu","uk",['N1']),
  ("soccer_belgium_first_div","B1","Belgian Pro League","eu","uk",['B1']),
  ("soccer_spl","SC0","Scottish Premiership","eu","uk",['SC0','SC1']),
  ("soccer_turkey_super_league","T1","Super Lig","eu","uk",['T1']),
  ("soccer_greece_super_league","G1","Super League Greece","eu","uk",['G1']),
  ("soccer_uefa_champs_league","CL","Champions League","eu","uk",EU),
  ("soccer_uefa_europa_league","EL","Europa League","eu","uk",EU),
  ("soccer_uefa_europa_conference_league","ECL","Conference League","eu","uk",EU),
  ("soccer_usa_mls","USA","MLS","basic","us",['USA']),
  ("soccer_brazil_campeonato","BRA","Brazil Serie A","basic","us",['BRA']),
  ("soccer_argentina_primera_division","ARG","Argentina Primera","basic","us",['ARG']),
  ("soccer_japan_j_league","JAP","J1 League","basic","eu",['JAP']),
  ("soccer_mexico_ligamx","MEX","Liga MX","basic","us",['MEX']),
]

OVERRIDE = {
  "Nottingham Forest":"Nott'm Forest","Manchester United":"Man United","Manchester City":"Man City",
  "Leeds United":"Leeds","Coventry City":"Coventry","Ipswich Town":"Ipswich","Hull City":"Hull",
  "Brighton and Hove Albion":"Brighton","Newcastle United":"Newcastle","Tottenham Hotspur":"Tottenham",
  "West Ham United":"West Ham","Wolverhampton Wanderers":"Wolves","Sheffield United":"Sheffield United",
  "Paris Saint Germain":"Paris SG","Borussia Dortmund":"Dortmund","Bayer Leverkusen":"Leverkusen",
  "Borussia Monchengladbach":"M'gladbach","Atletico Madrid":"Ath Madrid","Athletic Bilbao":"Ath Bilbao",
  "Real Sociedad":"Sociedad","Real Betis":"Betis","Inter Milan":"Inter","AC Milan":"Milan",
  "Rayo Vallecano":"Vallecano","Celta Vigo":"Celta","Deportivo La Coruna":"La Coruna",
  "LA Galaxy":"Los Angeles Galaxy","Atlanta United FC":"Atlanta Utd",
  "Atletico Mineiro":"Atletico-MG","Athletico Paranaense":"Athletico-PR","America Mineiro":"America MG",
  "Botafogo":"Botafogo RJ","Vasco da Gama":"Vasco","Guadalajara":"Guadalajara Chivas",
}
# generic words that shouldn't drive a match on their own
WEAK = {'united','city','town','real','deportivo','sporting','atletico','club','junior','juniors','fc'}

def load(f): return json.load(open(os.path.join(HERE, f)))
def norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii','ignore').decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    drop = {'fc','cf','sc','ac','afc','cd','ud','sd','club','de','do','da','del','calcio','ssc',
            'as','us','if','bk','fk','sv','vfl','vfb','tsg','rc','rcd','cp','sk','ca','aa','ec','fk'}
    s = " ".join(w for w in s.split() if w not in drop)
    return re.sub(r"\s+", " ", s).strip()

def softmax(v):
    m = max(v); e = [math.exp(x-m) for x in v]; s = sum(e); return [x/s for x in e]

def predict(model, vals):
    feats = model['feats']
    z = [(vals[f]-model['mean'][i])/model['scale'][i] for i,f in enumerate(feats)]
    logits = [sum(z[j]*model['coef'][c][j] for j in range(len(feats))) + model['intercept'][c]
              for c in range(len(model['classes']))]
    p = dict(zip(model['classes'], softmax(logits)))
    return [round(p.get('H',0),3), round(p.get('D',0),3), round(p.get('A',0),3)]

def feats_for(kind, sh, sa):
    rh, ra = sh['roll'], sa['roll']            # [sf, sa, gf, ga]
    if kind == 'eu':
        if None in (sh['elo'], sa['elo'], sh['form'], sa['form'], rh[0], rh[1], ra[0], ra[1]):
            return None
        return {'elo_diff':sh['elo']-sa['elo'],'form_diff':sh['form']-sa['form'],
                'sot_att':rh[0]-ra[1],'sot_def':ra[0]-rh[1],'gf_att':rh[2]-ra[3],'gf_def':ra[2]-rh[3],
                'h_sf':rh[0],'h_sa':rh[1],'a_sf':ra[0],'a_sa':ra[1]}
    if None in (sh['form'], sa['form'], rh[2], rh[3], ra[2], ra[3]):
        return None
    return {'form_diff':sh['form']-sa['form'],'gf_att':rh[2]-ra[3],'gf_def':ra[2]-rh[3],
            'h_gf':rh[2],'h_ga':rh[3],'a_gf':ra[2],'a_ga':ra[3]}

class Matcher:
    def __init__(self, state, divs):
        self.state = state
        pool = [t for t in state if (not divs or state[t].get('div') in divs)]
        self.cands = []
        for t in pool:
            nt = norm(t)
            if nt: self.cands.append((t, nt, set(nt.split())))
    def find(self, api_name):
        if api_name in OVERRIDE and OVERRIDE[api_name] in self.state:
            return OVERRIDE[api_name]
        na = norm(api_name); ta = set(na.split())
        if not ta: return None
        best, bs = None, 0.0
        for name, nt, tc in self.cands:
            if nt == na: return name
            shared = ta & tc
            strong = shared - WEAK
            r = difflib.SequenceMatcher(None, na, nt).ratio()
            if shared:
                big = max(len(w) for w in shared)
                subset = tc <= ta or ta <= tc
                sc = len(strong) * 3 + len(shared) + big * 0.2 + (3 if subset else 0) + r
                if not strong and big < 5 and not subset and r < 0.7:
                    sc = 0                       # only a weak word in common -> not enough
            else:
                sc = r if r >= 0.82 else 0       # no shared token: need very close spelling
            if sc > bs: bs, best = sc, name
        return best if bs > 0 else None

def fetch(key, region):
    if '--mock' in sys.argv:
        for a in sys.argv[sys.argv.index('--mock')+1:]:
            if a.startswith(key+'='):
                return json.load(open(a.split('=',1)[1]))
        return []
    apikey = os.environ.get('THE_ODDS_API_KEY','').strip()
    if not apikey: return None
    url = (f"https://api.the-odds-api.com/v4/sports/{key}/odds/"
           f"?apiKey={apikey}&regions={region}&markets=h2h&oddsFormat=decimal")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            left = r.headers.get('x-requests-remaining')
            data = json.load(r)
        print(f"  [{key}] {len(data)} events (credits left: {left})")
        return data
    except Exception as e:
        print(f"  [{key}] skipped: {e}")
        return []

def consensus(ev):
    hn, an = ev['home_team'], ev['away_team']
    acc = {hn: [], an: [], 'Draw': []}
    for bk in ev.get('bookmakers', []):
        for mk in bk.get('markets', []):
            if mk['key'] != 'h2h': continue
            for o in mk['outcomes']:
                if o['name'] in acc: acc[o['name']].append(o['price'])
    if not (acc[hn] and acc[an] and acc['Draw']): return None
    avg = lambda L: sum(L)/len(L)
    ip = [1/avg(acc[hn]), 1/avg(acc['Draw']), 1/avg(acc[an])]; s = sum(ip)
    return hn, an, [round(x/s,3) for x in ip]

def fdate(iso):
    try: return datetime.datetime.strptime(iso[:10], "%Y-%m-%d").strftime("%d %b")
    except Exception: return iso[:10]

def main():
    models = {'eu': load('model_eu.json'), 'basic': load('model_basic.json')}
    state = load('team_state.json')
    out = []
    for key, code, name, kind, region, divs in LEAGUES:
        events = fetch(key, region)
        if events is None:
            print("THE_ODDS_API_KEY not set — leaving site unchanged."); return
        if not events: continue
        mt = Matcher(state, divs); fixtures = []
        for ev in events:
            if ev.get('sport_key') != key: continue
            c = consensus(ev)
            if not c: continue
            hn, an, mkt = c
            H, A = mt.find(hn), mt.find(an)
            row = {'date': fdate(ev.get('commence_time','')), 'home': (H or hn), 'away': (A or an), 'market': mkt}
            vals = feats_for(kind, state[H], state[A]) if (H and A) else None
            if vals is not None:
                row['model'] = predict(models[kind], vals)
            else:
                row['model'] = mkt; row['nomodel'] = True
            fixtures.append(row)
        if fixtures:
            modeled = sum(1 for f in fixtures if not f.get('nomodel'))
            print(f"{name}: {len(fixtures)} fixtures, {modeled} modelled")
            out.append({'code': code, 'name': name, 'fixtures': fixtures})
    if not out:
        print("No fixtures anywhere. Done (no changes)."); return

    data = load('base_data.json')
    data['upcoming_leagues'] = out
    data['generated'] = datetime.date.today().strftime("%d %b %Y")
    html = open(os.path.join(HERE, 'index.html')).read()
    newdata = "const DATA = " + json.dumps(data) + ";\nconst RES"
    html = re.sub(r"const DATA = .*?;\nconst RES", lambda _m: newdata, html, count=1, flags=re.S)
    open(os.path.join(HERE, 'index.html'), 'w').write(html)
    tot = sum(len(l['fixtures']) for l in out)
    print(f"\n[done] {len(out)} leagues, {tot} fixtures written.")

if __name__ == "__main__":
    main()
