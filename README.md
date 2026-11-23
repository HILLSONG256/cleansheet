# CleanSheet robot 🤖

This repo runs itself. Once a day a free GitHub robot:

1. grabs this week's Premier League fixtures + bookmaker odds (the-odds-api)
2. runs the prediction model
3. refreshes the website and publishes it to Netlify (cleansheetstats.tech)

You don't touch anything day to day. To force a run now: **Actions → Update CleanSheet predictions → Run workflow**.

## What each file is
- `robot.py` — the whole pipeline (plain Python, no installs needed)
- `model.json` — the trained model (tiny, exported numbers)
- `team_state.json` — each team's recent form the model uses
- `base_data.json` — the site's track-record / recent-results content
- `index.html` — the website itself
- `.github/workflows/update.yml` — the daily schedule

## The 3 secrets it needs (Settings → Secrets and variables → Actions)
- `THE_ODDS_API_KEY` — your the-odds-api key
- `NETLIFY_AUTH_TOKEN` — a Netlify personal access token
- `NETLIFY_SITE_ID` — your Netlify site's API ID

## Keeping the model fresh
`model.json` and `team_state.json` are a snapshot of team form. They don't
update themselves. Every month or so, re-export them so the form stays current.
