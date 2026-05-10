import pandas as pd, re, glob
files = sorted(glob.glob('validation_pii/*.parquet'))
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
NAME_RE = re.compile(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b')
EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
domain_eq_last = 0
total = 0
for _, row in df.iterrows():
    name = email = None
    for turn in row['conversation']:
        m = NAME_RE.search(turn['instruction'] + ' ' + turn['output'])
        if m and not name:
            cand = m.group(1)
            if cand.split()[0].lower() not in ('the','what','for','is'):
                name = cand
        em = EMAIL_RE.search(turn['output'])
        if em and 'redacted' not in em.group(0).lower():
            email = em.group(0)
    if name and email:
        total += 1
        last = name.split()[-1].lower()
        domroot = email.split('@')[1].split('.')[0].lower()
        if domroot == last:
            domain_eq_last += 1
print(f'domain==lastname in REAL data: {domain_eq_last}/{total}')
