import requests, os

HERE = os.getcwd()

CSV = os.path.join(HERE, 'output', 'submission.csv')

print(f'Submitting {CSV}...')

with open(CSV, 'rb') as f:

    resp = requests.post(

        'http://35.192.205.84:80/submit/27-p4ms',

        headers={'X-API-Key': '31fd8c57481049a79ce9e526df488d56'},

        files={'file': ('submission.csv', f, 'text/csv')},

        timeout=120,

    )

print('Status:', resp.status_code)

try:

    print('Response:', resp.json())

except Exception:

    print('Raw:', resp.text)
