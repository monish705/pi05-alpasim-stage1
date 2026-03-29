import re
import requests
u = 'https://www.nuscenes.org/nuscenes#download'
r = requests.get(u, timeout=30)
print('status', r.status_code, 'len', len(r.text))
for token in ['map', 'expansion', 'v1.0-mini', '.zip', '.tgz', '.tar', '/data/']:
    if token in r.text:
        print('contains', token)
for m in re.findall(r'https://www\.nuscenes\.org/data/[^"\']+', r.text):
    print(m)