from pathlib import Path
p = Path('/home/ubuntu/alpamayo-stack/alpasim/src/driver/pyproject.toml')
lines = p.read_text(encoding='utf-8').splitlines()
out = []
seen = 0
for line in lines:
    if line.startswith('alpamayo1_5 = { git = "https://github.com/NVlabs/alpamayo1.5.git"'):
        seen += 1
        if seen > 1:
            continue
    out.append(line)
p.write_text('\n'.join(out) + '\n', encoding='utf-8')
for i, line in enumerate(out[40:46], start=41):
    print(i, line)
