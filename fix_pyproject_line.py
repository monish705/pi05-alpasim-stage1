from pathlib import Path
p = Path('/home/ubuntu/alpamayo-stack/alpasim/src/driver/pyproject.toml')
lines = p.read_text(encoding='utf-8').splitlines()
lines[43] = 'alpamayo1_5 = { git = "https://github.com/NVlabs/alpamayo1.5.git", rev = "2eff7037e47afb96a578b3d1bca453a373cd781e" }'
p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(lines[43])
