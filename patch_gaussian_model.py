from pathlib import Path
import re

path = Path('/home/ubuntu/autosim/gaussian-grouping/scene/gaussian_model.py')
text = path.read_text()
pattern = re.compile(r"\s*objects_dc = np.zeros\(\(xyz.shape\[0\], self.num_objects, 1\)\)\n\s*for idx in range\(self.num_objects\):\n\s*objects_dc\[:,idx,0\] = np.asarray\(plydata.elements\[0\]\[\"obj_dc_\"\+str\(idx\)\]\)\n")
new = '''        obj_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("obj_dc_")]
        objects_dc = np.zeros((xyz.shape[0], self.num_objects, 1))
        if len(obj_names) >= self.num_objects:
            for idx in range(self.num_objects):
                objects_dc[:,idx,0] = np.asarray(plydata.elements[0]["obj_dc_"+str(idx)])
        else:
            # no object channels in PLY; leave zeros
            pass
'''
text, n = pattern.subn(new, text, count=1)
if n != 1:
    raise SystemExit('pattern not found or multiple found: %d' % n)
path.write_text(text)
print('patched gaussian_model.py')
