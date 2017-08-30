#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
from PIL import Image

from sklearn.cluster import KMeans

from db import get_session
from tools.facebook_retina_top import FacebookRetinaEntry

print('fetch and prepare data...')

retina = np.zeros((128, 128), dtype=float)
# X, T = [], []
with get_session() as session:
    for entry in session.query(FacebookRetinaEntry).filter(FacebookRetinaEntry.slug == 'ladygaga'):
        sample_retina = np.zeros((128, 128), dtype=int)
        for pos in entry.data['retina']:
            y, x = divmod(pos, 128)
            sample_retina[x, y] = 1
        retina += sample_retina
# X.append(sample_retina)
#        T.append(entry.data['tokens'])

# X = np.array(X)
print('... done;')

max_value = np.amax(retina)
retina /= max_value
Image.fromarray(np.uint8(retina * 255)).show()
retina[retina < 0.33] = 0
retina[retina > 0] = 1.

X = np.array(list(map(np.array, zip(*retina.nonzero()))))
print('clustering...')
clu = KMeans()
predicted = clu.fit_predict(X)
print("num clusters:", len(set(clu.labels_)))

img = Image.new('RGB', (128, 128), 'white')
pixels = img.load()
colors = {}
for (x, y), pred in zip(X, predicted):
    if pred not in colors:
        colors[pred] = (int(random.uniform(0.2, 0.8) * 255), int(random.uniform(0.1, 0.9) * 255), int(random.uniform(0.1, 1.0) * 255))
    pixels[int(x), int(y)] = colors[pred]
img.show()
