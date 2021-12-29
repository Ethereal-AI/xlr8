# xlr8

Fast cosine similarity for Python

### Installing the package
1. Clone the repository.<br>
2. Run `pip install -e .` inside the local repository.<br>

### Usage

Using the cosine similarity function is straightforward.
```python
from xlr8.similarity import cosine_similarity
import numpy as np

x = np.random.rand(1000,1000)
y = np.random.rand(1000,1000)

cosine_similarity(x,y)
```