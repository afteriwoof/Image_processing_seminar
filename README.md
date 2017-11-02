# Digital Image Processing: Intro & Applications

### Jason Byrne PhD

#### Data Science Team Seminar - Royal Mail Group Business Intelligence 

#### Sept 2017

Sources include: 

"Digital Image Processing" - Gonzalez & Woods

http://scikit-image.org/

https://matplotlib.org/

http://www.scipy-lectures.org/
    
Notebook content available at this dropbox link:
tiny.cc/fbvfny

Viewable on Jupyter nbviewer here:
http://nbviewer.jupyter.org/github/afteriwoof/Image_processing_seminar/blob/master/Image_processing_seminar.ipynb

```
from skimage import data, io, filters, util

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.size'] = 10

import numpy as np

import skimage
print(skimage.__version__)
```
0.13.0

A digital image is a numeric representation (a matrix) of an image f(x,y) that has been discretized both in spatial coordinates and brightness.
# Option to load an example image: img = data.horse() img = data.moon() img = data.astronaut() img = data.coins() io.imshow(img) io.show() # Or load your own image instead (though it might not be ideal for the demo).


## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/afteriwoof/Image_processing_seminar/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/afteriwoof/Image_processing_seminar/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
