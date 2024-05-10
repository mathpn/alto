# alto

Set of tools to process and enhance digitalized sheet music.

Digitalizing sheet music can be challenging. Most commercial sheet music bindings do not sit well on conventional scanners. Moreover, traditional scanning software often fails to correctly process digitalized sheet music since they're optimized for text.

## Dewarping

Digitalized sheet music is often warped due to page curvature on scanners. This tool dewarps the image using a modified version of the algorithm originally developed by [Matt Zucker](https://github.com/mzucker) on [page_dewarp](https://github.com/mzucker/page_dewarp), which was later ported to Python 3 by [Louis Maddox](https://github.com/lmmx) [here](https://github.com/lmmx/page-dewarp). The algorithm was very well described by the first author [here](https://mzucker.github.io/2016/08/15/page-dewarping.html).

To dewarp the image, keypoints which should be horizontal must be detected. The original implementation does it very well for text. However, sheet music contains text, lines, arcs and other symbols. Thus, this step was modified to detect the horizontal staff lines.

### Using the dewarp tool

To dewarp the image `warped.png`, run:

```bash
python dewarp.py --input-image warped.png
```

## Setup

Create a Python virtual environment. Then, install the dependencies:

```bash
pip install -r requirements.txt
```

## License

The `page_dewarp` repository is under the MIT license, which is included in the LICENSE file without modifications.
