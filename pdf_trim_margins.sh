#!/bin/bash

mkdir .tmpimages
python extract_pdf_images.py --pdf-file $1 --img-output-dir .tmpimages

i=200

for file in ./.tmpimages/*.png; do
	convert "$file" -threshold 50% "$file"
	convert "$file" -trim -alpha remove +repage "${file}"
done

img2pdf --pagesize A4 -o "${1%.*}_trimmed.${1##*.}" .tmpimages/*.png
rm -r .tmpimages
