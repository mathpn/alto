import os

import fitz


def recoverpix(doc, item):
    xref = item[0]  # xref of PDF image
    smask = item[1]  # xref of its /SMask

    # special case: /SMask or /Mask exists
    if smask > 0:
        pix0 = fitz.Pixmap(doc.extract_image(xref)["image"])
        if pix0.alpha:  # catch irregular situation
            pix0 = fitz.Pixmap(pix0, 0)  # remove alpha channel
        mask = fitz.Pixmap(doc.extract_image(smask)["image"])

        try:
            pix = fitz.Pixmap(pix0, mask)
        except Exception:  # fallback to original base image in case of problems
            pix = fitz.Pixmap(doc.extract_image(xref)["image"])

        if pix0.n > 3:
            ext = "pam"
        else:
            ext = "png"

        return {  # create dictionary expected by caller
            "ext": ext,
            "colorspace": pix.colorspace.n,
            "image": pix.tobytes(ext),
        }

    # special case: /ColorSpace definition exists
    # to be sure, we convert these cases to RGB PNG images
    if "/ColorSpace" in doc.xref_object(xref, compressed=True):
        pix = fitz.Pixmap(doc, xref)
        pix = fitz.Pixmap(fitz.csRGB, pix)
        return {  # create dictionary expected by caller
            "ext": "png",
            "colorspace": 3,
            "image": pix.tobytes("png"),
        }
    return doc.extract_image(xref)


def extract_pdf_images(
    pdf_file: str,
    output_dir: str,
    min_size: int = 100,
    min_byte_size: int = 2048,
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    doc = fitz.open(pdf_file)
    page_count = doc.page_count

    xreflist = []
    img_files = []
    for pno in range(page_count):
        print(f"-> extracting images from page {pno+1}/{page_count}...")
        il = doc.get_page_images(pno)
        for img in il:
            xref = img[0]
            if xref in xreflist:
                continue
            width = img[2]
            height = img[3]
            if min(width, height) <= min_size:
                continue
            image = recoverpix(doc, img)
            imgdata = image["image"]

            if len(imgdata) <= min_byte_size:
                continue

            img_file = "img_%05i_%05i.%s" % (pno + 1, xref, image["ext"])
            img_file_path = os.path.join(output_dir, img_file)
            img_files.append(img_file)
            fout = open(img_file_path, "wb")
            fout.write(imgdata)
            fout.close()
            xreflist.append(xref)

    return sorted(set(img_files))
