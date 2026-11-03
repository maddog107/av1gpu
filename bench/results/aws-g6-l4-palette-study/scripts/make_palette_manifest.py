#!/usr/bin/env python3
"""Build bench/palette_study_images.csv: the downloaded test images of the palette study (bench/results/
aws-g6-l4-palette-study/), with URL, licence, author, source page and SHA-256 of the file as served.

  make_palette_manifest.py OUT.csv DLDIR

Classes:
  comics   scanned public-domain comic pages (JPEG as served, halftone + paper grain): archive.org books marked
           Public Domain Mark 1.0 (golden-age titles whose copyright was not renewed; also on Comic Book Plus /
           Digital Comic Museum) and Little Nemo (1905-1911) on Wikimedia Commons
  flat     flat-colour / anime-style digital art, PNG originals from Wikimedia Commons (CC BY-SA / CC BY / CC0 /
           PD; the study flattens transparency onto white and also codes a JPEG q90 4:2:0 version of each)
  painted  Pepper&Carrot pages (David Revoy, CC BY 4.0): painted webcomic, JPEG as published
  screen   real screenshots (PNG) from Wikimedia Commons; the synthetic graphics and the headless-browser
           renders are made by the study's scripts (mk_graphics.py, render_pages.sh), not downloaded
  photo    the 24 Kodak PNGs (r0k.us); the size-range JPEGs come from bench/openimages_v4_gamut.csv
"""
import csv
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

from PIL import Image

UA = "av1gpu-bench/0.1 (https://github.com/maddog107/av1gpu; palette study)"
COMMONS = "https://commons.wikimedia.org/w/api.php"
IA_PD = "https://creativecommons.org/publicdomain/mark/1.0/"

IA = [  # (item, archive, [page file names], short name, publisher)
    ("superworld-comics-001",
     "Superworld Comics 001 (Komos Publications) (1940) (c2c) (Ambroise 'Dipthong' Jones+Mark Bowen) FIXED.cbz",
     [f"Superworld Comics 001 {p:02d}.jpg" for p in (3, 7, 13, 22, 29, 37, 45, 53)], "superworld1940", "Komos Publications, 1940"),
    ("americas-greatest-comics-1-fawcett-publications-1941",
     "Americas Greatest 001 (Fawcett) (1941) (c2c - 2 fiche) (DrJohnnyDiablo+Superscan+srca1941).cbz",
     [f"America's Greatest 001-{p:02d}.jpg" for p in (5, 11, 17, 26, 38, 50, 62, 77)], "americasgreatest1941",
     "Fawcett Publications, 1941"),
    ("rocket-to-the-moon-nn-avon-1951", "Rocket to the Moon #nn (Avon, 1951).cbz", ["10.jpg", "19.jpg", "25.jpg"],
     "rockettothemoon1951", "Avon, 1951"),
    ("my-secret-marriage-008-1954-07.-superior-darwin-ia", "My Secret Marriage 008 1954-07.Superior (Darwin-IA).cbr",
     [f"My Secret Marriage 008 1954-07.Superior (Darwin-IA)/MySecretMarriage08p{p:03d}.jpg" for p in (10, 19, 28)],
     "mysecretmarriage1954", "Superior Comics, 1954"),
    ("flaming-love-01-1949-12.-quality-hp-darwination-ia", "Flaming Love 01 1949-12.Quality (HP-Darwination-IA).cbr",
     [f"Flaming Love 01 1949-12.Quality (HP-Darwination-IA)/FlamingLove01p{p:02d}.jpg" for p in (10, 19, 28)],
     "flaminglove1949", "Quality Comics, 1949"),
]

COMMONS_FILES = {
    "comics": [
        "File:Little Nemo 1905-10-15.jpg",
        "File:Little Nemo 1906-01-07 (non-US).jpeg",
        "File:Little Nemo 1908-02-16.jpg",
        "File:Little Nemo 1908-07-19b.jpg",
        "File:Little Nemo 1909-06-27.jpg",
        "File:Winsor McCay, Little Nemo in Slumberland - Climbing the Great North Pole, 1907, NGA 142154.jpg",
    ],
    "flat": [
        "File:Wikipe-tan mopping.png",
        "File:Wikipe-tan needs your help.png",
        "File:Wikipe-tan GothLoli.png",
        "File:Wikipe-tan frontview.png",
        "File:Wikipe tan casual.png",
        "File:Wikipe-tan wearing baro't saya.png",
        "File:Wikipe-tan in Cleopatra-style costume.png",
        "File:Wikipe-tan coaching.png",
        "File:Wikipe-tan at Valentine's Day.png",
        "File:Wikipe-tan in Santa Costume.png",
        "File:Wikipe-tan Cartoon - Something is missing.png",
        "File:Commons-tan Cartoon - There are these days.png",
        "File:Commons-tan brainwashing.png",
        "File:Morevna Project - Pepper&Carrot (animated comic) - Episode 6 - The Potion Contest - Poster.png",
        "File:Ivan Tsarevich and cameo appearances of Pepper and Carrot in Morevna Project episode 4.png",
        "File:Marcosticks-Human hand curled in prep for holding chopsticks.png",
        "File:Personal Arms Of Wikipe-tan.png",
        "File:Wikipetan slime rants.png",
        # SVG originals: the study renders them with rsvg-convert (long side 2560 px, on white)
        "File:Anime Girl.svg",
        "File:Neko Wikipe-tan.svg",
        "File:Purple Haired Anime Girl With Star Background.svg",
        "File:Sylva vector art.svg",
        "File:Wikipe-tan doctor.svg",
        "File:Wikipe-tan soldier.svg",
    ],
    "screen": [
        "File:Frwiki-Tour Eiffel-after.png",
        "File:RecentChanges2012-01-19.PNG",
        "File:Screenshot of BLP edit counter tool.png",
        "File:AI Source Verification.png",
        "File:Rc-mittel.png",
        "File:DiffViewer3.png",
        "File:AlmaLinux 8.9 KDE Dolphin 22.04.1 Breeze Light - English.png",
        "File:Arco Linux 24.05.11 Plasma Dolphin 24.02.2 Breeze Dark - English.png",
        "File:ArcoLinux 24.09.02 Plasma Firefox 129.0.1 about - English.png",
        "File:LibreOffice Writer 7.5.0 Windows10.png",
        "File:LibreOffice Calc 7.1.2 (released in 2021-04, running on Linux and GNOME with the default icon set).png",
        "File:Kdenlive 20.04.1.png",
    ],
}

PEPPER = [("ep35_The-Reflection", "E35", p) for p in ("P01", "P04", "P07")] + \
         [("ep38_The-Healer", "E38", p) for p in ("P02", "P05", "P08")]


def get(url, tries=4):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=180).read()
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(3 * (t + 1))


def commons_info(titles):
    out = {}
    for i in range(0, len(titles), 40):
        q = dict(action="query", prop="imageinfo", titles="|".join(titles[i:i + 40]), iiprop="url|size|mime|sha1|extmetadata",
                 iiextmetadatafilter="LicenseShortName|LicenseUrl|Artist", format="json")
        d = json.loads(get(COMMONS + "?" + urllib.parse.urlencode(q)))
        norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
        for p in d["query"]["pages"].values():
            ii = p["imageinfo"][0]
            em = ii.get("extmetadata", {})
            artist = re.sub(r"<[^>]+>", "", em.get("Artist", {}).get("value", "")).strip()
            out[norm.get(p["title"], p["title"])] = dict(
                url=ii["url"].split("?")[0], landing=ii["descriptionurl"], license=em.get("LicenseShortName", {}).get("value", ""),
                license_url=em.get("LicenseUrl", {}).get("value", ""), author=" ".join(artist.split())[:120])
    return out


class svg_size:
    """width / height of an SVG from its root element's width/height or viewBox (Image.open stand-in)"""
    mode = "SVG"

    def __init__(self, data):
        head = data[:4000].decode("utf-8", "replace")
        root = re.search(r"<svg[^>]*>", head, re.S).group(0)
        w = re.search(r'\swidth="([0-9.]+)', root)
        h = re.search(r'\sheight="([0-9.]+)', root)
        if w and h:
            self.width, self.height = round(float(w.group(1))), round(float(h.group(1)))
        else:
            vb = [float(v) for v in re.search(r'viewBox="([^"]+)"', root).group(1).replace(",", " ").split()]
            self.width, self.height = round(vb[2]), round(vb[3])
        self.size = (self.width, self.height)


def slug(s):
    s = re.sub(r"^File:", "", s)
    s = re.sub(r"\.(png|jpg|jpeg|svg)$", "", s, flags=re.I)
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:60].lower()


def main():
    out_csv, dl = sys.argv[1], sys.argv[2]
    os.makedirs(dl, exist_ok=True)
    rows = []

    def add(cls, file, url, author, title, landing, lic, lic_url, note=""):
        path = os.path.join(dl, file)
        if not os.path.exists(path):
            data = get(url)
            open(path, "wb").write(data)
        data = open(path, "rb").read()
        im = Image.open(path) if not path.endswith(".svg") else svg_size(data)
        rows.append(dict(file=file, image_id=os.path.splitext(file)[0], cls=cls, url=url,
                         sha256=hashlib.sha256(data).hexdigest(), width=im.width, height=im.height, bytes=len(data),
                         mode=im.mode, author=author, title=title, landing_url=landing, license=lic,
                         license_url=lic_url, note=note))
        print(cls, file, im.size, im.mode, len(data), flush=True)

    for item, arc, pages, short, pub in IA:
        for p in pages:
            url = f"https://archive.org/download/{item}/{urllib.parse.quote(arc)}/{urllib.parse.quote(p)}"
            stem = re.sub(r"[^0-9]", "", os.path.basename(p))[-3:]
            add("comics", f"comics_{short}_p{stem}.jpg", url, pub, os.path.basename(p),
                f"https://archive.org/details/{item}", "Public Domain Mark 1.0 (archive.org)", IA_PD)
    for cls, titles in COMMONS_FILES.items():
        info = commons_info(titles)
        for t in titles:
            r = info[t]
            ext = os.path.splitext(r["url"])[1].lower().replace("jpeg", "jpg")
            add(cls, f"{cls}_{slug(t)}{ext}", r["url"], r["author"], t[5:], r["landing"], r["license"], r["license_url"])
    for ep, e, p in PEPPER:
        url = f"https://www.peppercarrot.com/0_sources/{ep}/hi-res/en_Pepper-and-Carrot_by-David-Revoy_{e}{p}.jpg"
        add("painted", f"painted_peppercarrot_{e.lower()}{p.lower()}.jpg", url, "David Revoy",
            f"Pepper&Carrot {ep} {p}", f"https://www.peppercarrot.com/en/webcomic/{ep}.html", "CC BY 4.0",
            "https://creativecommons.org/licenses/by/4.0/")
    for i in range(1, 25):
        add("photo", f"photo_kodim{i:02d}.png", f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png", "Eastman Kodak",
            f"Kodak Lossless True Color Image Suite kodim{i:02d}", "http://r0k.us/graphics/kodak/",
            "released by Kodak for unrestricted use", "")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "rows,", sum(r["width"] * r["height"] for r in rows) / 1e6, "MP")


if __name__ == "__main__":
    main()
