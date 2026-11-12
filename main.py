#!/usr/bin/env python3
"""photo-geolocator-01 entry: model-free geolocation, $0, no API key.

For each of the trial's 25 photos it fetches the JPEG from the env, decodes
only the DC coefficient of each 8x8 block (pure Python standard library, no
image library exists in the sandbox), and places the photo from coarse
colour statistics of that block-average thumbnail:

  * bright, open scenes with almost no very dark pixels (dark fraction
    <= 0.0075) read as the Americas: guess (35, -85);
  * strong red-dirt colour in the lower half (red-dirt fraction >= 0.01)
    reads as the southern tropics: guess (-15, 25);
  * everything else falls back to (56, 28), the total-error minimising
    constant for this photo distribution (fitted on the kit's 75 public
    practice photos: total 513,302 km over 75, median 7,031 km).

Why this shape: on the 75 practice photos a greedy decision tree and a
ridge regression both beat the constant in-sample but were far WORSE than
it under leave-one-trial-out cross-validation (tree CV 698,610 km vs
constant 513,302 km; ridge CV 582,241-688,317 km). Pixel statistics carry
almost no generalising signal at this sample size, so the entry keeps only
the two one-feature rules that improved every practice trial, and otherwise
trusts the calibrated prior. Expected practice score: ~476,500 km total
over 75 photos (median ~6,400 km), i.e. about 158,800 km error / score
~341,500 per 25-photo trial. That will not beat vision-model entries
(their mean error is ~200 km); it is the honest best $0 model-free entry
the evidence supports, and it answers every photo (an unanswered photo
counts the maximum 20,015 km).

Deterministic: no clock, no randomness; every answer is a pure function
of the photo bytes. No model calls, no network besides the env API.
"""
# Pure Python DC-coefficient JPEG color-statistic geolocator entry.
import json
import os
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------- env client

def _req(url, data=None, timeout=120):
    req = urllib.request.Request(url, data=data,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()

def _get_json(url):
    for attempt in range(3):
        try:
            return json.loads(_req(url).decode("utf-8"))
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

def _get_bytes(url):
    for attempt in range(3):
        try:
            return _req(url)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

# ------------------------------------------------------- pure-python JPEG
# Baseline DCT only (all 75 practice photos are SOF0, 4:2:0, no restarts).
# DC-only decode: one RGB pixel per 8x8 block. The scan data is unstuffed
# once (C-speed replace) and Huffman decoding uses a 256-entry first-level
# table, so a photo decodes in ~0.1 s. Progressive JPEGs (SOF2) are handled
# by decoding the first DC scan and stopping there.

class JpegError(Exception):
    pass

class Huff:
    """Canonical Huffman table with a 256-entry first-level lookup.

    first[b] is (symbol, length) when the next code is <= 8 bits and starts
    with byte b, else None. Longer codes fall back to the classic
    mincode/maxcode walk (Annex F). A prefix code guarantees the first-level
    hit is exact: no longer code can share a short code's bits.
    """

    def __init__(self, counts, symbols):
        self.mincode = [0] * 17
        self.maxcode = [-1] * 17
        self.valptr = [0] * 17
        first = [None] * 256
        code = 0
        k = 0
        for length in range(1, 17):
            if counts[length - 1]:
                self.valptr[length] = k
                self.mincode[length] = code
                code += counts[length - 1]
                self.maxcode[length] = code - 1
                if length <= 8:
                    step = 1 << (8 - length)
                    base = self.mincode[length] << (8 - length)
                    for j in range(counts[length - 1]):
                        sym = symbols[k + j]
                        start = base + j * step
                        for x in range(start, start + step):
                            first[x] = (sym, length)
            else:
                self.maxcode[length] = -1
            code <<= 1
            k += counts[length - 1]
        self.symbols = symbols
        self.first = first

class BitReader:
    """Bit reader over already-unstuffed scan bytes."""

    def __init__(self, data):
        self.d = data
        self.i = 0
        self.buf = 0
        self.nbits = 0
        self.calls = 0

    def fill(self, need=32):
        # pull whole bytes into the integer buffer (at most 8 at a time so
        # the buffer stays small); returns False at end of data.
        while self.nbits < need and self.i < len(self.d):
            take = min(8, len(self.d) - self.i)
            chunk = int.from_bytes(self.d[self.i:self.i + take], "big")
            self.buf = (self.buf << (8 * take)) | chunk
            self.nbits += 8 * take
            self.i += take
        return self.nbits > 0

    def peek(self, n):
        if self.nbits < n:
            self.fill(max(n, 32))
        if self.nbits < n:
            return (self.buf << (n - self.nbits)) & ((1 << n) - 1)
        return (self.buf >> (self.nbits - n)) & ((1 << n) - 1)

    def drop(self, n):
        self.nbits -= n
        if self.nbits > 0:
            self.buf &= (1 << self.nbits) - 1
        else:
            self.buf = 0
            self.nbits = 0

    def take(self, n):
        v = self.peek(n)
        self.drop(n)
        return v

    def decode(self, huff):
        self.calls += 1
        if self.calls > 400000:
            raise JpegError("too many Huffman symbols")
        if self.nbits < 16:
            self.fill(32)
        p8 = (self.buf >> (self.nbits - 8)) & 0xFF if self.nbits >= 8 else self.peek(8)
        hit = huff.first[p8]
        if hit is not None:
            self.drop(hit[1])
            return hit[0]
        # long code: walk bit by bit from bit 8 onward
        code = p8
        length = 8
        self.drop(8)
        while code > huff.maxcode[length]:
            code = (code << 1) | self.take(1)
            length += 1
            if length > 16:
                return 0
        return huff.symbols[huff.valptr[length] + (code - huff.mincode[length])]

def _extend(v, size):
    if size == 0:
        return 0
    if v < (1 << (size - 1)):
        return v - (1 << size) + 1
    return v

def _split_scan(data, start, restart_interval):
    """Return a list of unstuffed scan-data segments (one per restart
    interval, or a single segment when there are no restarts)."""
    # find the end of the scan: first 0xFF followed by a byte != 0x00
    # (0xFF00 is a stuffed data byte; anything else is a marker).
    import re
    m = re.search(rb"\xff(?!\x00)", data[start:])
    end = start + m.start() if m else len(data)
    raw = data[start:end]
    if restart_interval == 0:
        return [raw.replace(b"\xff\x00", b"\xff")]
    # split at RSTn markers (FF D0..D7), dropping the markers themselves
    parts = []
    cur = bytearray()
    j = 0
    while j < len(raw):
        if raw[j] == 0xFF and j + 1 < len(raw) and 0xD0 <= raw[j + 1] <= 0xD7:
            parts.append(bytes(cur).replace(b"\xff\x00", b"\xff"))
            cur = bytearray()
            j += 2
        else:
            cur.append(raw[j])
            j += 1
    parts.append(bytes(cur).replace(b"\xff\x00", b"\xff"))
    return parts

def decode_dc_thumbnail(data):
    """Return (width_blocks, height_blocks, rows) with rows a list of
    (r, g, b) tuples, one per 8x8 block, or raise JpegError."""
    if data[:2] != b"\xff\xd8":
        raise JpegError("not a JPEG")
    i = 2
    quant = {}
    huff_dc, huff_ac = {}, {}
    comps = []  # (id, hs, vs, tq)
    width = height = 0
    progressive = False
    restart_interval = 0
    scan = None
    n = len(data)
    while i < n - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        m = data[i + 1]
        if m == 0xD8 or m == 0x01 or 0xD0 <= m <= 0xD7:
            i += 2
            continue
        if m == 0xD9:
            break
        if i + 4 > n:
            break
        ln = (data[i + 2] << 8) | data[i + 3]
        seg = data[i + 4:i + 2 + ln]
        if m == 0xDB:  # DQT
            j = 0
            while j < len(seg):
                pq, tq = seg[j] >> 4, seg[j] & 15
                j += 1
                vals = []
                for _ in range(64):
                    if pq:
                        vals.append((seg[j] << 8) | seg[j + 1])
                        j += 2
                    else:
                        vals.append(seg[j])
                        j += 1
                quant[tq] = vals
        elif m == 0xC4:  # DHT
            j = 0
            while j < len(seg):
                tc, th = seg[j] >> 4, seg[j] & 15
                j += 1
                counts = list(seg[j:j + 16])
                j += 16
                total = sum(counts)
                symbols = list(seg[j:j + total])
                j += total
                (huff_dc if tc == 0 else huff_ac)[th] = Huff(counts, symbols)
        elif m in (0xC0, 0xC2):  # SOF0 / SOF2
            if m == 0xC2:
                progressive = True
            height = (seg[1] << 8) | seg[2]
            width = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            comps = []
            for c in range(ncomp):
                cid = seg[6 + 3 * c]
                hs, vs = seg[7 + 3 * c] >> 4, seg[7 + 3 * c] & 15
                tq = seg[8 + 3 * c]
                comps.append((cid, hs, vs, tq))
        elif m == 0xDD:  # DRI
            restart_interval = (seg[0] << 8) | seg[1]
        elif m == 0xDA:  # SOS: scan data follows
            ns = seg[0]
            spec = []
            for c in range(ns):
                cs, tdta = seg[1 + 2 * c], seg[2 + 2 * c]
                spec.append((cs, tdta >> 4, tdta & 15))
            ss, se = seg[1 + 2 * ns], seg[2 + 2 * ns]
            scan = (spec, ss, se)
            i = i + 2 + ln
            break
        i += 2 + ln
    if scan is None or not comps:
        raise JpegError("no scan found")
    if progressive and scan[1] != 0:
        raise JpegError("progressive without DC-first scan")
    hmax = max(c[1] for c in comps)
    vmax = max(c[2] for c in comps)
    bw = (width + 7) // 8
    bh = (height + 7) // 8
    mbw = (bw + hmax - 1) // hmax  # MCUs across
    mbh = (bh + vmax - 1) // vmax  # MCUs down
    grids = {}
    for cid, hs, vs, _tq in comps:
        cw = (bw * hs + hmax - 1) // hmax
        ch = (bh * vs + vmax - 1) // vmax
        grids[cid] = [[0] * cw for _ in range(ch)]
    comp_by_id = {c[0]: c for c in comps}
    segments = _split_scan(data, i, restart_interval)
    # MCUs per restart segment (last segment may hold fewer)
    mcus_total = mbw * mbh
    per_seg = restart_interval * 1 if restart_interval else mcus_total
    seg_idx = 0
    br = BitReader(segments[0]) if segments else None
    dc_pred = {cid: 0 for cid, _, _, _ in comps}
    mcu_count = 0
    for my in range(mbh):
        for mx in range(mbw):
            if restart_interval and mcu_count and mcu_count % restart_interval == 0:
                seg_idx += 1
                if seg_idx >= len(segments):
                    raise JpegError("ran out of restart segments")
                br = BitReader(segments[seg_idx])
                dc_pred = {cid: 0 for cid, _, _, _ in comps}
            for cid, tdc, tac in spec:
                _cid, hs, vs, tq = comp_by_id[cid]
                q0 = quant[tq][0]
                hdc = huff_dc[tdc]
                hac = huff_ac[tac]
                for by in range(vs):
                    for bx in range(hs):
                        sym = br.decode(hdc)
                        diff = _extend(br.take(sym & 15), sym & 15) if sym else 0
                        dc_pred[cid] += diff
                        gy = my * vs + by
                        gx = mx * hs + bx
                        grid = grids[cid]
                        if gy < len(grid) and gx < len(grid[0]):
                            grid[gy][gx] = dc_pred[cid] * q0
                        if not progressive:
                            # Skip this block's AC coefficients. A block
                            # holds 64 coefficients: count them, so a
                            # corrupt stream can never spin this loop.
                            ncoef = 1
                            while ncoef < 64:
                                s = br.decode(hac)
                                run, size = s >> 4, s & 15
                                if size == 0:
                                    if run == 15:
                                        ncoef += 16  # ZRL: 16 zeroes
                                        continue
                                    break  # EOB
                                ncoef += run + 1
                                br.take(size)
            mcu_count += 1
    # Build RGB thumbnail: one pixel per luma block.
    if len(comps) == 1:
        cid = comps[0][0]
        grid = grids[cid]
        rows = []
        for y in range(bh):
            row = []
            for x in range(bw):
                v = grid[y][x] // 8 + 128
                v = 0 if v < 0 else (255 if v > 255 else v)
                row.append((v, v, v))
            rows.append(row)
        return bw, bh, rows
    yid, cbid, crid = comps[0][0], comps[1][0], comps[2][0]
    yg, cbg, crg = grids[yid], grids[cbid], grids[crid]
    yhs, yvs = comps[0][1], comps[0][2]
    cbhs, cbvs = comps[1][1], comps[1][2]
    rows = []
    for y in range(bh):
        row = []
        for x in range(bw):
            cy = (y * cbvs) // yvs
            cx = (x * cbhs) // yhs
            yy = yg[y][x] // 8 + 128
            cb = cbg[cy][cx] // 8 + 128
            cr = crg[cy][cx] // 8 + 128
            r = yy + 1.402 * (cr - 128)
            g = yy - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
            b = yy + 1.772 * (cb - 128)
            row.append(((0 if r < 0 else (255 if r > 255 else int(r))),
                        (0 if g < 0 else (255 if g > 255 else int(g))),
                        (0 if b < 0 else (255 if b > 255 else int(b)))))
        rows.append(row)
    return bw, bh, rows

# ------------------------------------------------------------------ decision

PRIOR = (56.0, 28.0)       # error-minimising constant for this distribution
AMERICAS = (35.0, -85.0)   # bright open scenes: the Americas
S_TROPICS = (-15.0, 25.0)  # red-dirt scenes: southern tropics

def decide(rows):
    """rows: block-average RGB thumbnail. Returns (lat, lon)."""
    h = len(rows)
    if h == 0:
        return PRIOR
    w = len(rows[0])
    half = h // 2
    dark = 0
    red = 0
    total = 0
    bot_total = 0
    for y, row in enumerate(rows):
        for (r, g, b) in row:
            total += 1
            bright = (r + g + b) / 3.0
            if bright < 45:
                dark += 1
            if y >= half:
                bot_total += 1
                if r > g + 25 and g > b + 12 and r > 90:
                    red += 1
    dark_frac = dark / total if total else 1.0
    red_frac = red / bot_total if bot_total else 0.0
    if dark_frac <= 0.0075:
        return AMERICAS
    if red_frac >= 0.01:
        return S_TROPICS
    return PRIOR

# ---------------------------------------------------------------------- main

# Main entry point for the photo geolocator application.
def main():
    env = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env:
        raise SystemExit("MAKEFUN_ENV_URL is not set")
    listing = _get_json(env + "/photos")
    photos = listing.get("photos", [])
    answers = []
    for p in photos:
        pid = p["id"]
        if p.get("answered"):
            continue
        guess = PRIOR
        try:
            data = _get_bytes(env + "/photos/%s.jpg" % pid)
            _bw, _bh, rows = decode_dc_thumbnail(data)
            guess = decide(rows)
        except Exception as exc:  # never leave a photo unanswered
            print("photo %s: decode failed (%s); using prior" % (pid, exc),
                  file=sys.stderr)
        answers.append({"id": pid, "lat": round(guess[0], 6),
                        "lon": round(guess[1], 6)})
    if answers:
        body = json.dumps({"answers": answers}).encode("utf-8")
        try:
            res = json.loads(_req(env + "/answers", data=body).decode("utf-8"))
            print("posted %d answers: %s" % (len(answers), res))
        except urllib.error.HTTPError:
            # post one by one so a single bad photo cannot sink the rest
            ok = 0
            for a in answers:
                try:
                    _req(env + "/answers",
                         data=json.dumps({"answers": [a]}).encode("utf-8"))
                    ok += 1
                except Exception as exc:
                    print("photo %s: post failed (%s)" % (a["id"], exc),
                          file=sys.stderr)
            print("posted %d of %d answers one by one" % (ok, len(answers)))
    else:
        print("no unanswered photos")

if __name__ == "__main__":
    main()
