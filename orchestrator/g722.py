"""
Pure-Python ITU-T G.722 codec — encoder (64 kbit/s, 16 kHz SB-ADPCM).

Fase 1 of the self-hosted orchestrator: encode G722 frames the ReSpeaker
IoT SDK can decode, WITHOUT an ffmpeg subprocess per frame. Faithful port
of the SpanDSP/Asterisk C reference (Steve Underwood, public domain) —
same tables, same arithmetic order, same integer-shift semantics — so the
bitstream matches the reference (and ffmpeg's g722 encoder) exactly.

ITU-T G.722: 16 kHz sampling, 64 kbit/s. One output byte per input PAIR
of samples (QMF splits into low/high sub-bands; 6-bit + 2-bit ADPCM).
A 20 ms frame @16 kHz = 320 samples = 160 bytes (Agora IoT SDK frame).
"""

INT16_MAX = 32767
INT16_MIN = -32768

# Tables from g722_encode.c (SpanDSP/Asterisk reference)
Q6 = (0, 35, 72, 110, 150, 190, 233, 276, 323, 370, 422, 473,
      530, 587, 650, 714, 786, 858, 940, 1023, 1121, 1219, 1339, 1458,
      1612, 1765, 1980, 2195, 2557, 2919, 0, 0)
ILN = (0, 63, 62, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
       18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 0)
ILP = (0, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47,
       46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 0)
WL = (-60, -30, 58, 172, 334, 538, 1198, 3042)
RL42 = (0, 7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 0)
ILB = (2048, 2093, 2139, 2186, 2233, 2282, 2332, 2383, 2435, 2489, 2543,
       2599, 2656, 2714, 2774, 2834, 2896, 2960, 3025, 3091, 3158, 3228,
       3298, 3371, 3444, 3520, 3597, 3676, 3756, 3838, 3922, 4008)
QM4 = (0, -20456, -12896, -8968, -6288, -4240, -2584, -1200,
       20456, 12896, 8968, 6288, 4240, 2584, 1200, 0)
QM2 = (-7408, -1616, 7408, 1616)
QMF_COEFFS = (3, -11, 12, 32, -210, 951, 3876, -805, 362, -156, 53, -11)
IHN = (0, 1, 0)
IHP = (0, 3, 2)
WH = (0, -214, 798)
RH2 = (2, 1, 2, 1)


def _saturate(amp):
    """C int16 truncation with clamping (saturate() in the reference)."""
    if INT16_MIN <= amp <= INT16_MAX:
        return amp
    return INT16_MAX if amp > INT16_MAX else INT16_MIN


class _Band:
    """ADPCM sub-band state (b1 = 6-bit low band, b2 = 2-bit high band)."""
    __slots__ = ("det", "s", "sp", "sz", "nb",
                 "d", "b", "r", "p", "a", "ap", "bp", "sg")

    def __init__(self, det):
        self.det = det
        self.s = 0
        self.sp = 0
        self.sz = 0
        self.nb = 0
        self.d = [0] * 7    # d[0..6]
        self.b = [0] * 7    # b[0..6] (1..6 used)
        self.r = [0] * 3    # r[0..2]
        self.p = [0] * 3    # p[0..2]
        self.a = [0] * 3    # a[0..2] (1..2 used)
        self.ap = [0] * 3
        self.bp = [0] * 7
        self.sg = [0] * 7   # sg[0..2] used by UPPOL2/UPPOL1; sg[0..6] by UPZERO


def _block4(band, d):
    """Block 4: RECONS, PARREC, UPPOL2, UPPOL1, UPZERO, DELAYA, FILTEP, FILTEZ, PREDIC."""
    # RECONS
    band.d[0] = d
    band.r[0] = _saturate(band.s + d)
    # PARREC
    band.p[0] = _saturate(band.sz + d)
    # UPPOL2
    for i in range(3):
        band.sg[i] = band.p[i] >> 15
    wd1 = _saturate(band.a[1] << 2)
    wd2 = -wd1 if band.sg[0] == band.sg[1] else wd1
    if wd2 > 32767:
        wd2 = 32767
    wd3 = (wd2 >> 7) + (128 if band.sg[0] == band.sg[2] else -128)
    wd3 += (band.a[2] * 32512) >> 15
    if wd3 > 12288:
        wd3 = 12288
    elif wd3 < -12288:
        wd3 = -12288
    band.ap[2] = wd3
    # UPPOL1
    band.sg[0] = band.p[0] >> 15
    band.sg[1] = band.p[1] >> 15
    wd1 = 192 if band.sg[0] == band.sg[1] else -192
    wd2 = (band.a[1] * 32640) >> 15
    band.ap[1] = _saturate(wd1 + wd2)
    wd3 = _saturate(15360 - band.ap[2])
    if band.ap[1] > wd3:
        band.ap[1] = wd3
    elif band.ap[1] < -wd3:
        band.ap[1] = -wd3
    # UPZERO
    wd1 = 0 if d == 0 else 128
    band.sg[0] = d >> 15
    for i in range(1, 7):
        band.sg[i] = band.d[i] >> 15
        wd2 = wd1 if band.sg[i] == band.sg[0] else -wd1
        wd3 = (band.b[i] * 32640) >> 15
        band.bp[i] = _saturate(wd2 + wd3)
    # DELAYA
    for i in range(6, 0, -1):
        band.d[i] = band.d[i - 1]
        band.b[i] = band.bp[i]
    for i in range(2, 0, -1):
        band.r[i] = band.r[i - 1]
        band.p[i] = band.p[i - 1]
        band.a[i] = band.ap[i]
    # FILTEP
    wd1 = _saturate(band.r[1] + band.r[1])
    wd1 = (band.a[1] * wd1) >> 15
    wd2 = _saturate(band.r[2] + band.r[2])
    wd2 = (band.a[2] * wd2) >> 15
    band.sp = _saturate(wd1 + wd2)
    # FILTEZ
    band.sz = 0
    for i in range(6, 0, -1):
        wd1 = _saturate(band.d[i] + band.d[i])
        band.sz += (band.b[i] * wd1) >> 15
    band.sz = _saturate(band.sz)
    # PREDIC
    band.s = _saturate(band.sp + band.sz)


class G722Encoder:
    """64 kbit/s G.722: 16 kHz s16 PCM in, octet stream out (1 byte / 2 samples)."""

    def __init__(self):
        self.x = [0] * 24          # QMF delay line
        self.b0 = _Band(32)        # low sub-band
        self.b1 = _Band(8)         # high sub-band

    def encode(self, pcm):
        """
        pcm: sequence of int16 samples (16 kHz, any length — consumed in PAIRS).
        Returns bytes: one G.722 octet per two input samples.
        len(pcm) must be even; a 20 ms frame (320 samples) -> 160 bytes.
        """
        out = bytearray(len(pcm) // 2)
        out_pos = 0
        x = self.x
        b0, b1 = self.b0, self.b1
        qc = QMF_COEFFS
        j = 0
        L = len(pcm)

        while j < L:
            # QMF transmit: shuffle delay down, push the next pair
            x[0] = x[2]; x[1] = x[3]; x[2] = x[4]; x[3] = x[5]
            x[4] = x[6]; x[5] = x[7]; x[6] = x[8]; x[7] = x[9]
            x[8] = x[10]; x[9] = x[11]; x[10] = x[12]; x[11] = x[13]
            x[12] = x[14]; x[13] = x[15]; x[14] = x[16]; x[15] = x[17]
            x[16] = x[18]; x[17] = x[19]; x[18] = x[20]; x[19] = x[21]
            x[20] = x[22]; x[21] = x[23]
            x[22] = pcm[j]
            x[23] = pcm[j + 1]
            j += 2

            sumodd = 0
            sumeven = 0
            for i in range(12):
                sumodd += x[2 * i] * qc[i]
                sumeven += x[2 * i + 1] * qc[11 - i]
            xlow = (sumeven + sumodd) >> 14
            xhigh = (sumeven - sumodd) >> 14

            # --- Block 1L, SUBTRA ---
            el = _saturate(xlow - b0.s)
            # --- Block 1L, QUANTL ---
            wd = el if el >= 0 else -(el + 1)
            i = 1
            while i < 30:
                wd1 = (Q6[i] * b0.det) >> 12
                if wd < wd1:
                    break
                i += 1
            ilow = ILN[i] if el < 0 else ILP[i]
            # --- Block 2L, INVQAL ---
            ril = ilow >> 2
            wd2 = QM4[ril]
            dlow = (b0.det * wd2) >> 15
            # --- Block 3L, LOGSCL ---
            il4 = RL42[ril]
            wd = (b0.nb * 127) >> 7
            b0.nb = wd + WL[il4]
            if b0.nb < 0:
                b0.nb = 0
            elif b0.nb > 18432:
                b0.nb = 18432
            # --- Block 3L, SCALEL ---
            wd1 = (b0.nb >> 6) & 31
            wd2 = 8 - (b0.nb >> 11)
            wd3 = (ILB[wd1] << -wd2) if wd2 < 0 else (ILB[wd1] >> wd2)
            b0.det = wd3 << 2
            _block4(b0, dlow)

            # --- Block 1H, SUBTRA ---
            eh = _saturate(xhigh - b1.s)
            # --- Block 1H, QUANTH ---
            wd = eh if eh >= 0 else -(eh + 1)
            wd1 = (564 * b1.det) >> 12
            mih = 2 if wd >= wd1 else 1
            ihigh = IHN[mih] if eh < 0 else IHP[mih]
            # --- Block 2H, INVQAH ---
            dhigh = (b1.det * QM2[ihigh]) >> 15
            # --- Block 3H, LOGSCH ---
            ih2 = RH2[ihigh]
            wd = (b1.nb * 127) >> 7
            b1.nb = wd + WH[ih2]
            if b1.nb < 0:
                b1.nb = 0
            elif b1.nb > 22528:
                b1.nb = 22528
            # --- Block 3H, SCALEH ---
            wd1 = (b1.nb >> 6) & 31
            wd2 = 10 - (b1.nb >> 11)
            wd3 = (ILB[wd1] << -wd2) if wd2 < 0 else (ILB[wd1] >> wd2)
            b1.det = wd3 << 2
            _block4(b1, dhigh)

            out[out_pos] = (ihigh << 6) | ilow
            out_pos += 1
        return bytes(out)