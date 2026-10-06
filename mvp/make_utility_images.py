"""Render the 'utility' half of the demo library: receipts, prescriptions, medicine
strips, payment and chat screenshots, notes, tickets, ID card.

All names, brands and numbers are fictional. Images are made to look like a phone
photo of paper (slight rotation, table background) or like a phone screenshot.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "library" / "utility"
OUT.mkdir(parents=True, exist_ok=True)
F = "/usr/share/fonts/truetype/dejavu/"
random.seed(11)


def font(size, bold=False, mono=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    if bold:
        name += "-Bold"
    return ImageFont.truetype(F + name + ".ttf", size)


def table_photo(paper: Image.Image, bg=(120, 92, 70), angle=None) -> Image.Image:
    """Place a paper image on a wooden-ish table and tilt it, like a quick phone photo."""
    W, H = 720, 960
    base = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(base)
    for y in range(0, H, 6):  # wood grain
        shade = random.randint(-10, 10)
        d.line([(0, y), (W, y + random.randint(-3, 3))], fill=tuple(max(0, min(255, c + shade)) for c in bg), width=3)
    angle = random.uniform(-6, 6) if angle is None else angle
    p = paper.convert("RGBA").rotate(angle, expand=True, resample=Image.BICUBIC)
    p.thumbnail((W - 80, H - 80))
    shadow = Image.new("RGBA", p.size, (0, 0, 0, 90))
    base.paste(shadow, ((W - p.width) // 2 + 10, (H - p.height) // 2 + 12), p)
    base.paste(p, ((W - p.width) // 2, (H - p.height) // 2), p)
    return base.filter(ImageFilter.GaussianBlur(0.6))


def paper(w, h, color=(250, 248, 240)):
    return Image.new("RGB", (w, h), color)


def receipt(shop, sub, lines, total, date, foot="Thank you! Visit again"):
    img = paper(420, 140 + 34 * len(lines) + 200)
    d = ImageDraw.Draw(img)
    y = 25
    d.text((210, y), shop, font=font(24, True, True), fill=(20, 20, 20), anchor="mt"); y += 36
    d.text((210, y), sub, font=font(14, mono=True), fill=(60, 60, 60), anchor="mt"); y += 26
    d.text((20, y), f"Date: {date}", font=font(15, mono=True), fill=(40, 40, 40)); y += 26
    d.line([(20, y), (400, y)], fill=(90, 90, 90), width=1); y += 12
    for item, amt in lines:
        d.text((20, y), item[:26], font=font(16, mono=True), fill=(25, 25, 25))
        d.text((400, y), f"{amt:,.2f}", font=font(16, mono=True), fill=(25, 25, 25), anchor="ra"); y += 34
    d.line([(20, y), (400, y)], fill=(90, 90, 90), width=1); y += 14
    d.text((20, y), "TOTAL", font=font(19, True, True), fill=(0, 0, 0))
    d.text((400, y), f"Rs {total:,.2f}", font=font(19, True, True), fill=(0, 0, 0), anchor="ra"); y += 50
    d.text((210, y), foot, font=font(14, mono=True), fill=(70, 70, 70), anchor="mt")
    return img


def prescription(doctor, clinic, patient, date, meds, advice):
    img = paper(620, 820, (255, 255, 252))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 620, 110], fill=(225, 238, 250))
    d.text((30, 22), doctor, font=font(28, True), fill=(20, 50, 100))
    d.text((30, 62), f"MBBS, MD  ·  {clinic}", font=font(16), fill=(40, 70, 110))
    d.text((30, 135), f"Patient: {patient}", font=font(18), fill=(30, 30, 30))
    d.text((420, 135), f"Date: {date}", font=font(18), fill=(30, 30, 30))
    d.text((30, 185), "Rx", font=font(40, True), fill=(20, 50, 100))
    y = 250
    for m in meds:
        d.text((60, y), "• " + m, font=font(20), fill=(25, 25, 60)); y += 48
    y += 20
    d.text((30, y), "Advice: " + advice, font=font(17), fill=(50, 50, 50))
    d.text((420, 760), "Signature", font=font(15), fill=(120, 120, 120))
    d.line([(400, 750), (590, 750)], fill=(80, 80, 80), width=2)
    return img


def medicine_strip(name, strength, n=10):
    img = Image.new("RGB", (640, 300), (200, 204, 210))
    d = ImageDraw.Draw(img)
    for x in range(0, 640, 4):
        d.line([(x, 0), (x, 300)], fill=(190 + (x // 4) % 20, 194 + (x // 4) % 20, 200 + (x // 4) % 20))
    d.rectangle([0, 0, 640, 70], fill=(30, 120, 160))
    d.text((20, 15), f"{name} {strength}", font=font(30, True), fill=(255, 255, 255))
    for i in range(n):
        cx = 50 + (i % 5) * 120
        cy = 130 if i < 5 else 230
        d.ellipse([cx - 32, cy - 32, cx + 32, cy + 32], fill=(225, 228, 232), outline=(150, 150, 160), width=3)
        d.ellipse([cx - 20, cy - 20, cx + 20, cy + 20], fill=(250, 250, 250))
    d.text((620, 278), "Batch B2611  Exp 08/2027", font=font(12), fill=(60, 60, 60), anchor="rs")
    return table_photo(img, bg=(235, 235, 230))


def phone_frame(draw_fn, title, header=(18, 18, 18), bg=(245, 245, 245)):
    img = Image.new("RGB", (540, 1080), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 540, 40], fill=header)
    d.text((20, 10), "9:41", font=font(16, True), fill=(255, 255, 255))
    d.text((520, 10), "87%", font=font(16), fill=(255, 255, 255), anchor="ra")
    d.rectangle([0, 40, 540, 110], fill=header)
    d.text((70, 60), title, font=font(24, True), fill=(255, 255, 255))
    d.text((20, 58), "←", font=font(28, True), fill=(255, 255, 255))
    draw_fn(d)
    return img


def payment(amount, to, upi, date, note):
    def body(d):
        d.ellipse([220, 200, 320, 300], fill=(30, 150, 80))
        d.text((270, 250), "✓", font=font(56, True), fill=(255, 255, 255), anchor="mm")
        d.text((270, 350), "Payment successful", font=font(26, True), fill=(20, 20, 20), anchor="mm")
        d.text((270, 430), f"₹{amount:,}", font=font(56, True), fill=(10, 10, 10), anchor="mm")
        d.text((270, 510), f"Paid to {to}", font=font(22), fill=(40, 40, 40), anchor="mm")
        d.text((270, 545), upi, font=font(18), fill=(100, 100, 100), anchor="mm")
        d.rounded_rectangle([40, 620, 500, 820], radius=18, fill=(255, 255, 255), outline=(220, 220, 220))
        d.text((70, 645), f"Note: {note}", font=font(18), fill=(50, 50, 50))
        d.text((70, 690), date, font=font(18), fill=(50, 50, 50))
        d.text((70, 735), f"Txn ID {random.randint(10**11, 10**12 - 1)}", font=font(18), fill=(50, 50, 50))
    return phone_frame(body, "PayNow", header=(40, 30, 120))


def chat(contact, msgs, date):
    def body(d):
        d.text((270, 140), date, font=font(15), fill=(110, 110, 110), anchor="mm")
        y = 180
        for who, text, t in msgs:
            fnt = font(20)
            lines, line = [], ""
            for w in text.split():
                if d.textlength(line + " " + w, font=fnt) > 330:
                    lines.append(line.strip()); line = w
                else:
                    line += " " + w
            lines.append(line.strip())
            h = 30 * len(lines) + 34
            wmax = max(d.textlength(l, font=fnt) for l in lines) + 90
            if who == "me":
                x0, x1, col = 520 - wmax, 520, (215, 245, 200)
            else:
                x0, x1, col = 20, 20 + wmax, (255, 255, 255)
            d.rounded_rectangle([x0, y, x1, y + h], radius=14, fill=col)
            for i, l in enumerate(lines):
                d.text((x0 + 16, y + 10 + 30 * i), l, font=fnt, fill=(20, 20, 20))
            d.text((x1 - 12, y + h - 8), t, font=font(13), fill=(110, 110, 110), anchor="rs")
            y += h + 18
    return phone_frame(body, contact, header=(25, 95, 85), bg=(236, 229, 221))


def notes_board(title, lines, dark=False):
    bg = (35, 60, 45) if dark else (248, 248, 246)
    ink = (235, 235, 225) if dark else (30, 60, 150)
    img = Image.new("RGB", (900, 640), bg)
    d = ImageDraw.Draw(img)
    d.text((40, 30), title, font=font(36, True), fill=ink)
    y = 110
    for l in lines:
        d.text((60, y), l, font=font(26), fill=ink); y += 58
    img = img.rotate(random.uniform(-3, 3), expand=False, fillcolor=(90, 90, 95))
    return img.filter(ImageFilter.GaussianBlur(0.9))


def ticket(kind, lines, date, color=(200, 60, 40)):
    img = paper(680, 360, (255, 253, 245))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 680, 70], fill=color)
    d.text((24, 18), kind, font=font(28, True), fill=(255, 255, 255))
    y = 95
    for l in lines:
        d.text((24, y), l, font=font(21), fill=(30, 30, 30)); y += 40
    d.text((656, 330), date, font=font(18, True), fill=(30, 30, 30), anchor="rs")
    return img


def id_card():
    img = paper(640, 400, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 80], fill=(120, 20, 40))
    d.text((24, 20), "Westbrook Institute of Technology, Pune", font=font(22, True), fill=(255, 255, 255))
    d.rectangle([30, 110, 190, 300], fill=(210, 210, 215))
    d.ellipse([75, 130, 145, 200], fill=(170, 170, 178))
    d.rectangle([65, 210, 155, 290], fill=(170, 170, 178))
    for i, l in enumerate(["STUDENT ID", "Name: Riya Sharma", "Program: B.Tech (CSE)", "ID No: WIT/2023/0417",
                           "Valid till: Jun 2027", "Blood group: B+"]):
        d.text((220, 115 + i * 38), l, font=font(22 if i else 24, bold=(i == 0)), fill=(30, 30, 30))
    return img


# ---------------------------------------------------------------- the library
ITEMS = []


def add(fn, img, **meta):
    img.convert("RGB").save(OUT / f"{fn}.jpg", quality=86)
    ITEMS.append({"file": f"utility/{fn}.jpg", **meta})


def build():
    # --- Sick, Feb 2026 ---
    add("rx_feb26", table_photo(prescription("Dr. Meera Kulkarni", "Kothrud Family Clinic, Pune", "Riya Sharma",
                                             "11/02/2026", ["Azithral 500 — 1 tab daily × 3 days",
                                                            "Dolo 650 — SOS for fever", "Cetirizine 10 — at night × 5 days"],
                                             "Steam inhalation, warm fluids, rest")),
        date="2026-02-11T18:20", source="camera", moment="sick_feb26",
        caption="photo of a doctor's prescription paper on a table", ocr="Dr. Meera Kulkarni prescription Azithral Dolo 650 Cetirizine",
        purpose="medical record")
    add("strip_feb26", medicine_strip("AZITHRAL", "500 mg", 6), date="2026-02-11T19:02", source="camera", moment="sick_feb26",
        caption="close-up of a silver tablet strip", ocr="AZITHRAL 500 mg", purpose="medicine to remember")
    add("pharmacy_feb26", table_photo(receipt("CITY MEDICOS", "Karve Road, Pune", [("Azithral 500 (5)", 132.0),
                                                                                  ("Dolo 650 (15)", 33.6), ("Cetirizine (10)", 18.0)],
                                              183.6, "11-02-2026")),
        date="2026-02-11T19:05", source="camera", moment="sick_feb26",
        caption="photo of a printed shop receipt", ocr="CITY MEDICOS pharmacy bill Azithral Dolo", purpose="bill")
    add("lab_feb26", table_photo(receipt("PRIME DIAGNOSTICS", "Blood test report summary",
                                         [("Haemoglobin 12.9 g/dL", 0.0), ("WBC 11,200 /uL (H)", 0.0), ("Platelets 2.4 lakh", 0.0)],
                                         650.0, "12-02-2026", foot="Report verified")),
        date="2026-02-12T11:40", source="whatsapp", moment="sick_feb26",
        caption="photo of a printed document with a table of values", ocr="PRIME DIAGNOSTICS blood test report WBC", purpose="medical record")
    # older medicine strips — look-alikes from other times
    add("strip_aug25", medicine_strip("ALLEGRA", "120 mg", 10), date="2025-08-03T21:10", source="camera", moment="hostel_life",
        caption="close-up of a silver tablet strip", ocr="ALLEGRA 120 mg", purpose="medicine to remember")
    add("strip_nov24", medicine_strip("PAN-D", "40 mg", 10), date="2024-11-18T08:30", source="camera", moment="home",
        caption="close-up of a silver tablet strip", ocr="PAN-D 40 mg", purpose="medicine for mom")

    # --- Internship PG rent receipts (look-alikes) ---
    for mon, d, amt in [("Jun", "2026-06-05", 14500), ("Jul", "2026-07-05", 14500), ("Aug", "2026-08-04", 14500)]:
        add(f"rent_{mon.lower()}26", table_photo(receipt("SAI COMFORT PG", "HSR Layout, Bengaluru",
                                                          [(f"Rent - {mon} 2026", float(amt)), ("Electricity", 620.0)],
                                                          amt + 620.0, f"{d[8:]}-{d[5:7]}-2026", foot="Received with thanks")),
            date=f"{d}T20:15", source="camera", moment="internship_blr",
            caption="photo of a printed shop receipt", ocr=f"SAI COMFORT PG rent {mon} 2026 HSR Layout", purpose="rent proof")
    add("rent_pay_jun26", payment(15120, "Sai Comfort PG", "saicomfortpg@okbank", "05 Jun 2026, 8:11 PM", "June rent"),
        date="2026-06-05T20:11", source="screenshot", moment="internship_blr",
        caption="phone screenshot of a payment app", ocr="Payment successful 15,120 Sai Comfort PG June rent", purpose="rent proof")

    # --- Payments (look-alikes) ---
    pays = [(450, "Ravi Tea Stall", "Chai + maggi", "2025-09-14T17:40", "hostel_life"),
            (2300, "Rohan Mehta", "Goa trip share", "2025-12-21T22:05", "goa_trip"),
            (780, "Café Bodega", "Dinner", "2025-12-19T21:30", "goa_trip"),
            (1200, "Ananya Rao", "Fest tickets", "2025-02-08T13:12", "hostel_life"),
            (349, "Bookworm Store", "DSA book", "2025-08-21T16:02", "hostel_life"),
            (5600, "Westbrook Institute", "Hostel mess fee", "2025-07-28T11:20", "hostel_life")]
    for i, (amt, to, note, d, mom) in enumerate(pays):
        add(f"pay_{i}", payment(amt, to, to.lower().replace(' ', '').replace('é', 'e') + "@okbank",
                                d[:10] + ", " + d[11:], note),
            date=d, source="screenshot", moment=mom, caption="phone screenshot of a payment app",
            ocr=f"Payment successful {amt} {to} {note}", purpose="payment proof")

    # --- Chats ---
    add("wifi_chat", chat("Hostel Wing C", [("other", "wifi pwd for the new router is", "21:02"),
                                             ("other", "WingC@5G_2025", "21:02"), ("me", "thanksss", "21:05")], "Sep 2, 2025"),
        date="2025-09-02T21:05", source="screenshot", moment="hostel_life", caption="phone screenshot of a chat conversation",
        ocr="wifi pwd for the new router is WingC@5G_2025", purpose="save password")
    add("address_chat", chat("Ananya", [("me", "send the cafe location pls", "17:20"),
                                         ("other", "Bodega Café, Fontainhas, Panaji — near the yellow church", "17:22")], "Dec 19, 2025"),
        date="2025-12-19T17:22", source="screenshot", moment="goa_trip", caption="phone screenshot of a chat conversation",
        ocr="Bodega Café Fontainhas Panaji near the yellow church", purpose="save address")
    add("prof_chat", chat("DBMS Prof (TA)", [("other", "Assignment 3 deadline extended to 14 Oct, submit on portal", "10:15"),
                                              ("me", "Thank you sir", "10:31")], "Oct 9, 2025"),
        date="2025-10-09T10:31", source="screenshot", moment="hostel_life", caption="phone screenshot of a chat conversation",
        ocr="Assignment 3 deadline extended to 14 Oct", purpose="deadline")

    # --- Notes ---
    add("notes_dbms", notes_board("DBMS — Normalisation", ["1NF: atomic values", "2NF: no partial dependency",
                                                           "3NF: no transitive dependency", "BCNF: every determinant is a key"]),
        date="2025-09-23T11:05", source="camera", moment="hostel_life", caption="photo of a whiteboard with handwriting",
        ocr="DBMS Normalisation 1NF 2NF 3NF BCNF", purpose="class notes")
    add("notes_os", notes_board("OS — Deadlock conditions", ["Mutual exclusion", "Hold and wait", "No preemption",
                                                             "Circular wait"], dark=True),
        date="2025-10-07T09:40", source="camera", moment="hostel_life", caption="photo of a blackboard with chalk writing",
        ocr="OS Deadlock conditions mutual exclusion hold and wait", purpose="class notes")
    add("notes_cn", notes_board("CN — TCP 3-way handshake", ["SYN →", "← SYN-ACK", "ACK →", "then data transfer"]),
        date="2025-10-15T14:20", source="camera", moment="hostel_life", caption="photo of a whiteboard with handwriting",
        ocr="CN TCP 3-way handshake SYN ACK", purpose="class notes")

    # --- Tickets / documents ---
    add("train_ticket", table_photo(ticket("RAILWAY E-TICKET", ["PNR 482-1937650   Train 12779",
                                                                 "Pune Jn (PUNE) → Madgaon (MAO)", "Coach B2  Berth 34 (LB)",
                                                                 "Passenger: Riya Sharma, F, 20"], "18 Dec 2025")),
        date="2025-12-10T22:40", source="download", moment="goa_trip", caption="photo of a printed ticket",
        ocr="PNR train Pune Madgaon Riya Sharma 18 Dec 2025", purpose="travel ticket")
    add("flight_ticket", ticket("SKYLARK AIR — BOARDING PASS", ["Flight SK 517   Seat 14C", "PNQ Pune → BLR Bengaluru",
                                                                  "Boarding 06:10   Gate 4", "Passenger: SHARMA/RIYA"],
                                "01 Jun 2026", color=(20, 90, 170)),
        date="2026-05-30T23:10", source="download", moment="internship_blr", caption="image of a boarding pass",
        ocr="Boarding pass PNQ BLR flight SK 517 01 Jun 2026", purpose="travel ticket")
    add("movie_ticket", ticket("CINEMAX — M-TICKET", ["Interstellar (Re-release)  IMAX", "Screen 3   Row H  Seats 11,12",
                                                       "Phoenix Mall, Pune"], "26 Jan 2026", color=(150, 30, 120)),
        date="2026-01-24T19:00", source="screenshot", moment="hostel_life", caption="phone screenshot of a ticket",
        ocr="CINEMAX Interstellar IMAX Phoenix Mall", purpose="ticket")
    add("college_id", table_photo(id_card()), date="2024-08-05T10:00", source="camera", moment="hostel_life",
        caption="photo of an identity card", ocr="Westbrook Institute of Technology Student ID Riya Sharma", purpose="ID proof")
    add("hostel_fee", table_photo(receipt("WESTBROOK INSTITUTE", "Hostel fee receipt", [("Hostel fee Sem 5", 48000.0),
                                                                                        ("Mess advance", 22000.0)],
                                          70000.0, "28-07-2025", foot="Accounts section")),
        date="2025-07-28T12:00", source="camera", moment="hostel_life", caption="photo of a printed shop receipt",
        ocr="WESTBROOK INSTITUTE Hostel fee receipt Sem 5", purpose="fee proof")
    add("cafe_bill_goa", table_photo(receipt("BODEGA CAFE", "Fontainhas, Panaji", [("Cold coffee", 220.0), ("Bebinca", 180.0),
                                                                                  ("Prawn rissois", 320.0)], 756.0, "19-12-2025")),
        date="2025-12-19T21:25", source="camera", moment="goa_trip", caption="photo of a printed shop receipt",
        ocr="BODEGA CAFE Fontainhas Panaji Bebinca", purpose="bill")

    (OUT.parent / "utility.json").write_text(json.dumps(ITEMS, indent=1, ensure_ascii=False))
    print(len(ITEMS), "utility images")


if __name__ == "__main__":
    build()
