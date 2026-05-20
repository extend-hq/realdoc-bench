#!/usr/bin/env python3
"""CMS-1500 specific filler. Uses a detailed box-by-box prompt for gpt-image-2
so the service-line TABLE (Box 24) actually gets populated, plus all the
patient/insured/provider boxes.

This is needed because the generic polish prompt doesn't know about CMS-1500's
peculiar layout (Box 21 ICD-10 grid, Box 24 6-row service-line table with 11
columns, Box 25/27/28 footer values).

For each source CMS-1500 blank × each persona:
  - render page as PNG
  - send to gpt-image-2 with CMS-specific box-mapping prompt
  - replace source page with returned PNG

CLI:
  --doc <fname>      restrict to one specific blank
  --persona N        restrict to one persona index
  --workers N        parallel workers (default 8)
  --dry-run          show plan
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
PERSONAS_DIR = DATASETS / "_personas"
CMS_FOLDER = DATASETS / "v1_cms_1500"


def load_dotenv(path: Path = REPO / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k and v and not os.environ.get(k):
            os.environ[k] = v
load_dotenv()


_oai = None
def get_openai_client():
    global _oai
    if _oai is None:
        from openai import OpenAI
        _oai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _oai


CMS_PROMPT_TEMPLATE = """TASK: Fill out the CMS-1500 form in the input image with the values below. Type values into the blank fields. The output must be VISUALLY IDENTICAL to the input form except that the empty fields now contain the values.

===== FORM PRESERVATION — ABSOLUTE RULES (READ BEFORE FILLING) =====
- The input is the modern CMS-1500 (02/12) form (OMB-0938-1197) — the RED/PINK printed-ink version with 33 boxes, NUCC-approved, with checkbox-style insurance type selector in Box 1 and a 6-row × 11-column service-line table in Box 24.
- DO NOT generate the older HCFA-1500 (12-90), HCFA-1500 (08-05), or any other version of the form.
- DO NOT redesign the form layout, change the printed wording of any label, alter the form's red/pink ink color, change box numbering, change column widths, or replace the official form with a "modernized" or "clean" version.
- Reproduce the form's exact original layout including: the QR code top-left, the "HEALTH INSURANCE CLAIM FORM" title, the "APPROVED BY NATIONAL UNIFORM CLAIM COMMITTEE (NUCC) 02/12" subtitle, all printed sub-labels in Box 1 (Medicare/Medicaid/TRICARE/CHAMPVA/GROUP HEALTH PLAN/FECA BLK LUNG/OTHER), the "READ BACK OF FORM BEFORE COMPLETING & SIGNING" boilerplate, all column headers in Box 24 (DATE(S) OF SERVICE / PLACE OF SERVICE / EMG / PROCEDURES, SERVICES, OR SUPPLIES / DIAGNOSIS POINTER / $ CHARGES / DAYS OR UNITS / EPSDT Family Plan / ID. QUAL. / RENDERING PROVIDER ID. #), and the "NUCC Instruction Manual available at: www.nucc.org" / "PLEASE PRINT OR TYPE" / "APPROVED OMB-0938-1197 FORM 1500 (02-12)" footer.
- Only ADD typed values + handwritten signature where blank fillable areas exist. Do not alter any other pixel.

Persona: {persona_label}

Persona: {persona_label}

===== BOX-BY-BOX MAPPING =====
Box 1 (insurance type — check exactly ONE): {ins_type}
Box 1a (INSURED'S I.D. NUMBER): {insured_id_number}
Box 2 (PATIENT'S NAME): {patient_last_name}, {patient_first_name} {patient_middle_initial}
Box 3 (PATIENT'S BIRTH DATE / SEX): MM={pdob_mm} DD={pdob_dd} YY={pdob_yy} | sex check: {patient_sex}
Box 4 (INSURED'S NAME): {insured_last_name}, {insured_first_name} {insured_middle_initial}
Box 5 (PATIENT'S ADDRESS): {patient_street_address} | {patient_city}, {patient_state} {patient_zip} | TEL ({patient_phone_area}) {patient_phone_main}
Box 6 (PATIENT RELATIONSHIP TO INSURED — check ONE): {relationship}
Box 7 (INSURED'S ADDRESS): {insured_street_address} | {insured_city}, {insured_state} {insured_zip} | TEL ({insured_phone_area}) {insured_phone_main}
Box 8 — RESERVED FOR NUCC USE — leave blank
Box 9 (OTHER INSURED'S NAME — only if other_insured_exists=true): {other_insured}
Box 10 (PATIENT'S CONDITION RELATED TO):
   a. Employment (Current or Previous)? — check {emp_yn}
   b. Auto Accident? — check {auto_yn} (and State if YES)
   c. Other Accident? — check {other_yn}
Box 11 (INSURED'S POLICY GROUP OR FECA NUMBER): {insured_policy_group_number}
   11a Insured's DOB / SEX: MM={idob_mm} DD={idob_dd} YY={idob_yy} | sex {insured_sex}
   11b Employer/School Name: {insured_employer_name}
   11c Insurance Plan Name or Program Name: {insured_insurance_plan_name}
   11d Is there another health benefit plan? — check {other_plan_yn}
Box 14 (DATE OF CURRENT ILLNESS/INJURY/PREGNANCY): {date_current_illness}
Box 16 (DATES PATIENT UNABLE TO WORK): FROM {date_unable_from} TO {date_unable_to}
Box 17 (NAME OF REFERRING PROVIDER): {referring_provider_name}
   17b NPI: {referring_provider_npi}
Box 18 (HOSPITALIZATION DATES): FROM {hosp_from} TO {hosp_to}
Box 20 (OUTSIDE LAB?): check {lab_yn}, $ CHARGES: {lab_amount}

Box 21 — DIAGNOSIS OR NATURE OF ILLNESS — fill EACH lettered slot A-L with the corresponding ICD-10 code below. Leave a slot empty if no code:
{diagnosis_table}

Box 24 — SERVICE LINE TABLE — fill row N of the 6-row table with the values below. The table columns from left to right are:
  A. DATE(S) OF SERVICE: From MM|DD|YY  To MM|DD|YY
  B. PLACE OF SERVICE (2-digit code, e.g. 11 = office)
  C. EMG (check if yes)
  D. PROCEDURES, SERVICES, OR SUPPLIES: CPT/HCPCS code | Modifier 1, 2, 3, 4
  E. DIAGNOSIS POINTER (letters from Box 21, e.g. "AB" or "ABCD")
  F. $ CHARGES
  G. DAYS OR UNITS
  H. EPSDT/Family Plan (check)
  I. ID. QUAL.
  J. RENDERING PROVIDER ID. # / NPI (write in the NPI row beneath)

Fill ONLY the rows for which a service_line_N entry exists below — leave the other rows empty:
{service_line_table}

Box 25 (FEDERAL TAX I.D. NUMBER): {federal_tax_id_number}  | check {tax_id_type} (SSN or EIN)
Box 26 (PATIENT'S ACCOUNT NO.): {patient_account_no}
Box 27 (ACCEPT ASSIGNMENT?): check {accept_assignment}
Box 28 (TOTAL CHARGE): $ {total_charges}
Box 29 (AMOUNT PAID): $ {amount_paid}
Box 30 (BALANCE DUE / Reserved): leave blank or $ {balance_due}
Box 31 (SIGNATURE OF PHYSICIAN OR SUPPLIER): handwritten cursive "{physician_name}" | DATE: {signature_date}
Box 32 (SERVICE FACILITY LOCATION): {service_facility}
   NPI: {service_facility_npi}
Box 33 (BILLING PROVIDER INFO & PH#): {billing_provider}
   NPI: {billing_provider_npi}

===== FORMATTING & STYLE =====
- Type values in a typewriter monospaced or simple sans-serif font, BLACK ink.
- For Box 31 signature: realistic blue-ink CURSIVE handwriting, looks hand-signed.
- For checkboxes: hand-drawn X or ✓ marks (small, in pen).
- For dates: write MM | DD | YY using the dividing pipes that are printed on the form.
- For dollar amounts: write the number without a $ sign in the $ CHARGES column.
- Box 24 service lines: align horizontally within each table row. Date in A, place in B, CPT in D, diagnosis pointer in E, charges in F, units in G, NPI in row beneath J.
- DO NOT redesign the form layout. Preserve the original red CMS-1500 form exactly.
- DO NOT add any text outside the printed form boxes.
- DO NOT check every checkbox — only check what the persona explicitly indicates.
"""


def format_persona_for_cms(persona: dict) -> dict[str, str]:
    """Convert persona fields dict into named parts for the CMS prompt template."""
    f = persona.get("fields", persona)

    def _date_parts(date_str: str) -> tuple[str, str, str]:
        if not date_str or not isinstance(date_str, str):
            return ("", "", "")
        m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_str)
        if not m:
            return ("", "", "")
        mm, dd, yyyy = m.group(1), m.group(2), m.group(3)
        yy = yyyy[-2:] if len(yyyy) >= 2 else yyyy
        return (mm.zfill(2), dd.zfill(2), yy.zfill(2))

    def _phone_parts(phone: str) -> tuple[str, str]:
        if not phone:
            return ("", "")
        m = re.match(r"\(?(\d{3})\)?[\s-]?(\d{3}[\s-]?\d{4})", phone)
        if not m:
            return ("", "")
        return (m.group(1), m.group(2))

    pdob = _date_parts(f.get("patient_date_of_birth", ""))
    idob = _date_parts(f.get("insured_date_of_birth", ""))
    p_area, p_main = _phone_parts(f.get("patient_phone", ""))
    i_area, i_main = _phone_parts(f.get("insured_phone", ""))

    # Insurance type — first one true
    ins_types = [
        ("medicare_checked", "MEDICARE"),
        ("medicaid_checked", "MEDICAID"),
        ("tricare_checked", "TRICARE"),
        ("champva_checked", "CHAMPVA"),
        ("group_health_plan_checked", "GROUP HEALTH PLAN"),
        ("feca_blklung_checked", "FECA BLK LUNG"),
        ("other_checked", "OTHER"),
    ]
    ins_type = "(check one applicable)"
    for k, label in ins_types:
        if f.get(k) is True:
            ins_type = label
            break
    if ins_type == "(check one applicable)":
        ins_type = "OTHER"

    # Sex
    p_sex = "M" if f.get("patient_sex_male_checked") else ("F" if f.get("patient_sex_female_checked") else "")
    i_sex = "M" if f.get("insured_sex_male_checked") else ("F" if f.get("insured_sex_female_checked") else "")

    # Relationship
    rel = ""
    for k, lbl in [("patient_relationship_to_insured_self_checked", "Self"),
                   ("patient_relationship_to_insured_spouse_checked", "Spouse"),
                   ("patient_relationship_to_insured_child_checked", "Child"),
                   ("patient_relationship_to_insured_other_checked", "Other")]:
        if f.get(k) is True:
            rel = lbl
            break
    if not rel:
        # default heuristic
        if f.get("patient_first_name") == f.get("insured_first_name"):
            rel = "Self"

    # Yes/No helpers
    def _yn(true_key: str, false_key: str) -> str:
        if f.get(true_key) is True: return "YES"
        if f.get(false_key) is True: return "NO"
        return ""

    emp_yn = _yn("patient_condition_related_to_employment_yes_checked",
                 "patient_condition_related_to_employment_no_checked")
    auto_yn = _yn("patient_condition_related_to_auto_accident_yes_checked",
                  "patient_condition_related_to_auto_accident_no_checked")
    other_yn = _yn("patient_condition_related_to_other_accident_yes_checked",
                   "patient_condition_related_to_other_accident_no_checked")
    other_plan_yn = _yn("other_health_benefit_plan_yes_checked",
                        "other_health_benefit_plan_no_checked")
    lab_yn = _yn("outside_lab_charges_yes_checked", "outside_lab_charges_no_checked")

    # Diagnosis code table (Box 21)
    dx_lines = []
    for letter in "ABCDEFGHIJKL":
        v = f.get(f"diagnosis_code_{letter.lower()}", "")
        if v:
            dx_lines.append(f"  {letter}. {v}")
    diagnosis_table = "\n".join(dx_lines) if dx_lines else "  (no diagnosis codes — leave Box 21 empty)"

    # Service line table (Box 24) — only emit rows that have data
    sl_lines = []
    for n in range(1, 7):
        cpt = f.get(f"service_line_{n}_cpt_code", "")
        if not cpt:
            continue
        df_mm, df_dd, df_yy = _date_parts(f.get(f"service_line_{n}_date_from", ""))
        dt_mm, dt_dd, dt_yy = _date_parts(f.get(f"service_line_{n}_date_to", ""))
        pos = f.get(f"service_line_{n}_place_of_service", "")
        emg = "X" if f.get(f"service_line_{n}_emg_checked") else ""
        mods = "/".join(filter(None, [f.get(f"service_line_{n}_modifier_{i}", "") for i in range(1, 5)]))
        dx_ptr = f.get(f"service_line_{n}_diagnosis_pointer", "")
        chg = f.get(f"service_line_{n}_charges", "")
        units = f.get(f"service_line_{n}_days_units", "")
        epsdt = "X" if f.get(f"service_line_{n}_epsdt_family_plan_checked") else ""
        npi = f.get(f"service_line_{n}_rendering_provider_npi", "")
        row = (f"  ROW {n}: A={df_mm}/{df_dd}/{df_yy} to {dt_mm}/{dt_dd}/{dt_yy} | "
               f"B={pos} | C(EMG)={emg} | "
               f"D=CPT:{cpt} | Mods:{mods or '-'} | "
               f"E=ptr {dx_ptr} | F=${chg} | G={units}u | H(EPSDT)={epsdt} | "
               f"J/NPI={npi}")
        sl_lines.append(row)
    service_line_table = "\n".join(sl_lines) if sl_lines else "  (no service lines — leave Box 24 empty)"

    # Tax ID type
    tax_id_type = "EIN" if f.get("federal_tax_id_ein_checked") else (
        "SSN" if f.get("federal_tax_id_ssn_checked") else "EIN")
    accept_assignment = "YES" if f.get("accept_assignment_yes_checked") else (
        "NO" if f.get("accept_assignment_no_checked") else "YES")

    return {
        "persona_label": persona.get("persona_label", "synthetic persona"),
        "ins_type": ins_type,
        "insured_id_number": str(f.get("insured_id_number", "") or ""),
        "patient_last_name": str(f.get("patient_last_name", "") or ""),
        "patient_first_name": str(f.get("patient_first_name", "") or ""),
        "patient_middle_initial": str(f.get("patient_middle_initial", "") or ""),
        "pdob_mm": pdob[0], "pdob_dd": pdob[1], "pdob_yy": pdob[2],
        "patient_sex": p_sex,
        "insured_last_name": str(f.get("insured_last_name", "") or ""),
        "insured_first_name": str(f.get("insured_first_name", "") or ""),
        "insured_middle_initial": str(f.get("insured_middle_initial", "") or ""),
        "patient_street_address": str(f.get("patient_street_address", "") or ""),
        "patient_city": str(f.get("patient_city", "") or ""),
        "patient_state": str(f.get("patient_state", "") or ""),
        "patient_zip": str(f.get("patient_zip", "") or ""),
        "patient_phone_area": p_area, "patient_phone_main": p_main,
        "relationship": rel or "Self",
        "insured_street_address": str(f.get("insured_street_address", "") or ""),
        "insured_city": str(f.get("insured_city", "") or ""),
        "insured_state": str(f.get("insured_state", "") or ""),
        "insured_zip": str(f.get("insured_zip", "") or ""),
        "insured_phone_area": i_area, "insured_phone_main": i_main,
        "other_insured": str(f.get("other_insured_name", "") or "(leave blank if not applicable)"),
        "emp_yn": emp_yn or "NO", "auto_yn": auto_yn or "NO", "other_yn": other_yn or "NO",
        "insured_policy_group_number": str(f.get("insured_policy_group_number", "") or ""),
        "idob_mm": idob[0], "idob_dd": idob[1], "idob_yy": idob[2],
        "insured_sex": i_sex,
        "insured_employer_name": str(f.get("insured_employer_name", "") or ""),
        "insured_insurance_plan_name": str(f.get("insured_insurance_plan_name", "") or ""),
        "other_plan_yn": other_plan_yn or "NO",
        "date_current_illness": str(f.get("date_current_illness_injury_pregnancy", "") or ""),
        "date_unable_from": str(f.get("date_patient_unable_to_work_from", "") or ""),
        "date_unable_to": str(f.get("date_patient_unable_to_work_to", "") or ""),
        "referring_provider_name": str(f.get("referring_provider_name", "") or ""),
        "referring_provider_npi": str(f.get("referring_provider_npi", "") or ""),
        "hosp_from": str(f.get("hospitalization_date_from", "") or ""),
        "hosp_to": str(f.get("hospitalization_date_to", "") or ""),
        "lab_yn": lab_yn or "NO",
        "lab_amount": str(f.get("outside_lab_charges_amount", "") or ""),
        "diagnosis_table": diagnosis_table,
        "service_line_table": service_line_table,
        "federal_tax_id_number": str(f.get("federal_tax_id_number", "") or ""),
        "tax_id_type": tax_id_type,
        "patient_account_no": str(f.get("patient_account_no", "") or ""),
        "accept_assignment": accept_assignment,
        "total_charges": str(f.get("total_charges", "") or ""),
        "amount_paid": str(f.get("amount_paid", "") or ""),
        "balance_due": str(f.get("balance_due", "") or ""),
        "physician_name": str(f.get("physician_name", "") or f.get("billing_provider_name", "") or "Dr. Smith"),
        "signature_date": str(f.get("signature_date", "") or f.get("date_current_illness_injury_pregnancy", "") or ""),
        "service_facility": str(f.get("service_facility_name", "") or ""),
        "service_facility_npi": str(f.get("service_facility_npi", "") or ""),
        "billing_provider": str(f.get("billing_provider_name", "") or ""),
        "billing_provider_npi": str(f.get("billing_provider_npi", "") or ""),
    }


def render_page_png(pdf: Path, dpi: int = 144) -> bytes:
    with fitz.open(pdf) as doc:
        page = doc[0]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")


def gpt_image_edit(page_png: bytes, persona: dict, model: str = "gpt-image-2") -> bytes | None:
    parts = format_persona_for_cms(persona)
    prompt = CMS_PROMPT_TEMPLATE.format(**parts)
    client = get_openai_client()
    try:
        resp = client.images.edit(
            model=model,
            image=("page.png", io.BytesIO(page_png), "image/png"),
            prompt=prompt,
            size="1024x1536",
            n=1,
        )
        return base64.b64decode(resp.data[0].b64_json)
    except Exception as e:
        print(f"   [image-gen err] {e}", flush=True)
        return None


def fill_one(src_pdf: Path, persona: dict, out_pdf: Path, model: str) -> bool:
    png = render_page_png(src_pdf)
    filled = gpt_image_edit(png, persona, model)
    if filled is None:
        return False
    new_doc = fitz.open()
    with fitz.open(src_pdf) as src:
        rect = src[0].rect
    page = new_doc.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, stream=filled, keep_proportion=False)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    new_doc.save(out_pdf, garbage=4, deflate=True)
    new_doc.close()
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc")
    ap.add_argument("--persona", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-suffix", default="_cmsfill")
    ap.add_argument("--image-model", default=os.environ.get("IMAGE_GEN_MODEL", "gpt-image-2"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Find blank CMS sources
    sources_dir = CMS_FOLDER / "sources"
    pdfs = sorted(sources_dir.glob("*.pdf"))
    if args.doc:
        pdfs = [p for p in pdfs if p.name == args.doc]

    personas = json.loads((PERSONAS_DIR / "v1_cms_1500.json").read_text())
    if args.persona is not None:
        personas = [personas[args.persona]]

    work = [(pdf, pi, persona) for pdf in pdfs for pi, persona in enumerate(personas)]
    print(f"Plan: {len(work)} CMS-1500 fills ({len(pdfs)} blanks × {len(personas)} personas)")
    print(f"  Workers: {args.workers}, model: {args.image_model}")
    if args.dry_run:
        for pdf, pi, p in work[:10]:
            print(f"  {pdf.name} × p{pi:02d} ({p.get('persona_label','?')})")
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    t0 = time.time()
    n_done = n_fail = 0
    def _job(pdf, pi, persona):
        out = CMS_FOLDER / "synthetic" / f"{pdf.stem}__p{pi:02d}{args.out_suffix}.pdf"
        ok = fill_one(pdf, persona, out, args.image_model)
        return (pdf, pi, persona, out, ok)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_job, p, pi, persona) for p, pi, persona in work]
        for i, fut in enumerate(as_completed(futures), 1):
            pdf, pi, persona, out, ok = fut.result()
            if ok:
                n_done += 1
                sz = out.stat().st_size // 1024
                print(f"  [{i}/{len(work)}] ✓ {out.name} ({sz}KB, {persona.get('persona_label','?')})", flush=True)
            else:
                n_fail += 1
                print(f"  [{i}/{len(work)}] ✗ {pdf.name} × p{pi:02d}", flush=True)
    print(f"\nDone — {n_done}/{len(work)} (fail {n_fail}) in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
