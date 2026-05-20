#!/usr/bin/env python3
"""Build a side-by-side HTML review of source-vs-downloaded for each v1_* doc type.

Layout:
    datasets/extraction/v1_review/
        INDEX.html                       — domain-grouped landing page
        <doc_type>.html                  — one per v1_* type
            LEFT  = source (customer PDFs/JSON or search description)
            RIGHT = downloaded v1_<type>/sources/*.pdf

Thumbnails are inlined as base64 PNGs (no external deps in the HTML).

The whole output dir is gitignored so it stays local.
"""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import fitz  # pymupdf

REPO = Path(__file__).resolve().parent.parent
DATASETS = REPO / "datasets" / "extraction"
CUSTOMER = DATASETS / "v1-customer"
REVIEW = DATASETS / "v1_review"


# ---------------------------------------------------------------------------
# Domain + per-type metadata
# ---------------------------------------------------------------------------

DOMAINS = {
    "Medical": [
        "v1_med_autopsy", "v1_med_protocol",
        "v1_clinical_consent", "v1_clinical_crf", "v1_clinical_facesheet",
        "v1_cms_1500",
    ],
    "Financial (incl. shipping/cargo)": [
        "v1_sec_10k", "v1_audit_report",
        "v1_acord_25", "v1_acord_125_126", "v1_acord_4",
        "v1_bol_inland", "v1_commercial_invoice", "v1_packing_list", "v1_pod",
    ],
    "Mortgage & Legal": [
        "v1_fnma_710", "v1_urar", "v1_cd", "v1_real_estate_contract",
    ],
    "Other (technical)": [
        "v1_tech_datasheet",
    ],
    "Budget & nonprofit financials": [
        "v1_budget_cafr", "v1_budget_sf424a", "v1_budget_amendment", "v1_budget_990",
    ],
}
# Extend with new sub-types
DOMAINS["Medical"].extend(["v1_vaers", "v1_ub04"])
DOMAINS["Financial (incl. shipping/cargo)"].extend([
    "v1_eu_sad", "v1_air_waybill", "v1_cmr", "v1_phyto_cert", "v1_loc",
    "v1_sec_10q", "v1_lloyds_mrc", "v1_esg",
])
DOMAINS["Mortgage & Legal"].extend(["v1_urla_1003", "v1_hud1"])
# Tax domain wasn't in original build_v1_review DOMAINS — add it
DOMAINS["Tax"] = ["v1_w2_1099"]
DOMAINS["Financial (incl. shipping/cargo)"].extend([
    "v1_bankruptcy_schedule", "v1_form_5500", "v1_fec_form3",
])

# Customer source mapping: which raw v1-customer/ files informed each type
CUSTOMER_MAPPING: dict[str, list[str]] = {
    "v1_fnma_710": [
        "MAFI983a7af809229ca45b2a1bec30a385_0_BRP_FORM-6NGSANcd-FK7YekP8.pdf",
        "MAFIa37494f8ac6b538fc8cebdcee3bd47_1_BRP_FORM-GgXdKSbl.pdf",
        "MAFIbc90eae4e731a5f6c40c8d96b1d741_0_BRP_FORM-TuKVedrI.pdf",
        "BRP%20Application%20Completeness%20Extractor%20vRY%20-%20JSON%20Schema_draft.json",
    ],
    "v1_urar": [
        "appraisal-test-checkbox-APnkSVaz.pdf",
        "appraisal-extractor_draft.json",
    ],
    "v1_real_estate_contract": [
        "Contract - MI Resale.pdf",
        "Contract - MN Acquisition.pdf",
        "Contract - MN Resale.pdf",
        "Contract - NV Acquisition.pdf",
        "Contract - NV Resale (one seller deceased).pdf",
        "Contract - OR Acquisition 2.pdf",
        "Contract - OR Acquisition.pdf",
        "Contract - OR Resale (1).pdf",
        "Contract - OR Resale.pdf",
        "Contract - TN Acquisition 2 (1).pdf",
        "Contract - TN Acquisition 2.pdf",
        "Real%20Estate%20Contract_draft.json",
    ],
    "v1_cd": [
        "CD_Processor_draft.json",
    ],
    "v1_bol_inland": [
        "master%20bill%20of%20lading_draft (2).json",
    ],
}

# Search context for types with no customer source
SEARCH_CONTEXT: dict[str, str] = {
    "v1_acord_25": (
        "User-provided image of an ACORD 25 Certificate of Liability Insurance. "
        "Search target: state govt + contractor sites for filled COI examples "
        "(state DOTs, county risk-mgmt offices). "
        "Schema design: COI header, producer block, insurer table, coverage grid, certificate holder, signatures."
    ),
    "v1_acord_125_126": (
        "User ask: more ACORD forms. Pulled ACORD 125 (Commercial Insurance Applicant Info) + "
        "ACORD 126 (Commercial General Liability Section), the two most-common commercial app forms. "
        "Filled samples are scarce — broker sites publish only blanks."
    ),
    "v1_acord_4": (
        "Interpreted as ACORD homeowner-line application family + Property Loss Notice "
        "(no current public form numbered '4' in the modern index; the closest carrier-bundled "
        "filename used '-Homeowners-4.pdf'). All public samples are blanks."
    ),
    "v1_bol_inland": (
        "Customer source: master BoL JSON schema only (no customer PDFs). "
        "Search target: FMCSA + CBP truck-manifest BOL templates + 3 worked exemplars from examples.com archive."
    ),
    "v1_commercial_invoice": (
        "Carrier-published templates (FedEx, DHL, US customs). No filled samples public — "
        "every shipper template is blank."
    ),
    "v1_packing_list": (
        "Carrier-published templates (FedEx UPL, DHL, ExFreight). Blank only."
    ),
    "v1_pod": (
        "Real signed PODs are emailed photos/scans in practice, not archived. "
        "Best public exemplars: IL court statewide PoD form + 1 SCOTUS docket POD cover letter (signed)."
    ),
    "v1_sec_10k": (
        "v0 had 1 hand-curated 10-K (fin_10k_001). Pulled Berkshire Hathaway annual reports 2020-2023 + "
        "JPMorgan Chase 2023 from investor-relations sites (sec.gov/Apple/Microsoft block scripted curls)."
    ),
    "v1_audit_report": (
        "v0 had 2 hand-curated audit reports. Pulled PCAOB inspection spotlight reports (2023/2024) + "
        "Floyd Advisory PCAOB-2022 review + DOJ OIG report 23-085 (BOP oversight audit)."
    ),
    "v1_med_autopsy": (
        "v0 had 1 hand-curated autopsy (med_autopsy_001). Pulled 5 publicly-released real autopsies: "
        "George Floyd (Hennepin Co ME 2020-3700, full report + court exhibit + press release), "
        "LA County 2020-05405 (redacted), William DeBose (case 20-1949)."
    ),
    "v1_med_protocol": (
        "v0 had 1 hand-curated protocol (med_protocol_001). Pulled 5 Phase-III protocols from "
        "ClinicalTrials.gov (NCT02312206 NEOD001, NCT02741245, NCT03658811, NCT05084053 TAK-771, NCT01891643)."
    ),
    "v1_tech_datasheet": (
        "v0 had 1 hand-curated datasheet (tech_datasheet_001). Pulled TI MSP430G2553, TI LM741 op-amp, "
        "Microchip PIC18F2455. STMicro/OnSemi blocked scripted curls."
    ),
    "v1_cms_1500": (
        "User ask. Pulled CMS official blank + AR DHS sample + 6 carrier/state-Medicaid blanks/guides + "
        "Monoferric annotated worked example (J1437 HCPCS, CPT 96365, ICD-10 D50.0, 100 billable units) — "
        "the only real filled CMS 1500 publicly available."
    ),
    "v1_clinical_consent": (
        "User ask. Pulled 4 real ICFs from ClinicalTrials.gov (NIH posting rules require sponsors to publish "
        "after recruitment closes) + FDA IRB consent guidance."
    ),
    "v1_clinical_crf": (
        "User ask. Pulled the real Swedish Match SM 08-01 / Covance 7694-105 CRF filed with FDA "
        "(76pp, blank because subject IDs are empty in the filing) + 4 CRF design training PDFs from NIH/NU/BU/OHSU."
    ),
    "v1_clinical_facesheet": (
        "User ask. Pulled 5 government-published intake/demographic forms (AHRQ HCUP, CDC URDO, "
        "NHLBI BioLINCC, Boston Advanced, TX DSHS advance directive). All blanks — PHI-heavy, "
        "real facesheets never published."
    ),
}


# PII / cleanliness status per type. Values:
#   "PUBLIC"     — released by government / FDA-filed / posted on CT.gov / SEC etc.
#   "BLANK"      — no real PII because docs are blank templates
#   "SYNTHETIC"  — filled but with synthetic/hypothetical data (CFPB Michael Jones samples, training scenarios)
#   "MIXED"      — combination
PII_STATUS: dict[str, tuple[str, str]] = {
    "v1_acord_25":             ("MIXED", "Worked agency examples with placeholder names (Sample Insured Inc, Bill Jones agent). No real customer PII."),
    "v1_acord_125_126":        ("BLANK", "All blanks."),
    "v1_acord_4":              ("BLANK", "All blanks."),
    "v1_bol_inland":           ("SYNTHETIC", "CBP + ArcBest templates + 3 examples.com worked filled samples."),
    "v1_commercial_invoice":   ("BLANK", "Carrier templates only."),
    "v1_packing_list":         ("BLANK", "Carrier templates only."),
    "v1_pod":                  ("PUBLIC", "IL court statewide form + SCOTUS docket POD letter (publicly filed)."),
    "v1_sec_10k":              ("PUBLIC", "SEC-filed public corp reports."),
    "v1_audit_report":         ("PUBLIC", "PCAOB / DOJ OIG public reports."),
    "v1_med_autopsy":          ("PUBLIC", "Publicly released medical-examiner reports (Floyd case is fully public; LACo case 2020-05405 is officially-redacted; DeBose case is press-released)."),
    "v1_med_protocol":         ("PUBLIC", "ClinicalTrials.gov-posted Phase III protocols."),
    "v1_clinical_consent":     ("PUBLIC", "ClinicalTrials.gov NIH-posted ICFs — NIH posting rules require publication after recruitment closes."),
    "v1_clinical_crf":         ("MIXED",  "FDA-filed Covance 7694-105 CRF (subject IDs blank in the filing itself) + 4 training PDFs."),
    "v1_clinical_facesheet":   ("BLANK",  "Government intake form templates only — no real patient PHI."),
    "v1_cms_1500":             ("SYNTHETIC", "Blanks + Monoferric worked example with synthetic billing values."),
    "v1_fnma_710":             ("BLANK",  "All blanks. No real borrower PII."),
    "v1_urar":                 ("SYNTHETIC", "RealVals 1004 training sample has addresses/names redacted; Freddie UAD 3.6 scenarios are synthetic."),
    "v1_cd":                   ("SYNTHETIC", "CFPB H-25 worked examples (Michael & Mary Jones, Anywhere Avenue, Money Bags Bank — all hypothetical)."),
    "v1_real_estate_contract": ("BLANK",  "State realtor/bar association templates."),
    "v1_tech_datasheet":       ("PUBLIC", "Vendor-published product datasheets."),
    "v1_budget_cafr":          ("PUBLIC", "City/county Annual Comprehensive Financial Reports — public by statute."),
    "v1_budget_sf424a":        ("MIXED",  "Mix of blank federal grant budget form + filled training examples."),
    "v1_budget_amendment":     ("PUBLIC", "Legislative/council-filed budget amendments — public records."),
    "v1_budget_990":           ("PUBLIC", "IRS nonprofit returns — public per IRC § 6104."),
    "v1_vaers":                ("PUBLIC", "VAERS forms — designed to be publicly filed, no PII concerns."),
    "v1_ub04":                 ("MIXED",  "Mix of blank UB-04 forms + Takeda worked-example with synthetic billing values."),
    "v1_eu_sad":               ("PUBLIC", "EU customs Single Administrative Document training material."),
    "v1_air_waybill":          ("PUBLIC", "IATA training PDFs for Air Waybills — public industry standards."),
    "v1_cmr":                  ("MIXED",  "CMR consignment note examples — published worked exemplars."),
    "v1_phyto_cert":           ("PUBLIC", "IPPC/FAO international phytosanitary certificate standards."),
    "v1_loc":                  ("PUBLIC", "ICC + commercial training docs for SWIFT MT700 letters of credit."),
    "v1_urla_1003":            ("SYNTHETIC", "Blank URLA + Enact training example with hypothetical borrower data."),
    "v1_hud1":                 ("SYNTHETIC", "Blank HUD-1 settlement statements + worked examples."),
    "v1_w2_1099":              ("BLANK",  "IRS official blank W-2, 1099, W-9, W-4 forms — no real PII."),
    "v1_sec_10q":              ("PUBLIC", "Berkshire Hathaway SEC-filed 10-Q quarterly reports."),
    "v1_lloyds_mrc":           ("PUBLIC", "London insurance market reform contract guidance."),
    "v1_esg":                  ("PUBLIC", "Microsoft public CDP responses + sustainability reports."),
    "v1_bankruptcy_schedule":  ("PUBLIC", "PACER court filings — Chapter 11 records are public by law (FTX, real filed schedules)."),
    "v1_form_5500":            ("BLANK",  "IRS Form 5500 blank + instructions. Real filed 5500s require EFAST2 portal access."),
    "v1_fec_form3":            ("PUBLIC", "FEC-filed campaign finance forms — public by FECA. Includes Form 1, Form 3, Form 3X, Schedule E."),
}

# Search contexts for new types
SEARCH_CONTEXT.update({
    "v1_vaers": "FDA/HHS Vaccine Adverse Event Reporting System — multi-page form, real public reports.",
    "v1_ub04": "CMS-1450 institutional claim form (hospital analog to CMS-1500). 80+ form locators.",
    "v1_eu_sad": "EU Single Administrative Document — master customs declaration with 54 boxes.",
    "v1_air_waybill": "IATA Air Waybill standard. Universal air freight bill structure.",
    "v1_cmr": "CMR consignment note (international road transport convention).",
    "v1_phyto_cert": "IPPC phytosanitary certificate (international agricultural export).",
    "v1_loc": "Letter of Credit + SWIFT MT700 documentary credit messages.",
    "v1_urla_1003": "Fannie Mae URLA Form 1003 — uniform residential mortgage loan application (9pp).",
    "v1_hud1": "HUD-1 settlement statement (predecessor to CD, still used for cash RE closings).",
    "v1_w2_1099": "IRS W-2 wage statement + 1099 information returns + W-9/W-4 forms.",
    "v1_sec_10q": "SEC 10-Q quarterly financial report (complement to 10-K).",
    "v1_lloyds_mrc": "Lloyd's of London Market Reform Contract — UK insurance market form.",
    "v1_esg": "Corporate ESG / CDP / sustainability disclosure reports.",
    "v1_bankruptcy_schedule": "PACER court records — Chapter 11 schedules of assets, liabilities, creditors (FTX exemplars).",
    "v1_form_5500": "DOL Form 5500 retirement plan annual return — pension/401(k) analog to Form 990.",
    "v1_fec_form3": "FEC campaign finance filings — Form 1/3/3X/Schedule E with itemized donor + expenditure tables.",
})

SEARCH_CONTEXT_BUDGET = {
    "v1_budget_cafr": (
        "City/county Annual Comprehensive Financial Reports (ACFR/CAFR). 100-500 pp documents with "
        "deep hierarchical tables (funds → departments → line items), multi-year comparisons, "
        "capital schedules, statistical sections. Pulled from Pittsburg KS, San Francisco CA, "
        "South San Francisco CA, SFCTA."
    ),
    "v1_budget_sf424a": (
        "Federal grant budget form SF-424A (Budget Information - Non-Construction). Required for "
        "NIH/NSF/HUD/DOE grant applications. Has section A (budget summary), B (budget categories), "
        "C (non-federal resources), D (forecasted cash needs), E (federal funds for future periods), "
        "F (other budget info)."
    ),
    "v1_budget_amendment": (
        "Budget amendment / fund transfer request forms. Smaller (2-10pp) but rich: checkbox grids, "
        "multi-line justification, signature blocks. Pulled from Milwaukee County, KY legislature, "
        "City of Billings MT."
    ),
    "v1_budget_990": (
        "IRS Form 990 nonprofit annual returns (Schedule O blank + Schedule A blank + 4 real filed 990s "
        "from Red Cross, American Cancer Society, ProPublica, Internet Archive). Real 990s are "
        "structurally rich: balance sheets, schedules of functional expenses, governance Q&A, "
        "executive compensation tables."
    ),
}
SEARCH_CONTEXT.update(SEARCH_CONTEXT_BUDGET)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_pdf_thumb(pdf: Path, page_idx: int, dpi: int = 72) -> str | None:
    """Render one PDF page as a base64 PNG data URL."""
    try:
        with fitz.open(pdf) as doc:
            if page_idx >= doc.page_count:
                return None
            page = doc.load_page(page_idx)
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            b = pix.tobytes("png")
            return "data:image/png;base64," + base64.b64encode(b).decode()
    except Exception:
        return None


def render_pdf_card(pdf: Path, max_pages: int = 2, dpi: int = 60) -> str:
    """Render up to N page thumbnails as an inline gallery."""
    thumbs = []
    try:
        with fitz.open(pdf) as doc:
            n = min(doc.page_count, max_pages)
            for i in range(n):
                thumb = render_pdf_thumb(pdf, i, dpi=dpi)
                if thumb:
                    thumbs.append(thumb)
            if doc.page_count > max_pages:
                thumbs.append(f"+{doc.page_count - max_pages} more pages")
    except Exception:
        return '<div class="thumb-missing">render error</div>'

    parts = []
    for t in thumbs:
        if t.startswith("data:"):
            parts.append(f'<img class="thumb" src="{t}"/>')
        else:
            parts.append(f'<div class="thumb-more">{html.escape(t)}</div>')
    return '<div class="thumb-row">' + "".join(parts) + "</div>"


def render_json_card(json_path: Path) -> str:
    """Show first ~40 lines of a JSON schema as inline preview."""
    try:
        text = json_path.read_text()[:4000]
    except Exception as e:
        return f'<pre class="json-preview err">read error: {e}</pre>'
    # try to pretty-print
    try:
        obj = json.loads(text + ("..." if len(text) == 4000 else ""))
        text = json.dumps(obj, indent=2)
    except Exception:
        pass
    return f'<pre class="json-preview">{html.escape(text[:2000])}{"…" if len(text) > 2000 else ""}</pre>'


def load_manifest(folder: Path) -> list[dict]:
    m = folder / "_sources_manifest.csv"
    if not m.exists():
        return []
    with m.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Per-type page
# ---------------------------------------------------------------------------

DECISION_JS = r"""
const STORE_KEY = 'realdoc_bench_decisions_v1';
function getStore() { try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch(e) { return {}; } }
function saveStore(s) { localStorage.setItem(STORE_KEY, JSON.stringify(s)); }

function setDecision(btn, choice) {
  const card = btn.closest('.card');
  const t = card.dataset.docType, f = card.dataset.filename;
  const store = getStore();
  if (!store[t]) store[t] = {};
  if (store[t][f] === choice) delete store[t][f];   // toggle off if same
  else store[t][f] = choice;
  if (Object.keys(store[t]).length === 0) delete store[t];
  saveStore(store); applyCardState(card, (store[t]||{})[f]); updateCounts();
}
function applyCardState(card, state) {
  card.classList.remove('accepted', 'rejected');
  const a = card.querySelector('.btn-accept');
  const r = card.querySelector('.btn-reject');
  a && a.classList.remove('active'); r && r.classList.remove('active');
  const s = card.querySelector('.decision-status');
  if (state === 'accept') { card.classList.add('accepted'); a && a.classList.add('active'); s.textContent = 'ACCEPTED'; }
  else if (state === 'reject') { card.classList.add('rejected'); r && r.classList.add('active'); s.textContent = 'REJECTED'; }
  else if (s) { s.textContent = ''; }
}
function init() {
  const store = getStore();
  document.querySelectorAll('.card[data-filename]').forEach(card => {
    const t = card.dataset.docType, f = card.dataset.filename;
    applyCardState(card, (store[t] || {})[f]);
  });
  updateCounts();
}
function updateCounts() {
  const store = getStore();
  let a = 0, r = 0, total = 0;
  document.querySelectorAll('.card[data-filename]').forEach(c => {
    total++;
    const t = c.dataset.docType, f = c.dataset.filename;
    const s = (store[t] || {})[f];
    if (s === 'accept') a++; else if (s === 'reject') r++;
  });
  const u = total - a - r;
  const el = document.getElementById('decisionSummary');
  if (el) el.innerHTML = `<span class="a">${a} accepted</span> · <span class="r">${r} rejected</span> · <span class="u">${u} undecided</span> / ${total}`;
}
function bulkSet(choice, kindFilter) {
  const store = getStore();
  document.querySelectorAll('.card[data-filename]').forEach(c => {
    if (kindFilter && !((c.dataset.kind || '').includes(kindFilter))) return;
    const t = c.dataset.docType, f = c.dataset.filename;
    if (!store[t]) store[t] = {};
    store[t][f] = choice;
  });
  saveStore(store); init();
}
function clearPage() {
  const store = getStore();
  document.querySelectorAll('.card[data-filename]').forEach(c => {
    const t = c.dataset.docType, f = c.dataset.filename;
    if (store[t]) delete store[t][f];
    if (store[t] && Object.keys(store[t]).length === 0) delete store[t];
  });
  saveStore(store); init();
}
function exportDecisions() {
  const blob = new Blob([JSON.stringify({
    schema_version: 1,
    exported_at: new Date().toISOString(),
    decisions: getStore()
  }, null, 2)], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'v1_review_decisions.json';
  a.click();
}
function importDecisions() {
  const input = document.createElement('input');
  input.type = 'file'; input.accept = '.json';
  input.onchange = e => {
    const file = e.target.files[0]; if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const obj = JSON.parse(ev.target.result);
        const dec = obj.decisions || obj;
        saveStore(dec); init();
        alert('Imported decisions for ' + Object.keys(dec).length + ' doc types.');
      } catch (err) { alert('Failed to parse JSON: ' + err); }
    };
    reader.readAsText(file);
  };
  input.click();
}
document.addEventListener('DOMContentLoaded', init);
"""

CSS = """
:root { color-scheme: light; }
body { font: 14px -apple-system, system-ui, sans-serif; margin: 0; background: #f5f5f7; }
.banner { padding: 12px 24px; background: white; border-bottom: 1px solid #ddd; position: sticky; top: 0; z-index: 5; }
.banner h1 { margin: 0; font-size: 18px; }
.banner .sub { color: #666; font-size: 12px; }
.banner a { color: #06c; text-decoration: none; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px 24px; }
.col h2 { margin: 4px 0 12px 0; font-size: 14px; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }
.card { background: white; border-radius: 8px; padding: 12px; margin-bottom: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.card h3 { margin: 0 0 6px 0; font-size: 13px; word-break: break-all; }
.card .meta { font-size: 11px; color: #666; line-height: 1.5; }
.card .meta a { color: #06c; text-decoration: none; word-break: break-all; }
.thumb-row { display: flex; gap: 6px; margin-top: 8px; align-items: flex-start; flex-wrap: wrap; }
.thumb { width: 140px; border: 1px solid #ddd; border-radius: 4px; }
.thumb-more { font-size: 11px; color: #888; padding: 6px 10px; align-self: center; }
.thumb-missing { font-size: 11px; color: #b00; padding: 6px; }
.json-preview { font: 11px Menlo, monospace; background: #fafafa; padding: 8px; border-radius: 4px; max-height: 240px; overflow: auto; white-space: pre-wrap; word-break: break-all; }
.json-preview.err { color: #b00; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; margin-right: 4px; }
.b-public { background: #cfe9ff; color: #014b8a; }
.b-blank { background: #d8f5d8; color: #1a5e1a; }
.b-synthetic { background: #fff3cd; color: #6b4f00; }
.b-mixed { background: #e6dcff; color: #4a2a8c; }
.b-customer { background: #ffdcdc; color: #8a0000; }
.b-filled { background: #d8f5d8; color: #1a5e1a; }
.b-blank-tag { background: #eee; color: #555; }
.b-guide { background: #e3e3ff; color: #2a2a8c; }
.b-pages { background: #f0f0f0; color: #555; font-weight: 400; }
.status-row { margin: 4px 0 12px; font-size: 12px; }
.status-row .desc { color: #666; margin-top: 4px; }
.search-context { background: #fffceb; padding: 12px; border-radius: 8px; border-left: 3px solid #f0c000; line-height: 1.55; }
.empty { color: #888; padding: 20px; text-align: center; font-style: italic; }

/* Synth variants nested */
.synth-block { margin-top: 12px; padding-top: 8px; border-top: 1px solid #eee; }
.synth-block summary { cursor: pointer; font-size: 11px; color: #555; font-weight: 600; padding: 4px 0; }
.synth-block summary:hover { color: #06c; }
.synth-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 8px; margin-top: 8px; }
.synth-variant { padding: 8px; background: #f8f8ff; }
.synth-variant h4 { margin: 0 0 6px 0; font-size: 11px; font-weight: 600; color: #444; }
.synth-variant .thumb { width: 140px; }
.synth-variant .decision-row { margin-top: 6px; padding-top: 6px; gap: 4px; }
.synth-variant .btn { font-size: 11px; padding: 2px 6px; }

/* Decision UI */
.card { transition: border-left 0.15s, background 0.15s; border-left: 4px solid transparent; }
.card.accepted { border-left-color: #1a9e1a; background: #f4fcf4; }
.card.rejected { border-left-color: #c00; background: #fff5f5; opacity: 0.7; }
.decision-row { margin-top: 10px; padding-top: 10px; border-top: 1px dashed #e0e0e0; display: flex; align-items: center; gap: 6px; }
.btn { font: inherit; padding: 4px 12px; border-radius: 6px; border: 1px solid #ccc; background: white; cursor: pointer; font-size: 12px; }
.btn:hover { background: #f0f0f0; }
.btn-accept { color: #1a5e1a; border-color: #b6e6b6; }
.btn-accept:hover { background: #e0f5e0; }
.btn-accept.active { background: #1a9e1a; color: white; border-color: #1a9e1a; }
.btn-reject { color: #b00; border-color: #f5c4c4; }
.btn-reject:hover { background: #fde8e8; }
.btn-reject.active { background: #c00; color: white; border-color: #c00; }
.decision-status { margin-left: auto; font-size: 11px; color: #888; font-weight: 600; }

/* Toolbar */
.toolbar { background: #fafafa; padding: 8px 24px; border-bottom: 1px solid #eee; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.toolbar .spacer { flex: 1; }
.toolbar .summary { font-size: 12px; color: #333; font-weight: 600; margin-right: 12px; }
.toolbar .summary .a { color: #1a9e1a; }
.toolbar .summary .r { color: #c00; }
.toolbar .summary .u { color: #888; }
"""


def load_synth_manifest(folder: Path) -> dict[str, list[dict]]:
    """Return {parent_filename: [synth row, ...], ...}."""
    m = folder / "_synthetic_manifest.csv"
    out: dict[str, list[dict]] = {}
    if not m.exists():
        # Fall back: walk synthetic/ and synthesize basic rows by filename pattern
        synth_dir = folder / "synthetic"
        if not synth_dir.exists():
            return out
        for p in sorted(synth_dir.glob("*.pdf")):
            # Filename pattern: <parent_stem>__p<NN>.pdf
            m_match = re.match(r"^(.*?)__p(\d+)$", p.stem)
            if not m_match:
                continue
            stem, idx = m_match.group(1), m_match.group(2)
            # Find the parent by stem prefix
            parent_fname = stem + ".pdf"
            out.setdefault(parent_fname, []).append({
                "filename": p.name,
                "persona_idx": idx,
                "persona_label": f"persona {idx}",
            })
        return out
    with m.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        parent = r.get("parent_blank") or ""
        if parent:
            out.setdefault(parent, []).append(r)
    return out


def render_type_page(doc_type: str, out_path: Path) -> dict:
    """Render one HTML for `doc_type`. Returns stats for the index."""
    folder = DATASETS / doc_type
    manifest = load_manifest(folder)
    source_files = CUSTOMER_MAPPING.get(doc_type, [])
    search_ctx = SEARCH_CONTEXT.get(doc_type, "")
    pii_status, pii_desc = PII_STATUS.get(doc_type, ("?", ""))
    synth_by_parent = load_synth_manifest(folder)

    # --- LEFT (source) ---
    source_html_parts = []
    if source_files:
        source_html_parts.append(
            f'<div class="status-row"><span class="badge b-customer">CUSTOMER (gitignored)</span> '
            f'<span style="color:#666">never persisted in repo · used only to inform search</span></div>'
        )
        for fname in source_files:
            fpath = CUSTOMER / fname
            if not fpath.exists():
                source_html_parts.append(
                    f'<div class="card"><h3>{html.escape(fname)}</h3>'
                    f'<div class="meta" style="color:#b00">[missing locally]</div></div>'
                )
                continue
            ext = fpath.suffix.lower()
            if ext == ".pdf":
                body = render_pdf_card(fpath, max_pages=2, dpi=60)
                try:
                    with fitz.open(fpath) as d:
                        npages = d.page_count
                except Exception:
                    npages = "?"
                source_html_parts.append(
                    f'<div class="card"><h3>{html.escape(fname)}</h3>'
                    f'<div class="meta"><span class="badge b-pages">{npages} pp</span></div>'
                    f'{body}</div>'
                )
            elif ext == ".json":
                source_html_parts.append(
                    f'<div class="card"><h3>{html.escape(fname)}</h3>'
                    f'<div class="meta"><span class="badge b-mixed">schema</span></div>'
                    f'{render_json_card(fpath)}</div>'
                )
    if search_ctx:
        source_html_parts.append(
            f'<div class="search-context">{html.escape(search_ctx)}</div>'
        )
    if not source_html_parts:
        source_html_parts.append('<div class="empty">No customer source recorded.</div>')

    # --- RIGHT (downloaded) ---
    download_html_parts = []
    if not manifest:
        download_html_parts.append('<div class="empty">No manifest. Run write_v1_manifests.py.</div>')
    for row in manifest:
        fname = row["local_filename"]
        pdf = folder / "sources" / fname
        if not pdf.exists():
            download_html_parts.append(
                f'<div class="card"><h3>{html.escape(fname)}</h3>'
                f'<div class="meta" style="color:#b00">[missing on disk]</div></div>'
            )
            continue
        kind = (row.get("is_filled") or "").lower()
        if "fill" in kind:
            tag = '<span class="badge b-filled">FILLED</span>'
        elif "blank" in kind:
            tag = '<span class="badge b-blank-tag">BLANK</span>'
        elif "guide" in kind:
            tag = '<span class="badge b-guide">GUIDE</span>'
        else:
            tag = f'<span class="badge b-pages">{html.escape(kind or "?")}</span>'
        meta_html = (
            f'{tag}'
            f'<span class="badge b-pages">{row.get("doc_pages", "?")} pp</span>'
            f' <a href="{html.escape(row.get("source_url", ""))}" target="_blank">{html.escape(row.get("source_site", ""))}</a>'
            f'<br><span style="color:#888">{html.escape(row.get("notes", ""))}</span>'
        )
        body = render_pdf_card(pdf, max_pages=2, dpi=60)
        decision_row = (
            '<div class="decision-row">'
            '<button class="btn btn-accept" onclick="setDecision(this, \'accept\')">✓ Accept</button>'
            '<button class="btn btn-reject" onclick="setDecision(this, \'reject\')">✗ Reject</button>'
            '<span class="decision-status"></span>'
            '</div>'
        )
        kind_data = (row.get("is_filled") or "").lower()
        # Synth variants for this parent
        synth_rows = synth_by_parent.get(fname, [])
        synth_inner = ""
        for sr in sorted(synth_rows, key=lambda r: r.get("persona_idx", "")):
            spath = folder / "synthetic" / sr["filename"]
            if not spath.exists():
                continue
            sthumb = render_pdf_card(spath, max_pages=1, dpi=50)
            slabel = sr.get("persona_label", f"p{sr.get('persona_idx', '?')}")
            synth_decision = (
                '<div class="decision-row">'
                f'<button class="btn btn-accept" onclick="setDecision(this, \'accept\')">✓</button>'
                f'<button class="btn btn-reject" onclick="setDecision(this, \'reject\')">✗</button>'
                '<span class="decision-status"></span>'
                '</div>'
            )
            synth_inner += (
                f'<div class="synth-variant card" data-doc-type="{html.escape(doc_type)}" '
                f'data-filename="{html.escape(sr["filename"])}" data-kind="synthetic">'
                f'<h4>p{sr.get("persona_idx", "?")} · {html.escape(slabel)[:55]}</h4>'
                f'{sthumb}{synth_decision}</div>'
            )
        if synth_inner:
            synth_block = (
                '<details class="synth-block" open>'
                f'<summary>{len(synth_rows)} synth variants</summary>'
                f'<div class="synth-grid">{synth_inner}</div>'
                '</details>'
            )
        else:
            synth_block = ''
        download_html_parts.append(
            f'<div class="card" data-doc-type="{html.escape(doc_type)}" '
            f'data-filename="{html.escape(fname)}" data-kind="{html.escape(kind_data)}">'
            f'<h3>{html.escape(fname)}</h3>'
            f'<div class="meta">{meta_html}</div>'
            f'{body}{decision_row}{synth_block}</div>'
        )

    # Page totals
    n_pdfs = len(manifest)
    n_filled = sum(1 for r in manifest if "fill" in (r.get("is_filled") or "").lower())
    n_blank = sum(1 for r in manifest if "blank" in (r.get("is_filled") or "").lower())

    status_cls = {"PUBLIC": "b-public", "BLANK": "b-blank", "SYNTHETIC": "b-synthetic", "MIXED": "b-mixed"}.get(pii_status, "b-pages")
    html_out = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(doc_type)} review</title>
<style>{CSS}</style></head><body data-doc-type="{html.escape(doc_type)}">
<div class="banner">
  <h1>{html.escape(doc_type)} <span class="sub">· {n_pdfs} PDFs · {n_filled} filled · {n_blank} blank</span></h1>
  <div class="status-row">
    <span class="badge {status_cls}">{html.escape(pii_status)}</span>
    <span class="desc">{html.escape(pii_desc)}</span>
    · <a href="INDEX.html">← back to index</a>
  </div>
</div>
<div class="toolbar">
  <span class="summary" id="decisionSummary">…</span>
  <button class="btn" onclick="bulkSet('accept', null)">Accept all</button>
  <button class="btn" onclick="bulkSet('accept', 'fill')">Accept all FILLED</button>
  <button class="btn" onclick="bulkSet('reject', 'blank')">Reject all BLANK</button>
  <button class="btn" onclick="bulkSet('reject', 'guide')">Reject all GUIDE</button>
  <button class="btn" onclick="clearPage()">Clear page</button>
  <span class="spacer"></span>
  <button class="btn" onclick="exportDecisions()">⬇ Export decisions JSON</button>
  <button class="btn" onclick="importDecisions()">⬆ Import JSON</button>
</div>
<div class="cols">
  <div class="col">
    <h2>Source / context</h2>
    {''.join(source_html_parts)}
  </div>
  <div class="col">
    <h2>Downloaded ({n_pdfs} PDFs)</h2>
    {''.join(download_html_parts)}
  </div>
</div>
<script>{DECISION_JS}</script>
</body></html>
"""
    out_path.write_text(html_out)
    return {
        "doc_type": doc_type,
        "n_pdfs": n_pdfs,
        "n_filled": n_filled,
        "n_blank": n_blank,
        "pii_status": pii_status,
        "n_source": len(source_files),
        "has_search_ctx": bool(search_ctx),
    }


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------

def render_index(stats_by_type: dict[str, dict], out_path: Path) -> None:
    rows = []
    for domain, types in DOMAINS.items():
        rows.append(f'<h2>{html.escape(domain)}</h2>')
        rows.append('<table><thead><tr>'
                    '<th>Doc type</th><th>PDFs</th><th>Filled</th><th>Blank</th>'
                    '<th>PII status</th><th>Source</th><th>Decisions</th></tr></thead><tbody>')
        for t in types:
            s = stats_by_type.get(t)
            if not s:
                continue
            status_cls = {"PUBLIC": "b-public", "BLANK": "b-blank", "SYNTHETIC": "b-synthetic", "MIXED": "b-mixed"}.get(s["pii_status"], "b-pages")
            src_label = ""
            if s["n_source"]:
                src_label = f'<span class="badge b-customer">{s["n_source"]} customer file(s)</span>'
            elif s["has_search_ctx"]:
                src_label = '<span class="badge b-pages">search-only</span>'
            rows.append(
                f'<tr data-doc-type="{html.escape(t)}"><td><a href="{t}.html">{html.escape(t)}</a></td>'
                f'<td>{s["n_pdfs"]}</td><td>{s["n_filled"]}</td><td>{s["n_blank"]}</td>'
                f'<td><span class="badge {status_cls}">{html.escape(s["pii_status"])}</span></td>'
                f'<td>{src_label}</td>'
                f'<td class="dec-cell">—</td></tr>'
            )
        rows.append('</tbody></table>')

    out_path.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>v1 review index</title>
<style>
{CSS}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 24px; background: white; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #fafafa; font-size: 12px; text-transform: uppercase; color: #666; }}
tr:hover {{ background: #f8f8f8; }}
td a {{ color: #06c; text-decoration: none; font-weight: 600; }}
h2 {{ margin-top: 24px; padding: 0 24px; }}
.dec-cell {{ font-size: 12px; color: #666; }}
.dec-cell .a {{ color: #1a9e1a; font-weight: 600; }}
.dec-cell .r {{ color: #c00; font-weight: 600; }}
.dec-cell .u {{ color: #888; }}
</style></head><body>
<div class="banner">
  <h1>Dataset review · source vs downloaded</h1>
  <div class="sub">Click any row to inspect source (left) and downloaded PDFs (right) side-by-side.</div>
</div>
<div class="toolbar">
  <span class="summary" id="globalSummary">…</span>
  <span class="spacer"></span>
  <button class="btn" onclick="exportDecisions()">⬇ Export decisions JSON</button>
  <button class="btn" onclick="importDecisions()">⬆ Import JSON</button>
  <button class="btn" onclick="if(confirm('Clear ALL decisions across all types?')){{localStorage.removeItem('realdoc_bench_decisions_v1'); location.reload();}}">Clear all</button>
</div>
{''.join(rows)}
<script>
{DECISION_JS}
function refreshIndex() {{
  const store = getStore();
  let totalA = 0, totalR = 0, totalU = 0;
  document.querySelectorAll('tr[data-doc-type]').forEach(tr => {{
    const t = tr.dataset.docType;
    const pdfsCell = tr.cells[1];
    const total = parseInt(pdfsCell.textContent) || 0;
    const dec = store[t] || {{}};
    let a = 0, r = 0;
    for (const k in dec) {{ if (dec[k] === 'accept') a++; else if (dec[k] === 'reject') r++; }}
    const u = total - a - r;
    totalA += a; totalR += r; totalU += u;
    const cell = tr.querySelector('.dec-cell');
    if (a + r === 0) cell.innerHTML = '<span class="u">untouched</span>';
    else cell.innerHTML = `<span class="a">${{a}}✓</span> · <span class="r">${{r}}✗</span> · <span class="u">${{u}}?</span>`;
  }});
  const el = document.getElementById('globalSummary');
  if (el) el.innerHTML = `<span class="a">${{totalA}} accepted</span> · <span class="r">${{totalR}} rejected</span> · <span class="u">${{totalU}} undecided</span> across all types`;
}}
document.removeEventListener('DOMContentLoaded', init);
document.addEventListener('DOMContentLoaded', refreshIndex);
window.addEventListener('focus', refreshIndex);
</script>
</body></html>
""")


def main() -> int:
    REVIEW.mkdir(parents=True, exist_ok=True)
    stats = {}
    for domain, types in DOMAINS.items():
        for t in types:
            out = REVIEW / f"{t}.html"
            s = render_type_page(t, out)
            stats[t] = s
            print(f"  wrote {out.relative_to(REPO)}  ({s['n_pdfs']} dl, {s['n_source']} src)")
    render_index(stats, REVIEW / "INDEX.html")
    print(f"\nWrote {REVIEW.relative_to(REPO)}/INDEX.html")
    print(f"Open with: open {REVIEW.relative_to(REPO)}/INDEX.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
