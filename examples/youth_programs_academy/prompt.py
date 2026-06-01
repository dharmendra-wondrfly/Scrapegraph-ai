"""
Structured extraction prompt for youth programs (aligned enums / JSON shape).

Replace SUBJECT_URL with the official website URL for each run.
``build_prompt`` also injects a CANDIDATE_URLS block: the list of program-relevant
URLs discovered by ``discovery.py``. The model must pick joiningLink from this list
or fall back to "no data available" — it must NEVER invent URLs.
"""

EXTRACTION_PROMPT_TEMPLATE = """
You are a structured data extraction engine.

Your task is to extract ALL valid programs from the provider's OFFICIAL website at SUBJECT_URL.

-----------------------------------
CANDIDATE URLS (USE ONLY THESE FOR joiningLink)
-----------------------------------
CANDIDATE_URLS_BLOCK

-----------------------------------
STRICT SOURCE RULES
-----------------------------------
• Extract ONLY from the official website
• DO NOT use third-party platforms
• DO NOT assume or infer missing programs

-----------------------------------
EXCLUDE THESE COMPLETELY
-----------------------------------
• Adult-only programs (18+)
• Parent programs
• One-time events or camps
• Competitions / tournaments
• Ceremonies / showcases
• Government programs
• Religious/church programs
• School-only restricted programs
• Programs not clearly listed on the website

If a program matches ANY of the above → SKIP IT

-----------------------------------
EXTRACTION PROCESS (MANDATORY)
-----------------------------------
1. Identify ALL program/class pages on the website
2. For CHILDCARE / DAYCARE providers: treat EVERY NAMED SERVICE, ACTIVITY, CLASS, or OFFERING listed as a SEPARATE program.
   Examples that MUST each become a separate program entry:
   - "Infant Care", "Toddler Care", "Preschool", "Pre-K", "Kindergarten", "After School Care"
   - "Structured Playtime", "Outdoor Play", "Art & Crafts", "Music Class", "STEM Activities"
   - "Age-Appropriate Curriculum", "Nutritional Meals Program", "Nap/Rest Program"
   Even if the website presents these as "features" or "amenities", extract EACH named item as its own program.

   STUB GUARD (MANDATORY): Before emitting any program, confirm it has a name AND
   at least one of (age range OR description text from the site). If you cannot
   support both, DO NOT emit the program. This prevents one-word stubs like
   "Outdoor Play" with every other field blank.
3. Validate each program against exclusion rules
4. Extract ONLY valid youth programs
5. Validate required fields before output

-----------------------------------
AGE GROUP (MANDATORY FIELD)
-----------------------------------

Always return numeric values only (in YEARS).

-----------------------------------
CONVERSION RULES (APPLY FIRST)
-----------------------------------

Convert all age formats into YEARS:

• Weeks → divide by 52

• Months → divide by 12

• Mixed formats → combine values

• Range conversion (preserve range after conversion)

-----------------------------------
PRIORITY (AFTER CONVERSION)
-----------------------------------
1. Exact ages → use directly (after conversion if needed)
2. Grades → convert:
K=5, 1=6, 2=7, 3=8, 4=9, 5=10,
6=11, 7=12, 8=13, 9=14,
10=15, 11=16, 12=17

3. Categories:
Toddlers → 1–3
Preschool → 3–5
Kids → 5–12
Middle School → 11–13
Teens → 13–17
High School → 14–17
Infants → 0–1

4. "All ages" or "Family" → return 5–17 only

5. If missing:
minAge = "no data available"
maxAge = "no data available"

-----------------------------------
IMPORTANT RULES
-----------------------------------

• Always return numbers only (no units like months/years)
• Round values to maximum 2 decimal places
• Do NOT return text like "kids", "teens", "infants"
• If multiple age ranges exist, choose the most specific range for that program

-----------------------------------
STRICT DATA RULES
-----------------------------------
• DO NOT hallucinate
• DO NOT generate missing data
• If not found → "no data available"
• Use exact website values only
• Program name must be specific

-----------------------------------
LITERAL-TEXT RULE (ZERO TOLERANCE — READ TWICE)
-----------------------------------
For these fields you MUST copy values that literally appear in the page text.
If the exact value is NOT physically present in the text you are reading, you
MUST return the empty/"no data available" form. NEVER infer, guess, or invent:

• schedules — only emit a day/startTime/endTime if that EXACT day and clock time
  appear verbatim on the page. No page text with times → schedules: []
  (Inventing plausible times like "Monday 9:00 AM" is a CRITICAL FAILURE.)
• prices / pricePerParticipant — only emit a number that appears verbatim
  ($, digits) on the page. No printed price → prices: [], pricingData per rules.
• ageGroup — only from ages/grades printed on the page (after conversion rules).
• maxNumberOfStudents, offerDiscount — only if literally stated.

Self-check before output: for every schedule time and every price you emit, ask
"can I point to those exact characters in the page text?" If no → delete it.

-----------------------------------
STRUCTURE RULES (VERY IMPORTANT)
-----------------------------------
• schedules MUST be an array (even if one)
• prices MUST be an array (even if one)

-----------------------------------
PRICE FORMAT RULES
-----------------------------------
Use consistent values like:

priceUnit:
"per session", "per hour", "per month", "per week"

-----------------------------------
PRICE TYPE CLASSIFICATION RULES
-----------------------------------

priceType MUST be exactly one of the following values:

• "Weekly / Monthly / Yearly Class Pricing"
• "Semester Pricing"
• "Single class"
• "Multi class (Package)"
• "Memberships / Passes"
• "Party packages"
• "Camps"

-----------------------------------
MAPPING RULES
-----------------------------------

• Recurring class (weekly/monthly) → "Weekly / Monthly / Yearly Class Pricing"
• Semester/session pricing → "Semester Pricing"
• Single class/session → "Single class"
• Packages / fixed duration bundles → "Multi class (Package)"
• Passes/memberships → "Memberships / Passes"
• Birthday/event packages → "Party packages"
• Camps → "Camps"

-----------------------------------
IMPORTANT RULES
-----------------------------------

• NEVER create new values like "Program Pricing"
• ALWAYS map to one of the above values
• Match the value EXACTLY (case + spacing must be identical)

-----------------------------------
PRICING DATA FIELD (MANDATORY)
-----------------------------------

Set "pricingData" based on pricing availability.

• Explicit numeric prices → "Price available"
• Explicitly FREE → "It is a free program"
• "Contact for pricing" etc. → "Price can be discussed"
• Pricing section unclear → "Not specified"
• No pricing info → "No data available"

-----------------------------------
PROGRAM TYPE CLASSIFICATION
-----------------------------------

"type" MUST be exactly one of:

• "Party"
• "camps"
• "classes"

Mapping:
• Birthday/event → "Party"
• Seasonal camp → "camps"
• Recurring class / training → "classes"

-----------------------------------
JOINING LINK FORMAT RULES
-----------------------------------

joiningLink = URL of the specific program page on the official website.

PRIORITY:
1. Program detail page
2. Category page if no detail page
3. Registration page only if nothing else exists
4. Else → "no data available"

STRICT URL RULES:
• You MUST pick joiningLink from the CANDIDATE_URLS block at the top of this prompt.
• If no candidate URL is a reasonable match for this program → "no data available".
• Do NOT invent URLs. Do NOT modify, append query strings, or guess fragments.
• Do NOT use third-party platforms (Mindbody, ClassDojo, Eventbrite, etc.) unless
  that exact URL appears in CANDIDATE_URLS.

-----------------------------------
FIELD SHAPE EXAMPLES (READ BEFORE OUTPUT)
-----------------------------------

ageGroup — numeric YEARS only, OR the exact string "no data available":
✓ { "minAge": "3", "maxAge": "5" }
✓ { "minAge": "0.5", "maxAge": "2" }
✓ { "minAge": "no data available", "maxAge": "no data available" }
✗ { "minAge": "3 years", "maxAge": "5 years old" }
✗ { "minAge": "kids", "maxAge": "teens" }
✗ { "minAge": "", "maxAge": "" }   ← never empty string; use "no data available"

schedules — ALWAYS an array. Each entry one day + start/end time:
✓ [{ "day": "Monday", "startTime": "4:00 PM", "endTime": "5:00 PM", "frequency": "weekly" }]
✓ []   ← when the website lists no schedule
✗ { "day": "Monday", ... }    ← never a bare object; wrap in []
✗ [{ "day": "Mon-Wed-Fri", ... }]  ← split into 3 separate entries

prices — ALWAYS an array. Numeric values go in pricePerParticipant / pricePerHour
WITHOUT currency symbols. Match priceType to one of the 7 canonical enum values:
✓ [{ "priceUnit": "per session", "priceType": "Single class",
     "pricePerParticipant": "25", "classDuration": "60 min" }]
✓ [{ "priceUnit": "per month", "priceType": "Weekly / Monthly / Yearly Class Pricing",
     "pricePerParticipant": "180" }]
✗ [{ "pricePerParticipant": "$25/class" }]   ← strip $ and "/class"
✗ "priceType": "Program Pricing"             ← not a canonical value
✗ "priceType": "monthly"                     ← must be the full enum string

joiningLink — string (URL from CANDIDATE_URLS, or "no data available"):
✓ "https://example.com/programs/dance-toddlers"
✓ "no data available"
✗ ""

-----------------------------------
PROVIDER PROFILE (REQUIRED OUTPUT ALONGSIDE programs)
-----------------------------------

Also produce a single "provider" object with provider-level details extracted from
the homepage, about page, and contact page only. Conservative — "no data available"
or empty list when not visible on the site.

Shape:
{
  "name": "<official business name>",
  "address": "<full street address including city, state, zip when visible>",
  "phone": "<primary phone, digits with separators>",
  "email": "<primary contact email>",
  "description": "<1–3 sentence summary written by the provider on their About page; do not paraphrase or invent>",
  "categories": ["<top-level offering categories visible on the site, e.g. 'Dance', 'Music'>"],
  "subjects": ["<more specific subjects/styles, e.g. 'Ballet', 'Hip Hop'>"]
}

Rules:
• Use EXACT text from the website for address/description.
• If a field is not visible on the site, use "no data available" for strings or
  [] for arrays. Do NOT fabricate.
• Provider name fallback: use the prominent name in the header/footer/logo alt text.

-----------------------------------
OUTPUT FORMAT
-----------------------------------
Produce ONLY structured output matching the provided schema:
• Root object has key "programs" (array of program objects) AND key "provider" (object)
• Follow schema field names and nesting exactly.

-----------------------------------
FINAL VALIDATION
-----------------------------------
Before returning:

✓ Every program has minAge & maxAge
✓ schedules is array
✓ prices is array
✓ No adult programs
✓ No excluded categories
✓ No fabricated values

If unsure → return "no data available"

-----------------------------------
REFERENCE EXAMPLE (ENUM VALUES MUST MATCH)
-----------------------------------
Example for prices[].priceType: use "Weekly / Monthly / Yearly Class Pricing" (not informal variants).
Example for type: use "classes" for recurring youth classes (not "class").
Each prices[] entry has a single "duration" field (do not duplicate keys).

"""


def _format_candidate_urls(candidate_urls: list[str] | None) -> str:
    """Render the CANDIDATE_URLS block for the prompt. Falls back when empty."""
    if not candidate_urls:
        return (
            "(no candidate URLs were discovered for this provider — "
            'use joiningLink "no data available" unless a program detail URL '
            "appears directly inside the page text you are reading.)"
        )
    bullets = "\n".join(f"- {u}" for u in candidate_urls)
    return (
        "joiningLink values for any program MUST come from this list "
        '(or be exactly "no data available"):\n' + bullets
    )


def build_prompt(subject_url: str, candidate_urls: list[str] | None = None) -> str:
    """Substitute the target official URL and discovered candidate URLs into the prompt.

    ``candidate_urls`` is the list returned by ``discovery.discover_program_urls``;
    the prompt constrains joiningLink to this set so the LLM cannot invent URLs.
    Optional for back-compat with older callers that pass only the URL.
    """
    rendered = EXTRACTION_PROMPT_TEMPLATE.replace("SUBJECT_URL", subject_url.strip())
    rendered = rendered.replace(
        "CANDIDATE_URLS_BLOCK", _format_candidate_urls(candidate_urls)
    )
    return rendered


PROVIDER_PROFILE_PROMPT = """
You are extracting provider-level (not program-level) details from the official
website at SUBJECT_URL.

Use the homepage and the about/contact pages provided. DO NOT extract individual
classes or programs — produce a single "provider" object with these fields:

  name         — official business name
  address      — full street address with city, state, zip when visible on the site
  phone        — primary phone number (digits + separators)
  email        — primary contact email
  description  — 1–3 sentences from the provider's About page, verbatim or near-verbatim
  categories   — top-level offering categories visible on the site (array of strings)
  subjects     — more specific subjects/styles (array of strings)

Rules:
• Use EXACT text from the website where possible.
• If a field is not visible on the site, return "no data available" for strings
  or [] for arrays.
• DO NOT fabricate any value.

Return ONLY structured output matching the ProviderProfile schema — no extra keys.
"""


def build_provider_profile_prompt(subject_url: str) -> str:
    """Render the provider-profile-only prompt for the auxiliary lite graph pass."""
    return PROVIDER_PROFILE_PROMPT.replace("SUBJECT_URL", subject_url.strip())
