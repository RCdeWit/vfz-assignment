"""Fetch the product catalogue behind the ziggo.nl configurator and turn it into one page.

The product pages under ``/internet`` show speeds but no prices. Prices appear
only in the configurator at ``/tv-internet``, which loads them from a JSON
endpoint under ``/services/blueriq/b2c/info`` with a base64 payload. This
module calls that endpoint and writes the result as a Page in the same shape
as a scraped page, so chunking, ingest, and the snapshot treat it like any
other page. The URL of the page is the configurator's, so answers cite it.

The page has one overview section per product family (Internet, Televisie,
Vast bellen) that lists every product with its monthly price, and one section
per product with the price, the promotion when there is one, and the
included features. Prices are those of a 12 month contract, the
configurator's default; the overview says so.

Run the module directly to fetch the catalogue and print the sections.
"""

import base64
import json
import logging
import re
import sys

import httpx

from ziggo_rag.config import settings
from ziggo_rag.pages import TIMEOUT_S, USER_AGENT, Page, ScrapeError, Section

log = logging.getLogger("ziggo_rag.catalogue")

# The configurator sends this payload on load, before the customer selects
# anything. An empty selection returns every product; responseType Full
# includes pricing and the feature lists.
CONTRACT_MONTHS = 12
PAYLOAD = {
    "selectedItems": [{"selectedItem": []}],
    "customerInformation": {"profile": "", "maxNetworkDownloadSpeed": None, "affiliatedBrand": ""},
    "responseType": "Full",
    "contractDuration": CONTRACT_MONTHS,
    "path": "/tv-internet",
}
PAGE_URL = "https://www.ziggo.nl/tv-internet"
PAGE_TITLE = "Abonnementen en prijzen in de bestelpagina van Ziggo"
# Family titles in the catalogue are English; the page text is Dutch like the
# rest of the corpus. Families not listed here (Mobile holds only discount
# lines) are skipped.
FAMILIES = {"Internet": "Internet", "Television": "Televisie", "Telephone": "Vast bellen"}
_TAGS = re.compile(r"<[^>]+>")


def fetch_catalogue() -> Page:
    """GET the catalogue and return it as a Page. Raises ScrapeError when the fetch fails."""
    payload = base64.b64encode(json.dumps(PAYLOAD, separators=(",", ":")).encode()).decode()
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = settings().catalogue_url
    try:
        resp = httpx.get(url, params={"payload": payload}, headers=headers, timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        raise ScrapeError(f"catalogue {url}: {type(e).__name__}: {e}") from e
    log.info("GET %s -> %d, %d bytes", url, resp.status_code, len(resp.content))
    if resp.status_code != 200:
        raise ScrapeError(f"catalogue {url}: status {resp.status_code}")
    try:
        page = catalogue_page(resp.json())
    except (ValueError, KeyError, TypeError) as e:
        raise ScrapeError(f"catalogue {url}: unexpected response: {e}") from e
    if not page["sections"]:
        raise ScrapeError(f"catalogue {url}: no priced products in the response")
    return page


def catalogue_page(data: dict) -> Page:
    """Build the Page from the decoded catalogue response.

    Reads ``subCatalog[].productFamily[].product[]``. A product is used when
    its family is in FAMILIES, it is visible, it has a non-negative monthly
    price, and its pricing has no ``bindingPeriod``. The entries without a
    price are contract options, gifts, and discount lines. The entries with a
    binding period (Kabel TV, TV Start CI+, standalone Vast Bellen) are
    monthly standalone products sold outside this configurator; their prices
    do not apply to the 12 month offer the rest of the page describes.
    """
    sections: list[Section] = []
    for sub in data["subCatalog"]:
        for family in sub["productFamily"]:
            name = FAMILIES.get(family["title"])
            if name is None:
                continue
            products = [p for p in family.get("product", []) if _priced(p)]
            if not products:
                continue
            sections.append(_overview(name, products))
            sections.extend(_product_section(name, p) for p in products)
    return {"url": PAGE_URL, "title": PAGE_TITLE, "sections": sections}


def _priced(product: dict) -> bool:
    pricing = product.get("pricing")
    return (
        bool(product.get("isVisible"))
        and pricing is not None
        and pricing["valueMonthly"] >= 0
        and "bindingPeriod" not in pricing
    )


def _overview(family: str, products: list[dict]) -> Section:
    lines = [
        f"Prijzen per maand bij een contract van {CONTRACT_MONTHS} maanden, "
        f"volgens de bestelpagina ziggo.nl/tv-internet."
    ]
    lines += [f"{p['name']}: {_price(p)}" for p in products]
    cheapest = min(products, key=lambda p: p["pricing"]["valueMonthly"])
    dearest = max(products, key=lambda p: p["pricing"]["valueMonthly"])
    if cheapest is not dearest:
        lines.append(f"Het goedkoopste is {cheapest['name']} ({_price(cheapest)}).")
        lines.append(f"Het duurste is {dearest['name']} ({_price(dearest)}).")
    if family == "Internet":
        # The product names carry the speed, so the fastest is the dearest
        # here; said in words because customers ask for "het snelste".
        lines.append(f"Het snelste internetabonnement is {dearest['name']}.")
    return {"heading": f"{family}: abonnementen en prijzen", "text": "\n".join(lines)}


def _product_section(family: str, product: dict) -> Section:
    lines = [f"Prijs: {_price(product)} (contract van {CONTRACT_MONTHS} maanden)."]
    pricing = product["pricing"]
    if pricing.get("promoValueMonthly", pricing["valueMonthly"]) != pricing["valueMonthly"]:
        months = pricing.get("discountDuration")
        during = f" gedurende {months} maanden" if months else ""
        lines.append(
            f"Actieprijs: {_euro(pricing['promoValueMonthly'])} per maand{during}, "
            f"daarna {_price(product)}."
        )
    offer = product.get("offerName")
    if offer and offer != product["name"]:
        lines.append(f"Volledige naam: {offer}.")
    included = [u["title"] for u in product.get("uspsList", []) if not u.get("isCrossed")]
    excluded = [u["title"] for u in product.get("uspsList", []) if u.get("isCrossed")]
    if included:
        lines.append("Inbegrepen: " + "; ".join(included) + ".")
    if excluded:
        lines.append("Niet inbegrepen: " + "; ".join(excluded) + ".")
    for key in ("benefitsText", "productCampaignDescription", "description"):
        text = _plain(product.get(key) or "")
        if text:
            lines.append(text)
    return {"heading": f"{family}: {product['name']}", "text": "\n".join(lines)}


def _price(product: dict) -> str:
    value = product["pricing"]["valueMonthly"]
    if value == 0:
        return product.get("priceZeroText") or "gratis"
    return f"{_euro(value)} per maand"


def _euro(value: float) -> str:
    """Format 41.95 as ``€ 41,95``, the notation used on the pages."""
    return "€ " + f"{value:.2f}".replace(".", ",")


def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub(" ", html)).strip()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        page = fetch_catalogue()
    except ScrapeError as e:
        log.error("%s", e)
        return 1
    for section in page["sections"]:
        print(f"## {section['heading']}\n{section['text']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
