"""
URL feature extraction for phishing detection.

Reproduces the URL-based (lexical) features of the dataset by Vrbančič et al.,
"Datasets for phishing websites detection", Data in Brief 33 (2020).
Only features computable from the URL string itself are used, so any URL
typed by the user can be classified instantly, without DNS / WHOIS lookups.

A URL is split into four parts and the same character counts are taken for each:

    http://secure-login.example.com/account/verify/update.php?id=1&ref=mail
           |_______ domain _______||____ directory ___||_ file _||_ params _|
"""

import re
from urllib.parse import urlsplit

# Characters counted in every URL part (same order as the dataset columns)
SPECIAL_CHARS = [
    ("dot", "."), ("hyphen", "-"), ("underline", "_"), ("slash", "/"),
    ("questionmark", "?"), ("equal", "="), ("at", "@"), ("and", "&"),
    ("exclamation", "!"), ("space", " "), ("tilde", "~"), ("comma", ","),
    ("plus", "+"), ("asterisk", "*"), ("hashtag", "#"), ("dollar", "$"),
    ("percent", "%"),
]

# Common top-level domains (used to count TLD-like tokens inside the URL)
COMMON_TLDS = {
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "io", "co",
    "us", "uk", "de", "fr", "it", "es", "nl", "ru", "br", "in", "cn", "jp",
    "au", "ca", "ch", "se", "no", "pl", "be", "at", "dk", "fi", "pt", "gr",
    "tr", "ir", "kr", "mx", "ar", "za", "id", "my", "sg", "hk", "tw", "vn",
    "ua", "cz", "ro", "hu", "il", "nz", "ie", "cl", "pk", "xyz", "top", "site",
    "online", "club", "shop", "app", "dev", "me", "tk", "ml", "ga", "cf", "gq",
    "ly", "cc", "tv", "ws", "su", "pw", "live", "icu", "sbs", "store", "tech",
}

URL_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bitly.com", "cutt.ly", "rebrand.ly", "shorturl.at", "tiny.cc",
    "rb.gy", "t.ly", "s.id", "v.gd", "qr.ae", "lnkd.in", "db.tt", "x.co",
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def _count_chars(text, prefix):
    return {f"qty_{name}_{prefix}": text.count(ch) for name, ch in SPECIAL_CHARS}


def _missing(prefix):
    """The dataset stores -1 when a URL part (directory/file/params) is absent."""
    feats = {f"qty_{name}_{prefix}": -1 for name, _ in SPECIAL_CHARS}
    feats[f"{prefix}_length"] = -1
    return feats


def _count_tlds(text):
    tokens = re.split(r"[^a-z0-9]+", text.lower())
    return sum(1 for t in tokens if t in COMMON_TLDS)


def extract_features(url):
    """Return a dict of lexical features for one URL (keys = FEATURE_COLUMNS)."""
    url = url.strip()
    if "://" not in url:
        url = "http://" + url

    parts = urlsplit(url)
    domain = parts.hostname or ""
    path = parts.path
    query = parts.query

    f = {}

    # --- Whole URL (the dataset stores URLs without the "http://" scheme) ---
    bare = url.split("://", 1)[1]
    f.update(_count_chars(bare, "url"))
    f["qty_tld_url"] = _count_tlds(bare)
    f["length_url"] = len(bare)

    # --- Domain ---
    f.update(_count_chars(domain, "domain"))
    f["qty_vowels_domain"] = sum(domain.count(v) for v in "aeiou")
    f["domain_length"] = len(domain)
    f["domain_in_ip"] = int(bool(IPV4_RE.match(domain)) or ":" in domain)
    f["server_client_domain"] = int("server" in domain or "client" in domain)

    # --- Directory (path up to and including the last "/") ---
    if path and "/" in path:
        directory = path[: path.rfind("/") + 1]
        file_name = path[path.rfind("/") + 1:]
    else:
        directory, file_name = "", ""

    if directory:
        f.update(_count_chars(directory, "directory"))
        f["directory_length"] = len(directory)
    else:
        f.update(_missing("directory"))

    # --- File (last path segment; empty but present when a directory exists) ---
    if directory:
        f.update(_count_chars(file_name, "file"))
        f["file_length"] = len(file_name)
    else:
        f.update(_missing("file"))

    # --- Parameters (query string) ---
    if query:
        f.update(_count_chars(query, "params"))
        f["params_length"] = len(query)
        f["tld_present_params"] = int(_count_tlds(query) > 0)
        f["qty_params"] = len([p for p in query.split("&") if p])
    else:
        f.update(_missing("params"))
        f["tld_present_params"] = -1
        f["qty_params"] = -1

    # --- Other ---
    f["email_in_url"] = int(bool(EMAIL_RE.search(url)))
    f["url_shortened"] = int(domain.lower().removeprefix("www.") in URL_SHORTENERS)

    return {col: f[col] for col in FEATURE_COLUMNS}


def _build_columns():
    cols = [f"qty_{n}_url" for n, _ in SPECIAL_CHARS] + ["qty_tld_url", "length_url"]
    cols += [f"qty_{n}_domain" for n, _ in SPECIAL_CHARS]
    cols += ["qty_vowels_domain", "domain_length", "domain_in_ip", "server_client_domain"]
    for part in ("directory", "file", "params"):
        cols += [f"qty_{n}_{part}" for n, _ in SPECIAL_CHARS] + [f"{part}_length"]
    cols += ["tld_present_params", "qty_params", "email_in_url", "url_shortened"]
    return cols


# The 98 URL-based columns used to train the SVM (in dataset order)
FEATURE_COLUMNS = _build_columns()
