"""
plot_classify.py -- assign a LAND TYPE and an OWNER CLASS to each land plot in an area search, using
only FREE data sources. Plugs into land_viability's area search (the land_plot layer).

LAND TYPE  <- OpenStreetMap land-use / natural / water tagging (ODbL, free via the Overpass API),
   spatially joined to each plot. Buckets:
     River, Lake, Residential, Retail, Industrial, Farmland, Woodland, Amenity, Institutional,
     Hybrid (a plot spanning two+ major types), Unknown (no OSM coverage / unmapped).

OWNER CLASS <- one of "Corporate", "Government", or "Other". ("Other" = likely individual / unknown --
   your app can display it as "Other - Likely Individual".) Two tiers:
   * FREE DEFAULT (no extra data, always on): a plot is GOVERNMENT when OSM shows a public / civic /
     institutional / military use, or an operator like a council, the NHS, Network Rail, the Crown,
     the Environment Agency, Homes England, etc. Everything else is "Other" -- because CORPORATE
     freehold ownership cannot be told from OSM alone.
   * OPTIONAL, adds CORPORATE (turn on by giving a CCOD/OCOD file + an OS_API_KEY): HM Land Registry's
     free Commercial & Corporate / Overseas Companies Ownership Data names every company- and
     public-body-owned title. Each plot's centre is reverse-geocoded to a postcode (OS Places), that
     postcode is looked up in CCOD/OCOD, and the registered proprietor's category decides the class:
       - councils / Crown / NHS / police / fire / government departments -> Government
       - companies, charities, housing associations, societies, overseas companies -> Corporate
       - no match -> Other (likely an individual, or unregistered).
     Postcode-level matching is approximate (a postcode can hold several titles) -- it's a best-effort
     signal, reported with a source string, not a guarantee.

Everything degrades gracefully: if Overpass, OS Places or CCOD is unavailable, plots still come back
with a best-effort type and owner_class="Other" -- the area search never fails on classification.

Download (only if you want the CORPORATE tier): CCOD + OCOD, free (register + accept a licence), from
https://use-land-property-data.service.gov.uk/  (they're CSV files).
"""
import csv
import os
import re
import sys
from collections import defaultdict

BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# The land-type buckets, in the priority order ties are broken by (most specific first).
LAND_TYPES = ["River", "Lake", "Institutional", "Residential", "Retail", "Industrial",
              "Farmland", "Woodland", "Amenity", "Hybrid", "Unknown"]
HYBRID_MIN_FRACTION = 0.25   # a plot is "Hybrid" if 2+ types each cover at least this share of it

# Words in an OSM operator/owner/name (or a CCOD proprietorship category) that mean public/government.
_GOV_WORDS = ("council", "authority", "county council", "borough", "district council", "city council",
              "government", "secretary of state", "the crown", "crown estate", "duchy", "royal",
              "ministry of", "department for", "department of", "\bmod\b", "ministry of defence",
              "\bnhs\b", "national health", "health service", "network rail", "national highways",
              "highways england", "environment agency", "forestry england", "forestry commission",
              "homes england", "police", "fire authority", "fire and rescue", "transport for")


def _osm_type_for_tags(t):
    """Map one OSM feature's tags to a land-type bucket, or None if it isn't a land-use feature."""
    if not t:
        return None
    g = {k: (v or "").lower() for k, v in t.items()}
    lu, nat, wat, ww = g.get("landuse", ""), g.get("natural", ""), g.get("water", ""), g.get("waterway", "")
    leis, am, off = g.get("leisure", ""), g.get("amenity", ""), g.get("office", "")
    # water first
    if ww in ("river", "stream", "canal") or wat == "river":
        return "River"
    if nat == "water" or wat in ("lake", "pond", "reservoir", "basin") or lu == "reservoir":
        return "Lake"
    # public / institutional / military
    if (off == "government" or lu in ("military", "religious")
            or am in ("school", "college", "university", "hospital", "clinic", "place_of_worship",
                      "prison", "fire_station", "police", "courthouse", "townhall")):
        return "Institutional"
    # built uses
    if lu == "residential":
        return "Residential"
    if lu in ("retail", "commercial") or "shop" in g:
        return "Retail"
    if lu in ("industrial", "warehouse", "port", "quarry"):
        return "Industrial"
    # rural
    if lu in ("farmland", "farmyard", "meadow", "orchard", "allotments", "vineyard", "greenhouse_horticulture"):
        return "Farmland"
    if lu == "forest" or nat == "wood":
        return "Woodland"
    # amenity / green space
    if (leis in ("park", "garden", "recreation_ground", "pitch", "playground", "common",
                 "nature_reserve", "golf_course")
            or lu in ("recreation_ground", "village_green", "grass", "cemetery")
            or nat in ("heath", "scrub", "grassland")):
        return "Amenity"
    return None


def _owner_from_osm(t):
    """Return 'Government' if an OSM feature's tags clearly indicate a public/civic/military owner,
    else None. (Corporate freehold ownership is NOT reliably in OSM, so it isn't inferred here.)"""
    if not t:
        return None
    g = {k: (v or "").lower() for k, v in t.items()}
    if g.get("landuse") == "military" or g.get("office") == "government":
        return "Government"
    if g.get("amenity") in ("school", "college", "university", "hospital", "police",
                            "fire_station", "courthouse", "townhall", "prison"):
        return "Government"
    blob = " ".join([g.get("operator", ""), g.get("owner", ""), g.get("name", "")])
    if any(re.search(w, blob) for w in _GOV_WORDS):
        return "Government"
    return None


# ============================================================================
# OSM fetch (one Overpass call per search)
# ============================================================================
def fetch_osm_landuse(aoi_bng, aoi_crs=BNG, timeout=60):
    """Fetch OSM land-use/natural/water/leisure/amenity polygons in the AOI via the Overpass API,
    returned as a BNG GeoDataFrame with 'ltype' and 'owner_hint' columns. Empty GDF on any failure."""
    import geopandas as gpd
    import requests
    from shapely.geometry import Polygon

    aoi_wgs = gpd.GeoSeries([aoi_bng], crs=aoi_crs).to_crs(WGS84).iloc[0]
    minx, miny, maxx, maxy = aoi_wgs.bounds           # lon/lat
    s, w, n, e = miny, minx, maxy, maxx               # Overpass wants (south,west,north,east)
    bbox = f"{s},{w},{n},{e}"
    # 'out geom;' returns tags AND geometry (the default 'body' verbosity includes tags). Also fetch
    # relations (big woods/water are multipolygons), not just ways.
    q = (f"[out:json][timeout:{timeout}];("
         f'way["landuse"]({bbox});way["natural"]({bbox});way["water"]({bbox});'
         f'way["waterway"]({bbox});way["leisure"]({bbox});way["amenity"]({bbox});'
         f'way["office"="government"]({bbox});'
         f'relation["landuse"]({bbox});relation["natural"]({bbox});relation["leisure"]({bbox});'
         f");out geom;")
    # Try several public Overpass mirrors -- the main one is often busy / rate-limits, which was
    # silently turning every plot 'Unknown'.
    endpoints = ["https://overpass-api.de/api/interpreter",
                 "https://overpass.kumi.systems/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
    data = None
    for url in endpoints:
        try:
            r = requests.post(url, data={"data": q}, timeout=timeout + 15)
            if r.status_code == 200:
                data = r.json()
                break
            print(f"    OSM: {url} -> HTTP {r.status_code}")
        except Exception as ex:
            print(f"    OSM: {url} failed ({ex})")
    if data is None:
        raise RuntimeError("all Overpass endpoints failed/blocked")
    elements = data.get("elements", [])
    print(f"    OSM: {len(elements)} element(s) returned by Overpass")
    def _poly(pts):
        if len(pts) < 3:
            return None
        try:
            p = Polygon([(q["lon"], q["lat"]) for q in pts])
            if not p.is_valid:
                p = p.buffer(0)
            return None if p.is_empty else p
        except Exception:
            return None

    geoms, tag_dicts = [], []
    for el in elements:
        tags = el.get("tags", {})
        if el.get("type") == "relation":                      # multipolygon: use its outer rings
            for m in el.get("members", []):
                if m.get("role") == "outer":
                    p = _poly(m.get("geometry") or [])
                    if p is not None:
                        geoms.append(p)
                        tag_dicts.append(tags)
            continue
        p = _poly(el.get("geometry") or [])
        if p is not None:
            geoms.append(p)
            tag_dicts.append(tags)
    if not geoms:
        return gpd.GeoDataFrame({"ltype": [], "owner_hint": []}, geometry=[], crs=BNG)
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=WGS84).to_crs(BNG)
    gdf["ltype"] = [_osm_type_for_tags(t) for t in tag_dicts]
    gdf["owner_hint"] = [_owner_from_osm(t) for t in tag_dicts]
    return gdf.reset_index(drop=True)


# ============================================================================
# Per-plot classification
# ============================================================================
def classify_land_type(plot_geom, osm_gdf):
    """(land_type, source) for one plot, by area-weighted overlap with OSM land-use features."""
    if osm_gdf is None or len(osm_gdf) == 0:
        return "Unknown", "no OSM data"
    try:
        cand = list(osm_gdf.sindex.query(plot_geom, predicate="intersects"))
    except Exception:
        cand = list(range(len(osm_gdf)))
    if not cand:
        return "Unknown", "no OSM overlap"
    area = plot_geom.area or 1.0
    frac = defaultdict(float)
    for i in cand:
        lt = osm_gdf.iloc[i]["ltype"]
        if not lt:
            continue
        try:
            inter = plot_geom.intersection(osm_gdf.geometry.iloc[i])
        except Exception:
            continue
        if not inter.is_empty:
            frac[lt] += inter.area / area
    if not frac:
        return "Unknown", "OSM features had no mapped land-use"
    major = sorted([lt for lt, f in frac.items() if f >= HYBRID_MIN_FRACTION],
                   key=lambda lt: LAND_TYPES.index(lt) if lt in LAND_TYPES else 99)
    if len(major) >= 2:
        return "Hybrid", "OSM: " + " + ".join(major)
    best = max(frac, key=frac.get)
    return best, f"OSM {best} ({round(frac[best] * 100)}% of plot)"


def _gov_from_osm(plot_geom, osm_gdf):
    """True if any overlapping OSM feature marks this plot as public/institutional/military."""
    if osm_gdf is None or len(osm_gdf) == 0:
        return False
    try:
        cand = list(osm_gdf.sindex.query(plot_geom, predicate="intersects"))
    except Exception:
        cand = list(range(len(osm_gdf)))
    for i in cand:
        if osm_gdf.iloc[i]["owner_hint"] == "Government":
            try:
                if plot_geom.intersects(osm_gdf.geometry.iloc[i]):
                    return True
            except Exception:
                pass
    return False


def classify_plots(plots_gdf, aoi_bng, aoi_crs=BNG, ccod=None, reverse_geocode_bulk=None):
    """Add land_type / owner_class / class_source columns to a plots GeoDataFrame (in the plots' CRS).
    Fetches OSM ONCE for the whole AOI, then classifies each plot. Ownership is done in BULK: plots that
    aren't already Government-by-OSM are reverse-geocoded to a postcode in a SINGLE batched call (per
    ~100 plots) rather than one HTTP request each -- this is what keeps a big search from timing out.
    Never raises: on any failure plots come back land_type='Unknown', owner_class='Other'."""
    from pyproj import Transformer
    out = plots_gdf.copy()
    osm = None
    try:
        osm = fetch_osm_landuse(aoi_bng, aoi_crs)
        print(f"  classify: {len(osm)} OSM land-use feature(s) in the AOI")
    except Exception as e:
        print(f"  classify: OSM land-use unavailable ({e}); land types will be 'Unknown'.")

    to_wgs = Transformer.from_crs(BNG, WGS84, always_xy=True)
    types, owners, sources = [], [], []
    pending_idx, pending_lonlat = [], []          # plots needing a CCOD postcode lookup
    for i, geom in enumerate(out.geometry):
        lt, lts = classify_land_type(geom, osm)
        types.append(lt)
        if _gov_from_osm(geom, osm):
            owners.append("Government")
            sources.append(f"type: {lts}; owner: OSM public/institutional use")
        else:
            owners.append("Other")               # provisional; may be upgraded by the CCOD batch below
            sources.append(f"type: {lts}; owner: no corporate/public match")
            if ccod is not None and reverse_geocode_bulk is not None:
                c = geom.centroid
                lon, lat = to_wgs.transform(c.x, c.y)
                pending_idx.append(i)
                pending_lonlat.append((lon, lat))

    # ONE batched reverse-geocode for every pending plot, then CCOD lookup.
    if pending_idx:
        try:
            postcodes = reverse_geocode_bulk(pending_lonlat)
        except Exception as e:
            print(f"  classify: bulk reverse-geocode failed ({e}); ownership stays Government/Other.")
            postcodes = [None] * len(pending_idx)
        for k, i in enumerate(pending_idx):
            pc = postcodes[k] if k < len(postcodes) else None
            cls = ccod.classify_postcode(pc) if pc else None
            if cls:
                owners[i] = cls
                sources[i] = sources[i].rsplit("owner:", 1)[0] + f"owner: CCOD {pc} -> {cls}"

    out["land_type"] = types
    out["owner_class"] = owners
    out["class_source"] = sources
    return out


# ============================================================================
# OPTIONAL: HM Land Registry CCOD/OCOD (the Corporate tier)
# ============================================================================
_GOV_CATEGORY_WORDS = ("council", "authority", "government", "crown", "secretary of state",
                       "police", "fire", "health", "nhs", "ministry", "department", "duchy")


def _proprietor_class(category, name=""):
    """Map a CCOD/OCOD proprietorship category (+ proprietor name) to 'Government' or 'Corporate'.
    Per your rules: charities/housing associations/companies/overseas -> Corporate; councils/Crown/
    public bodies -> Government."""
    blob = f"{category} {name}".lower()
    if any(w in blob for w in _GOV_CATEGORY_WORDS):
        return "Government"
    return "Corporate"


class CCOD:
    """A lightweight index of HM Land Registry CCOD/OCOD rows, keyed by postcode. classify_postcode()
    returns 'Government'/'Corporate'/None. Government wins if a postcode holds both (conservative)."""

    def __init__(self):
        self._by_postcode = {}   # normalised postcode -> 'Government' | 'Corporate'

    @staticmethod
    def _norm_pc(pc):
        return re.sub(r"\s+", "", (pc or "").upper())

    def load(self, *paths):
        """Load one or more CCOD/OCOD CSV files. Uses the 'Postcode' + 'Proprietorship Category (1)'
        (and 'Proprietor Name (1)') columns; tolerant of missing columns."""
        n = 0
        for path in paths:
            if not path or not os.path.exists(path):
                continue
            with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
                reader = csv.DictReader(f)
                fields = {c.lower(): c for c in (reader.fieldnames or [])}
                pc_c = next((fields[k] for k in fields if k == "postcode"), None)
                cat_c = next((fields[k] for k in fields if k.startswith("proprietorship category")), None)
                nm_c = next((fields[k] for k in fields if k.startswith("proprietor name")), None)
                if not pc_c:
                    continue
                for row in reader:
                    pc = self._norm_pc(row.get(pc_c))
                    if not pc:
                        continue
                    cls = _proprietor_class(row.get(cat_c, "") if cat_c else "",
                                            row.get(nm_c, "") if nm_c else "")
                    prev = self._by_postcode.get(pc)
                    # Government beats Corporate if a postcode has both.
                    self._by_postcode[pc] = "Government" if "Government" in (prev, cls) else cls
                    n += 1
        print(f"  CCOD/OCOD: indexed {len(self._by_postcode)} postcode(s) from {n} row(s)")
        return self

    def classify_postcode(self, postcode):
        return self._by_postcode.get(self._norm_pc(postcode))


def postcodesio_bulk_reverse_geocode():
    """Return a function that takes a LIST of (lon, lat) WGS84 points and returns a same-length list of
    postcodes (or None), using the FREE postcodes.io BULK reverse-geocode endpoint -- up to 100 points
    per HTTP request (no key, no account). This is what makes the Corporate tier fast: a 200-plot
    search does ~2 requests, not 200. Best-effort; never raises."""
    import requests

    def _bulk(lonlats):
        out = [None] * len(lonlats)
        for start in range(0, len(lonlats), 100):
            chunk = lonlats[start:start + 100]
            body = {"geolocations": [{"longitude": lo, "latitude": la, "limit": 1, "radius": 1000}
                                     for lo, la in chunk]}
            try:
                r = requests.post("https://api.postcodes.io/postcodes", json=body, timeout=30)
                r.raise_for_status()
                for k, item in enumerate(r.json().get("result") or []):
                    res = (item or {}).get("result") or []
                    if res:
                        out[start + k] = res[0].get("postcode")
            except Exception:
                pass          # leave this chunk as None; ownership just stays Other for those
        return out
    return _bulk
