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
    q = (f"[out:json][timeout:{timeout}];("
         f'way["landuse"]({bbox});way["natural"]({bbox});way["water"]({bbox});'
         f'way["waterway"]({bbox});way["leisure"]({bbox});way["amenity"]({bbox});'
         f'way["office"="government"]({bbox}););out tags geom;')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=timeout + 15)
    r.raise_for_status()
    geoms, tag_dicts = [], []
    for el in r.json().get("elements", []):
        pts = el.get("geometry") or []
        if len(pts) < 3:
            continue
        try:
            poly = Polygon([(p["lon"], p["lat"]) for p in pts])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
        except Exception:
            continue
        geoms.append(poly)
        tag_dicts.append(el.get("tags", {}))
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


def classify_owner(plot_geom, osm_gdf, ccod=None, reverse_geocode=None):
    """(owner_class, source) for one plot. Government from OSM public/institutional signal; Corporate/
    Government from optional CCOD (address->postcode->proprietor category); else 'Other'."""
    # 1) OSM public/institutional/military signal
    if osm_gdf is not None and len(osm_gdf):
        try:
            cand = list(osm_gdf.sindex.query(plot_geom, predicate="intersects"))
        except Exception:
            cand = list(range(len(osm_gdf)))
        for i in cand:
            if osm_gdf.iloc[i]["owner_hint"] == "Government":
                try:
                    if plot_geom.intersects(osm_gdf.geometry.iloc[i]):
                        return "Government", "OSM public/institutional use"
                except Exception:
                    pass
    # 2) optional CCOD/OCOD corporate + public match by postcode
    if ccod is not None and reverse_geocode is not None:
        try:
            postcode = reverse_geocode(plot_geom.centroid)
            cls = ccod.classify_postcode(postcode) if postcode else None
            if cls:
                return cls, f"HMLR CCOD/OCOD match ({postcode})"
        except Exception:
            pass
    return "Other", "no corporate/public match"


def classify_plots(plots_gdf, aoi_bng, aoi_crs=BNG, ccod=None, reverse_geocode=None):
    """Add land_type / owner_class / class_source columns to a plots GeoDataFrame (in the plots' CRS).
    Fetches OSM once for the whole AOI, then classifies each plot. Never raises: on any failure the
    plots come back with land_type='Unknown' and owner_class='Other'."""
    out = plots_gdf.copy()
    osm = None
    try:
        osm = fetch_osm_landuse(aoi_bng, aoi_crs)
        print(f"  classify: {len(osm)} OSM land-use feature(s) in the AOI")
    except Exception as e:
        print(f"  classify: OSM land-use unavailable ({e}); land types will be 'Unknown'.")
    types, owners, sources = [], [], []
    for geom in out.geometry:
        lt, lts = classify_land_type(geom, osm)
        oc, ocs = classify_owner(geom, osm, ccod, reverse_geocode)
        types.append(lt)
        owners.append(oc)
        sources.append(f"type: {lts}; owner: {ocs}")
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


def postcodesio_reverse_geocode():
    """Return a function point(BNG Point)->postcode using the FREE postcodes.io reverse lookup (no key,
    no account). Best-effort; never raises. This is the default reverse-geocoder for the Corporate tier."""
    import requests
    from pyproj import Transformer
    to_wgs = Transformer.from_crs(BNG, WGS84, always_xy=True)

    def _rev(point_bng):
        try:
            lon, lat = to_wgs.transform(point_bng.x, point_bng.y)
            r = requests.get("https://api.postcodes.io/postcodes",
                             params={"lon": lon, "lat": lat, "limit": 1, "radius": 1000}, timeout=15)
            r.raise_for_status()
            res = (r.json() or {}).get("result") or []
            if res:
                return res[0].get("postcode")
        except Exception:
            return None
        return None
    return _rev


def os_places_reverse_geocode(os_api_key):
    """Return a function point(BNG shapely Point)->postcode using the OS Places 'nearest' API (needs an
    OS Data Hub key). Returns None-yielding function if no key. Best-effort; never raises."""
    if not os_api_key:
        return None
    import requests

    def _rev(point_bng):
        try:
            r = requests.get("https://api.os.uk/search/places/v1/nearest",
                             params={"point": f"{point_bng.x},{point_bng.y}", "key": os_api_key,
                                     "srs": "BNG", "output_srs": "BNG"}, timeout=20)
            r.raise_for_status()
            results = r.json().get("results") or []
            if results:
                return (results[0].get("DPA") or results[0].get("LPI") or {}).get("POSTCODE")
        except Exception:
            return None
        return None
    return _rev
