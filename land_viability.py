"""
land_viability.py -- given a point (lat/long) + radius from your map interface, classify the land in
that area into BUILT-ON, VEGETATION (trees/bushes) and CANDIDATE (undeveloped) land, and export each
as GeoJSON your map can overlay.

This is the GIS data-overlay approach (NOT scraping Google imagery, which its terms forbid and which
would be far less accurate). It combines authoritative, licensed-for-this-use UK datasets:

  * BUILT-ON  -- building-footprint polygons from the OS Features API (WFS), using the free, open
                 OS Open Zoomstack local-buildings layer (product 'zoomstack_local_buildings'),
                 fetched via Ordnance Survey's own osdatahub package. Needs a free OS Data Hub API
                 key with the "OS Features API" added to your project. If you have no OS key, an
                 OpenStreetMap/Overpass fallback is used automatically (free, no key).
  * VEGETATION-- derived from Environment Agency National LIDAR Programme (open data, 1 m): the
                 canopy height = First-Return DSM minus DTM. Anything taller than CANOPY_MIN_HEIGHT_M
                 that isn't a building is treated as trees/bushes. Fetched via the EA WCS services.
  * CANDIDATE -- computed as the residual: the search area with buildings and vegetation removed.
                 (Water/roads/constraints like Green Belt, AONB, flood zones, TPOs are NOT subtracted
                 yet -- see the note at the bottom; those are the real "viability" filters to add
                 next, and you already have the Kent designation layers for them.)

Endpoints were verified live (Aug 2026), not guessed:
  OS Features API (WFS): https://api.os.uk/features/v1/wfs  (accessed via the osdatahub package; product 'zoomstack_local_buildings')
  EA LIDAR DTM WCS: https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs
  EA LIDAR DSM WCS: https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs

SETUP
    pip install geopandas shapely rasterio owslib pyproj requests numpy osdatahub aiohttp --break-system-packages
    # OS key (recommended): sign up free at osdatahub.os.uk -> create a project -> add the "OS
    #   Features API" to it -> copy the project API key. (NOT "OS Net"=GNSS, NOT NGD.)
    export OS_API_KEY="your-key"      # optional; without it the OSM buildings fallback is used

USAGE
    # command line (what your map's "search this area" button would shell out to, or replicate):
    python3 land_viability.py --lat 51.2786 --lon 0.5217 --radius 500 --out ./out
    # or import and call from your backend:
    from land_viability import analyze_area
    result = analyze_area(lat=51.2786, lon=0.5217, radius_m=500, out_dir="./out")
    # result -> {"buildings": gdf, "vegetation": gdf, "candidate": gdf, "stats": {...}}  (all EPSG:4326)

NOTE ON OS MASTERMAP: OS MasterMap Topography (most detailed buildings) is also available through the
OS Features API, but it's Premium data (consumes your OS Data Hub premium allowance and must be added
to the project). This script defaults to the free, open OS Open Zoomstack buildings so it works on a
plain free key. To use MasterMap instead once you have premium access, change OS_FEATURES_PRODUCT
below to the MasterMap Topography buildings product -- the rest of the pipeline is unchanged. Zoomstack
buildings are slightly generalised (good for area overviews; MasterMap is sharper at plot level).
"""
import argparse
import io
import json
import os
import sys

import requests

# Heavy geospatial deps -- import with a clear message if missing.
try:
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from shapely.geometry import shape, mapping, Point, box
    from shapely.ops import unary_union
    from shapely.geometry import Polygon, MultiPolygon
    from pyproj import Transformer
    from owslib.wcs import WebCoverageService
    from osdatahub import Extent, FeaturesAPI
except ImportError as e:
    sys.exit(f"Missing a geospatial dependency ({e}). Install them with:\n"
             "  pip install geopandas shapely rasterio owslib pyproj requests numpy osdatahub aiohttp --break-system-packages")

# ============================================================================
# CONFIG
# ============================================================================
OS_API_KEY = os.environ.get("OS_API_KEY", "")            # free OS Data Hub key; blank -> OSM fallback
W3W_API_KEY = os.environ.get("W3W_API_KEY", "")          # free what3words key; needed only for 3-word inputs
OS_FEATURES_PRODUCT = "zoomstack_local_buildings"        # free OS Open Zoomstack buildings via OS Features API
OS_FEATURES_MAX = 10000                                  # max buildings to pull per area (paged internally)

EA_DTM_WCS = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-terrain-model-dtm-1m/wcs"
EA_DSM_WCS = "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs"

CANOPY_MIN_HEIGHT_M = 1.0        # >= this many metres tall (and not a building) = trees/bushes. 1m keeps bushes; raise to ~2.5 for trees only.
MIN_FEATURE_AREA_M2 = 9.0        # drop slivers/noise smaller than this (m^2)
DEFAULT_RADIUS_M = 500
AOI_SHAPE = "circle"             # "circle" or "square"
BNG = "EPSG:27700"               # British National Grid -- OS/EA native CRS (metres)
WGS84 = "EPSG:4326"

_to_bng = Transformer.from_crs(WGS84, BNG, always_xy=True)
_to_wgs = Transformer.from_crs(BNG, WGS84, always_xy=True)


# ============================================================================
# AREA OF INTEREST
# ============================================================================
def build_aoi_bng(lat, lon, radius_m, shape_kind=AOI_SHAPE):
    """Return the area-of-interest polygon in British National Grid (metres)."""
    x, y = _to_bng.transform(lon, lat)
    centre = Point(x, y)
    if shape_kind == "square":
        return box(x - radius_m, y - radius_m, x + radius_m, y + radius_m)
    return centre.buffer(radius_m)  # circle (metres, because BNG is metric)


# ============================================================================
# BUILDINGS
# ============================================================================
def fetch_os_buildings(aoi_bng):
    """OS building polygons from the OS Features API (WFS) via the osdatahub package, using the free,
    open OS Open Zoomstack local-buildings layer. osdatahub handles WFS paging + axis order and
    returns GeoJSON in the extent's CRS -- we request EPSG:27700, so results come back in BNG.
    Returns an EPSG:27700 GeoDataFrame (possibly empty)."""
    minx, miny, maxx, maxy = aoi_bng.bounds
    extent = Extent.from_bbox((minx, miny, maxx, maxy), BNG)
    api = FeaturesAPI(OS_API_KEY, OS_FEATURES_PRODUCT, extent)
    fc = api.query(limit=OS_FEATURES_MAX)                 # GeoJSON FeatureCollection, geometries in BNG
    feats = fc.get("features", []) if isinstance(fc, dict) else list(getattr(fc, "features", []))
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    gdf = gpd.GeoDataFrame.from_features(feats, crs=BNG)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    return gdf


def fetch_osm_buildings(aoi_bng):
    """Free, no-key fallback: OpenStreetMap building footprints via the Overpass API. Returns an
    EPSG:27700 GeoDataFrame. Coverage is good in the UK but not as complete/authoritative as OS."""
    minx, miny, maxx, maxy = aoi_bng.bounds
    lon0, lat0 = _to_wgs.transform(minx, miny)
    lon1, lat1 = _to_wgs.transform(maxx, maxy)
    s, w, n, e = min(lat0, lat1), min(lon0, lon1), max(lat0, lat1), max(lon0, lon1)
    q = f'[out:json][timeout:60];(way["building"]({s},{w},{n},{e});relation["building"]({s},{w},{n},{e}););out geom;'
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": q}, timeout=90)
    r.raise_for_status()
    polys = []
    for el in r.json().get("elements", []):
        if el.get("type") == "way" and el.get("geometry"):
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(ring) >= 4:
                polys.append(Polygon(ring))
    if not polys:
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    return gpd.GeoDataFrame(geometry=polys, crs=WGS84).to_crs(BNG)


def get_buildings(aoi_bng):
    if OS_API_KEY:
        try:
            gdf = fetch_os_buildings(aoi_bng)
            print(f"  buildings: {len(gdf)} from OS Features API ({OS_FEATURES_PRODUCT})")
            return gdf, f"OS Features API ({OS_FEATURES_PRODUCT})"
        except Exception as ex:
            print(f"  OS Features buildings failed ({ex}); falling back to OpenStreetMap.")
    gdf = fetch_osm_buildings(aoi_bng)
    print(f"  buildings: {len(gdf)} from OpenStreetMap (no OS key set)" if not OS_API_KEY
          else f"  buildings: {len(gdf)} from OpenStreetMap (fallback)")
    return gdf, "OpenStreetMap"


# ============================================================================
# VEGETATION (EA LiDAR canopy height = First-Return DSM - DTM)
# ============================================================================
def _wcs_geotiff(wcs_url, aoi_bng):
    """GetCoverage a 1 m GeoTIFF over the AOI bbox from an EA LiDAR WCS. Returns (array, transform,
    nodata). Tries WCS 1.0.0 (simple bbox model) then 2.0.1. Raises on total failure."""
    minx, miny, maxx, maxy = [int(round(v)) for v in aoi_bng.bounds]
    last_err = None
    for version in ("1.0.0", "2.0.1"):
        try:
            wcs = WebCoverageService(wcs_url, version=version)
            cov_id = list(wcs.contents)[0]
            fmt = _pick_geotiff_format(wcs.contents[cov_id])
            if version == "1.0.0":
                resp = wcs.getCoverage(identifier=cov_id, bbox=(minx, miny, maxx, maxy),
                                       crs=BNG, format=fmt, resx=1, resy=1)
            else:
                resp = wcs.getCoverage(identifier=[cov_id], format=fmt,
                                       subsets=[("E", minx, maxx), ("N", miny, maxy)], crs=BNG)
            data = resp.read()
            with rasterio.io.MemoryFile(data) as mf, mf.open() as ds:
                return ds.read(1).astype("float32"), ds.transform, ds.nodata
        except Exception as ex:
            last_err = ex
            continue
    raise RuntimeError(f"WCS fetch failed for {wcs_url}: {last_err}")


def _pick_geotiff_format(coverage):
    fmts = getattr(coverage, "supportedFormats", None) or []
    for f in fmts:
        if "tif" in f.lower() or "geotiff" in f.lower():
            return f
    return "GeoTIFF"


def fetch_ea_vegetation(aoi_bng, canopy_min=CANOPY_MIN_HEIGHT_M):
    """Vegetation polygons (EPSG:27700) from EA LiDAR canopy height (First-Return DSM - DTM) >=
    canopy_min. Returns a GeoDataFrame; raises if the LiDAR services can't be reached."""
    dsm, dsm_tf, dsm_nd = _wcs_geotiff(EA_DSM_WCS, aoi_bng)
    dtm, dtm_tf, dtm_nd = _wcs_geotiff(EA_DTM_WCS, aoi_bng)
    rows = min(dsm.shape[0], dtm.shape[0])
    cols = min(dsm.shape[1], dtm.shape[1])
    dsm, dtm = dsm[:rows, :cols], dtm[:rows, :cols]
    valid = np.ones((rows, cols), dtype=bool)
    for arr, nd in ((dsm, dsm_nd), (dtm, dtm_nd)):
        if nd is not None:
            valid &= arr != nd
        valid &= np.isfinite(arr)
    canopy = np.where(valid, dsm - dtm, -9999.0)
    veg_mask = (canopy >= canopy_min) & valid
    if not veg_mask.any():
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    geoms = [shape(g) for g, v in raster_shapes(veg_mask.astype("uint8"), mask=veg_mask, transform=dsm_tf) if v == 1]
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    return gpd.GeoDataFrame(geometry=geoms, crs=BNG)


# ============================================================================
# PIPELINE
# ============================================================================
def _clean(gdf, aoi_bng):
    """Clip to AOI, drop empties/slivers, dissolve to one multipolygon-ish set."""
    if gdf is None or gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    clipped = gpd.clip(gdf, aoi_bng)
    clipped = clipped[clipped.geometry.notnull() & ~clipped.geometry.is_empty]
    clipped = clipped[clipped.geometry.area >= MIN_FEATURE_AREA_M2]
    return clipped


def analyze_area(lat, lon, radius_m=DEFAULT_RADIUS_M, shape_kind=AOI_SHAPE, out_dir=None,
                 canopy_min=CANOPY_MIN_HEIGHT_M):
    """Classify land in the AOI into buildings / vegetation / candidate. Returns a dict of EPSG:4326
    GeoDataFrames + area stats, and (if out_dir given) writes buildings/vegetation/candidate GeoJSON."""
    print(f"Analysing {radius_m} m {shape_kind} around ({lat}, {lon}) ...")
    aoi = build_aoi_bng(lat, lon, radius_m, shape_kind)
    aoi_area = aoi.area

    buildings, b_src = get_buildings(aoi)
    buildings = _clean(buildings, aoi)

    veg_src = "EA LIDAR (DSM-DTM canopy)"
    try:
        vegetation = fetch_ea_vegetation(aoi, canopy_min)
        vegetation = _clean(vegetation, aoi)
        # A building roof shows up as "tall" in the DSM too -- remove building footprints from veg.
        if not buildings.empty and not vegetation.empty:
            vegetation = gpd.GeoDataFrame(
                geometry=[vegetation.union_all().difference(buildings.union_all())], crs=BNG)
            vegetation = _clean(vegetation, aoi)
        print(f"  vegetation: {len(vegetation)} patch(es) from EA LiDAR canopy >= {canopy_min} m")
    except Exception as ex:
        print(f"  vegetation: UNAVAILABLE ({ex}). Candidate land will only exclude buildings.")
        vegetation = gpd.GeoDataFrame(geometry=[], crs=BNG)
        veg_src = "unavailable"

    # Candidate = AOI minus buildings minus vegetation.
    taken = [g for g in (
        buildings.union_all() if not buildings.empty else None,
        vegetation.union_all() if not vegetation.empty else None) if g is not None]
    candidate_geom = aoi.difference(unary_union(taken)) if taken else aoi
    candidate = _clean(gpd.GeoDataFrame(geometry=[candidate_geom], crs=BNG), aoi)

    def _pct(gdf):
        return float(round(100.0 * (gdf.geometry.area.sum() / aoi_area), 1)) if not gdf.empty else 0.0

    stats = {"centre": [float(lat), float(lon)], "radius_m": float(radius_m), "shape": shape_kind,
             "aoi_area_m2": float(round(aoi_area, 1)),
             "built_pct": _pct(buildings), "vegetation_pct": _pct(vegetation),
             "candidate_pct": _pct(candidate),
             "buildings_source": b_src, "vegetation_source": veg_src}

    out = {"buildings": buildings.to_crs(WGS84), "vegetation": vegetation.to_crs(WGS84),
           "candidate": candidate.to_crs(WGS84), "aoi": gpd.GeoDataFrame(geometry=[aoi], crs=BNG).to_crs(WGS84),
           "stats": stats}

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name in ("buildings", "vegetation", "candidate", "aoi"):
            path = os.path.join(out_dir, f"{name}.geojson")
            g = out[name]
            (g if not g.empty else gpd.GeoDataFrame(geometry=[], crs=WGS84)).to_file(path, driver="GeoJSON")
        with open(os.path.join(out_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  wrote buildings/vegetation/candidate/aoi .geojson + stats.json to {out_dir}/")

    print(f"  RESULT: built {stats['built_pct']}%  vegetation {stats['vegetation_pct']}%  "
          f"candidate {stats['candidate_pct']}%")
    return out


# ============================================================================
# GEOCODING -- turn a platform search (address / coordinates / what3words) into lat/long
# ============================================================================
import re as _re

_COORD_RE = _re.compile(r"^\s*(-?\d{1,3}(?:\.\d+)?)\s*[, ]\s*(-?\d{1,3}(?:\.\d+)?)\s*$")
_W3W_RE = _re.compile(r"^/{0,3}([^.\s/]+)\.([^.\s/]+)\.([^.\s/]+)$")


def _geocode_os_names(query, os_key):
    """Postcode / place-name geocode via the OS Names API (returns British National Grid). None on miss."""
    if not os_key:
        return None
    r = requests.get("https://api.os.uk/search/names/v1/find",
                     params={"query": query, "key": os_key, "maxresults": 1}, timeout=30)
    r.raise_for_status()
    results = r.json().get("results") or []
    if not results:
        return None
    g = results[0].get("GAZETTEER_ENTRY", {})
    if "GEOMETRY_X" not in g or "GEOMETRY_Y" not in g:
        return None
    lon, lat = _to_wgs.transform(float(g["GEOMETRY_X"]), float(g["GEOMETRY_Y"]))
    return lat, lon


def _geocode_nominatim(query):
    """Full-address fallback via OpenStreetMap Nominatim (free; be gentle -- <=1 req/s). None on miss."""
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "gb"},
                     headers={"User-Agent": "land-viability/1.0 (planning app)"}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if not js:
        return None
    return float(js[0]["lat"]), float(js[0]["lon"])


def _geocode_what3words(words, w3w_key):
    """3-word address -> coordinates via the what3words API. Needs a free what3words key."""
    if not w3w_key:
        raise ValueError("what3words input needs a what3words API key (set W3W_API_KEY).")
    r = requests.get("https://api.what3words.com/v3/convert-to-coordinates",
                     params={"words": words, "key": w3w_key}, timeout=30)
    r.raise_for_status()
    c = r.json().get("coordinates")
    if not c:
        raise ValueError(f"what3words could not resolve '{words}'.")
    return float(c["lat"]), float(c["lng"])


def resolve_location(query, os_key=None, w3w_key=None):
    """Turn a platform search string into (lat, lon) in WGS84. Accepts:
      * coordinates: "51.2786, 0.5217" (assumed lat,lon; auto-swaps if clearly lon,lat for GB)
      * what3words:  "filled.count.soap" or "///filled.count.soap"
      * address / postcode / place name: geocoded via OS Names, then OSM Nominatim as fallback.
    Raises ValueError if nothing resolves."""
    os_key = OS_API_KEY if os_key is None else os_key
    w3w_key = W3W_API_KEY if w3w_key is None else w3w_key
    q = (query or "").strip()
    if not q:
        raise ValueError("empty location query")

    m = _COORD_RE.match(q)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        # Assume lat,lon; if it's obviously lon,lat for GB (a is a plausible lon, b a plausible lat), swap.
        if not (49 <= a <= 61) and (49 <= b <= 61):
            a, b = b, a
        return a, b

    w = _W3W_RE.match(q)
    if w:
        return _geocode_what3words(q.lstrip("/"), w3w_key)

    for geocoder in (lambda: _geocode_os_names(q, os_key), lambda: _geocode_nominatim(q)):
        try:
            hit = geocoder()
        except Exception:
            hit = None
        if hit:
            return hit
    raise ValueError(f"Could not resolve location: {query!r}")


def analyze_query(query, radius_m=DEFAULT_RADIUS_M, shape_kind=AOI_SHAPE, canopy_min=CANOPY_MIN_HEIGHT_M,
                  os_key=None, w3w_key=None, out_dir=None):
    """Platform entry point: resolve a search string (address/coords/what3words) then classify the
    area. Returns a JSON-serialisable dict: {query, resolved:{lat,lon}, radius_m, stats, geojson}
    where geojson is a single FeatureCollection with each feature tagged properties.class in
    {building, vegetation, candidate, aoi} -- ready to style on your web map."""
    lat, lon = resolve_location(query, os_key, w3w_key)
    out = analyze_area(lat, lon, radius_m, shape_kind, out_dir=out_dir, canopy_min=canopy_min)
    feats = []
    for cls, key in (("building", "buildings"), ("vegetation", "vegetation"),
                     ("candidate", "candidate"), ("aoi", "aoi")):
        for geom in out[key].geometry:
            if geom is not None and not geom.is_empty:
                feats.append({"type": "Feature", "properties": {"class": cls}, "geometry": mapping(geom)})
    return {"query": query, "resolved": {"lat": lat, "lon": lon}, "radius_m": radius_m,
            "stats": out["stats"], "geojson": {"type": "FeatureCollection", "features": feats}}


def main():
    ap = argparse.ArgumentParser(description="Classify built / vegetation / candidate land in an area.")
    ap.add_argument("--query", help='address, "lat,lon", or what3words (e.g. "filled.count.soap")')
    ap.add_argument("--lat", type=float, help="latitude (WGS84) -- alternative to --query")
    ap.add_argument("--lon", type=float, help="longitude (WGS84) -- alternative to --query")
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS_M, help="radius (metres) / half-side for square")
    ap.add_argument("--shape", choices=["circle", "square"], default=AOI_SHAPE)
    ap.add_argument("--canopy", type=float, default=CANOPY_MIN_HEIGHT_M, help="min vegetation height (m)")
    ap.add_argument("--out", default="./land_viability_out", help="output folder for GeoJSON")
    a = ap.parse_args()
    if a.query:
        res = analyze_query(a.query, a.radius, a.shape, a.canopy, out_dir=a.out)
        print(f"  resolved {a.query!r} -> {res['resolved']}")
    elif a.lat is not None and a.lon is not None:
        analyze_area(a.lat, a.lon, a.radius, a.shape, a.out, a.canopy)
    else:
        ap.error("provide either --query, or both --lat and --lon")


if __name__ == "__main__":
    main()
