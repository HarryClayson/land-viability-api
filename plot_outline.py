"""
plot_outline.py -- attach an approximate LAND PLOT OUTLINE to each application from its geocoded
address, using HM Land Registry INSPIRE Index Polygons (registered freehold parcels; free, OGL).

HONEST SCOPE (read this before relying on it):
  * What you get is the registered FREEHOLD TITLE extent that CONTAINS the geocoded address point --
    a PROXY for "the plot". It is NOT the planning red-line application boundary. The true red-line
    site is only in the submitted location-plan PDF/drawing and is not machine-readable, so it cannot
    be scraped from an address by any means. This is the closest free, address-derived approximation.
  * A registered title can be larger or smaller than the actual application site (gardens split off,
    multiple titles, unregistered land, etc.), and geocoding error can land the point in a neighbour's
    parcel or on the road. Every attached row is flagged with how it matched ('contains' vs 'nearest'
    vs 'none') so you can filter on confidence. Treat the outline + area as indicative, not survey.

DATA (one-time per council): download the council's INSPIRE Index Polygons -- a GML file per local
authority, updated monthly, OGL -- from:
    https://use-land-property-data.service.gov.uk/datasets/inspire
Save each council's file (e.g. Canterbury.gml) somewhere and pass its path below.

USAGE
    import plot_outline as po
    idx = po.load_parcels("Canterbury.gml")                 # load once per council (can be slow/big)
    po.parcel_for_point(51.2798, 1.0755, idx)               # -> dict of plot fields for one point
    po.attach_to_csv("Canterbury/decision_reasons_enriched.csv", "Canterbury.gml")   # whole CSV

  or from the shell:
    python3 plot_outline.py --parcels Canterbury.gml --in-csv Canterbury/decision_reasons_enriched.csv

Adds these columns to each row (blank when there's no lat/lon or no match):
    plot_inspire_id     -- the INSPIRE polygon id of the matched parcel
    plot_area_m2        -- parcel area in square metres (computed in British National Grid)
    plot_match          -- 'contains' (point provably inside the parcel) or 'none'. By default only
                           'contains' matches are attached (see ACCURACY POLICY below); 'nearest' only
                           appears if you deliberately re-enable the fallback via NEAREST_TOLERANCE_M.
    plot_outline_geojson-- the parcel boundary as a GeoJSON geometry string (WGS84 lon/lat), ready to
                           drop onto a map as an overlay
    plot_outline_wkt    -- the same boundary as WKT (WGS84), for spreadsheets / PostGIS

Needs: geopandas + shapely + pyproj (pip install geopandas shapely pyproj --break-system-packages).
Reading .gml uses GDAL/fiona, which geopandas bundles.
"""
import argparse
import csv
import json
import os
import sys

# Raise Python's default CSV field-size cap (131072 chars) so reading never trips
# "_csv.Error: field larger than field limit" on a very long field -- e.g. a big/complex parcel's
# GeoJSON/WKT geometry that a previous --plots run already wrote, or a long officer-report text.
# Done defensively because some platforms reject sys.maxsize.
_csv_max = sys.maxsize
while True:
    try:
        csv.field_size_limit(_csv_max)
        break
    except OverflowError:
        _csv_max //= 10

# INSPIRE parcels are in British National Grid (EPSG:27700). We match/measure in BNG (metres) and
# output geometry in WGS84 (EPSG:4326) so it drops straight onto a web map.
BNG = "EPSG:27700"
WGS84 = "EPSG:4326"

# ACCURACY POLICY -- "contains-only" (the hardwired default).
# We ONLY attach an outline when the geocoded point falls INSIDE a parcel ('contains'). This is the
# high-confidence subset: for those rows the attached polygon provably occupies the point's location.
# When the point is inside no parcel we attach NOTHING (plot_match='none') rather than guess the
# nearest title -- because a "nearest" guess can silently be a neighbour's plot or one across the road,
# which is exactly the inaccuracy you don't want.
#
# NEAREST_TOLERANCE_M > 0 re-enables the old nearest-parcel fallback (more rows filled, but some of
# those fills will be wrong). Leave it at 0 unless you deliberately want coverage over accuracy.
#
# NOTE this filter guarantees the point is INSIDE the attached parcel; it cannot fix a geocode that
# lands in the WRONG parcel. Geocoding precision is the other accuracy lever -- set OS_API_KEY and
# feed the postcode so points land on the right building, not a street/postcode centroid.
NEAREST_TOLERANCE_M = 0        # 0 = contains-only (accurate); >0 = allow nearest-parcel fallback (looser)

PLOT_FIELDS = ["plot_inspire_id", "plot_area_m2", "plot_match", "plot_outline_geojson", "plot_outline_wkt"]
_EMPTY_PLOT = {k: "" for k in PLOT_FIELDS}


def _inspire_id_column(gdf):
    """Find the INSPIRE id column across GML/GeoJSON variants ('INSPIREID', 'inspireId', 'gml_id')."""
    for cand in ("INSPIREID", "inspireId", "INSPIRE_ID", "gml_id", "fid", "OBJECTID"):
        for col in gdf.columns:
            if col.lower() == cand.lower():
                return col
    return None


def load_parcels(path):
    """Load a council's INSPIRE parcels (GML/GeoJSON/shp/…) into a spatially-indexed structure in BNG.
    Returns a dict you pass to parcel_for_point(). Loading a full council file can take a while and a
    lot of RAM -- do it ONCE and reuse the returned index for every application in that council."""
    import geopandas as gpd
    if not os.path.exists(path):
        # FileNotFoundError (not SystemExit) so a web server calling this degrades gracefully instead
        # of the whole process exiting.
        raise FileNotFoundError(f"INSPIRE parcels file not found: {os.path.abspath(path)}. "
                                f"Download it from "
                                f"https://use-land-property-data.service.gov.uk/datasets/inspire")
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(BNG)          # INSPIRE ships as BNG; assume it if the file omits the CRS
    gdf = gdf.to_crs(BNG)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].reset_index(drop=True)
    id_col = _inspire_id_column(gdf)
    _ = gdf.sindex                       # build the spatial index up front
    print(f"loaded {len(gdf)} parcels from {path} (id column: {id_col})")
    return {"gdf": gdf, "id_col": id_col}


def parcel_for_point(lat, lon, parcels):
    """Match one (lat, lon) to its containing INSPIRE parcel (or the nearest within tolerance).
    Returns a dict of the PLOT_FIELDS (all blank on no match). Never raises."""
    out = dict(_EMPTY_PLOT)
    try:
        from shapely.geometry import Point, mapping
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer
        gdf = parcels["gdf"]
        id_col = parcels["id_col"]
        to_bng = Transformer.from_crs(WGS84, BNG, always_xy=True)
        x, y = to_bng.transform(lon, lat)
        pt = Point(x, y)

        # candidate parcels via the spatial index, then exact containment
        cand_idx = list(gdf.sindex.query(pt, predicate="intersects"))
        match_kind, geom, row = None, None, None
        for i in cand_idx:
            g = gdf.geometry.iloc[i]
            if g.covers(pt):
                match_kind, geom, row = "contains", g, gdf.iloc[i]
                break
        if geom is None and NEAREST_TOLERANCE_M > 0:
            # nearest parcel within tolerance (geocode often lands just outside the title)
            near_idx = list(gdf.sindex.nearest(pt, max_distance=NEAREST_TOLERANCE_M, return_all=False))
            # sindex.nearest returns [[input_idx],[geom_idx]] -> take the geom index if present
            gi = near_idx[1][0] if near_idx and len(near_idx) == 2 and len(near_idx[1]) else None
            if gi is not None:
                match_kind, geom, row = "nearest", gdf.geometry.iloc[gi], gdf.iloc[gi]
        if geom is None:
            out["plot_match"] = "none"
            return out

        out["plot_match"] = match_kind
        out["plot_area_m2"] = round(geom.area)                       # BNG -> already m²
        if id_col:
            out["plot_inspire_id"] = str(row[id_col])
        # geometry out in WGS84 for mapping
        to_wgs = Transformer.from_crs(BNG, WGS84, always_xy=True)
        geom_wgs = shp_transform(lambda xx, yy, z=None: to_wgs.transform(xx, yy), geom)
        out["plot_outline_geojson"] = json.dumps(mapping(geom_wgs))
        out["plot_outline_wkt"] = geom_wgs.wkt
    except Exception as e:
        out["plot_match"] = f"error: {e}"
    return out


def parcels_in_area(aoi, parcels_path, aoi_crs=BNG):
    """Return EVERY INSPIRE parcel that intersects an area of interest, as a GeoDataFrame in BNG --
    this is the 'outline all land plots in the region' layer for the app's area search.

    `aoi` is a shapely polygon in `aoi_crs` (default BNG; the search AOI from land_viability is BNG).
    Reads ONLY the parcels inside the AOI's bounding box from the file (a bbox filter, so it does NOT
    load the whole council file into memory -- this is what makes it viable on a server), then keeps
    those that actually intersect the AOI. Whole parcels are returned (not clipped to the AOI edge),
    so a plot straddling the search boundary is still shown in full. Empty GeoDataFrame if none."""
    import geopandas as gpd
    if not os.path.exists(parcels_path):
        raise FileNotFoundError(f"INSPIRE parcels file not found: {os.path.abspath(parcels_path)}")
    aoi_bng = gpd.GeoSeries([aoi], crs=aoi_crs).to_crs(BNG).iloc[0]
    minx, miny, maxx, maxy = aoi_bng.bounds
    gdf = gpd.read_file(parcels_path, bbox=(minx, miny, maxx, maxy))   # reads only the bbox's features
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=BNG)
    if gdf.crs is None:
        gdf = gdf.set_crs(BNG)
    gdf = gdf.to_crs(BNG)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.intersects(aoi_bng)].reset_index(drop=True)
    return gdf


def attach_to_csv(in_csv, parcels_path, out_csv=None, lat_field="lat", lon_field="lon"):
    """Attach plot-outline columns to every row of an enriched CSV that has lat/lon. Writes in place
    (out_csv defaults to in_csv). Loads the parcels once and reuses the index for all rows."""
    out_csv = out_csv or in_csv
    parcels = load_parcels(parcels_path)
    with open(in_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows to process")
        return
    n_matched = 0
    for r in rows:
        lat, lon = (r.get(lat_field) or "").strip(), (r.get(lon_field) or "").strip()
        if lat and lon:
            fields = parcel_for_point(float(lat), float(lon), parcels)
            r.update(fields)
            if fields["plot_match"] in ("contains", "nearest"):
                n_matched += 1
        else:
            r.update(_EMPTY_PLOT)
    fieldnames = list(rows[0].keys())
    for c in PLOT_FIELDS:
        if c not in fieldnames:
            fieldnames.append(c)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"attached plot outlines to {n_matched}/{len(rows)} rows -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Attach INSPIRE freehold plot outlines to an enriched CSV.")
    ap.add_argument("--parcels", required=True, help="council INSPIRE Index Polygons file (.gml/.geojson)")
    ap.add_argument("--in-csv", required=True, help="enriched CSV with lat/lon columns")
    ap.add_argument("--out-csv", help="output CSV (default: overwrite --in-csv)")
    a = ap.parse_args()
    attach_to_csv(a.in_csv, a.parcels, out_csv=a.out_csv)
