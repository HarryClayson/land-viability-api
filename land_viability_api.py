"""
land_viability_api.py -- a tiny web service so your PLATFORM (not a terminal) drives the land
classifier. Your map's search box POSTs/GETs a query (address, "lat,lon", or what3words) + a radius;
this resolves it to a location, runs the classifier, and returns GeoJSON your map can draw.

This is the piece that means a user never touches the terminal: your frontend calls this HTTP
endpoint, gets back the built/vegetation/candidate polygons, and styles them on the map.

SETUP
    pip install flask flask-cors --break-system-packages
    # plus the land_viability.py dependencies (geopandas, rasterio, owslib, pyproj, osdatahub, aiohttp, ...)
    export OS_API_KEY="your-os-key"       # for OS buildings + OS Names geocoding (optional; OSM fallback otherwise)
    export W3W_API_KEY="your-w3w-key"     # only needed if users search by what3words
    python3 land_viability_api.py         # serves on http://127.0.0.1:8000

ENDPOINT
    GET  /analyze?query=<address|lat,lon|what3words>&radius=<metres>[&shape=circle|square][&canopy=<m>]
    POST /analyze   { "query": "...", "radius": 500 }
    ->  { "query", "resolved": {lat,lon}, "radius_m", "stats": {...},
          "geojson": FeatureCollection with each feature's properties.class in
                     {building, vegetation, candidate, aoi, land_plot} }

    Land plot outlines: set the PARCELS_FILE env var to a prepared HM Land Registry INSPIRE parcels
    file covering your area (see note below) and the response also includes properties.class ==
    "land_plot" polygons -- every registered land parcel in the searched area. Clients can toggle per
    request with &plots=0 / &plots=1. Off automatically if PARCELS_FILE isn't set.

    GET  /health  -> {"ok": true}

Example your frontend would call:
    /analyze?query=Tonbridge&radius=500
    /analyze?query=51.2,0.26&radius=800&shape=square
    /analyze?query=filled.count.soap&radius=300

NOTE: this dev server (Flask's built-in) is fine for testing. For production put it behind gunicorn/
uwsgi + a reverse proxy, and restrict CORS origins (below) to your platform's domain.
"""
import os

from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except ImportError:
    CORS = None

from land_viability import analyze_query, DEFAULT_RADIUS_M, AOI_SHAPE, CANOPY_MIN_HEIGHT_M

# Path to a prepared HM Land Registry INSPIRE parcels file covering your whole area (all the councils
# you serve, merged into ONE spatially-indexed file). Leave unset to disable plot outlines.
# IMPORTANT: convert the per-council INSPIRE GML downloads into a single GeoPackage or FlatGeobuf with
# a spatial index first, e.g.  ogr2ogr kent_parcels.gpkg Canterbury.gml  (then -append the rest).
# A .gpkg/.fgb is read by BOUNDING BOX per request (fast, low memory); querying raw .gml scans the
# whole file each time and will be slow. Then: export PARCELS_FILE=/path/to/kent_parcels.gpkg
PARCELS_FILE = os.environ.get("PARCELS_FILE", "")
# Startup diagnostic: setting PARCELS_FILE is not enough -- the FILE itself must exist on the server.
# This prints to the Render logs the moment the service boots, so you can see at a glance whether the
# land-plot layer will work (and /health reports the same).
PARCELS_OK = bool(PARCELS_FILE) and os.path.exists(PARCELS_FILE)
if not PARCELS_FILE:
    print("[land-plots] PARCELS_FILE not set -> land-plot outlines DISABLED.")
elif not PARCELS_OK:
    print(f"[land-plots] PARCELS_FILE is set to {PARCELS_FILE!r} but NO FILE exists there -> land-plot "
          "outlines DISABLED. Deploy the .gpkg to the server at that path (an env var alone is not enough).")
else:
    print(f"[land-plots] using parcels file: {PARCELS_FILE}")

app = Flask(__name__)
if CORS:
    # For production, replace "*" with your platform's domain, e.g. origins=["https://app.example.com"].
    CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health")
def health():
    # Reports whether the land-plot layer is actually usable (file present), so you can check the
    # deploy from a browser: GET /health should show land_plots_enabled: true.
    return jsonify({"ok": True, "land_plots_enabled": PARCELS_OK,
                    "parcels_file_set": bool(PARCELS_FILE),
                    "parcels_file_exists": PARCELS_OK})


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        radius = float(body.get("radius", DEFAULT_RADIUS_M))
        shape = body.get("shape", AOI_SHAPE)
        canopy = float(body.get("canopy", CANOPY_MIN_HEIGHT_M))
        plots = body.get("plots", None)
    else:
        query = (request.args.get("query") or "").strip()
        radius = float(request.args.get("radius", DEFAULT_RADIUS_M))
        shape = request.args.get("shape", AOI_SHAPE)
        canopy = float(request.args.get("canopy", CANOPY_MIN_HEIGHT_M))
        plots = request.args.get("plots", None)

    # Include land-plot outlines when a parcels file is configured, unless the client turned it off.
    include_plots = bool(PARCELS_FILE) and str(plots).lower() not in ("0", "false", "no")
    parcels_path = PARCELS_FILE if include_plots else None

    if not query:
        return jsonify({"error": "missing 'query' (address, 'lat,lon', or what3words)"}), 400
    if shape not in ("circle", "square"):
        return jsonify({"error": "shape must be 'circle' or 'square'"}), 400
    # Guard-rail: cap the radius so a user can't request an enormous area that blows the RAM limit.
    # 1000 m is comfortable on a 512 MB (free) Render instance at LIDAR_RESOLUTION_M=2; raise it once
    # you're on a bigger instance.
    radius = max(50.0, min(radius, 1000.0))

    try:
        result = analyze_query(query, radius_m=radius, shape_kind=shape, canopy_min=canopy,
                               parcels_path=parcels_path)
        return jsonify(result)
    except ValueError as e:            # couldn't geocode the query
        return jsonify({"error": str(e)}), 422
    except Exception as e:             # upstream data source failed etc.
        return jsonify({"error": f"analysis failed: {e}"}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
