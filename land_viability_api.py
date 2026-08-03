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
                     {building, vegetation, candidate, aoi} }

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

app = Flask(__name__)
if CORS:
    # For production, replace "*" with your platform's domain, e.g. origins=["https://app.example.com"].
    CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/health")
def health():
    return jsonify({"ok": True})


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        query = (body.get("query") or "").strip()
        radius = float(body.get("radius", DEFAULT_RADIUS_M))
        shape = body.get("shape", AOI_SHAPE)
        canopy = float(body.get("canopy", CANOPY_MIN_HEIGHT_M))
    else:
        query = (request.args.get("query") or "").strip()
        radius = float(request.args.get("radius", DEFAULT_RADIUS_M))
        shape = request.args.get("shape", AOI_SHAPE)
        canopy = float(request.args.get("canopy", CANOPY_MIN_HEIGHT_M))

    if not query:
        return jsonify({"error": "missing 'query' (address, 'lat,lon', or what3words)"}), 400
    if shape not in ("circle", "square"):
        return jsonify({"error": "shape must be 'circle' or 'square'"}), 400
    # Guard-rail: cap the radius so a user can't request an enormous, slow/expensive area.
    radius = max(50.0, min(radius, 2000.0))

    try:
        result = analyze_query(query, radius_m=radius, shape_kind=shape, canopy_min=canopy)
        return jsonify(result)
    except ValueError as e:            # couldn't geocode the query
        return jsonify({"error": str(e)}), 422
    except Exception as e:             # upstream data source failed etc.
        return jsonify({"error": f"analysis failed: {e}"}), 502


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
