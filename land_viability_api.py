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
# OPTION 2 (host-externally): if the parcels file is too big to commit, host it somewhere public (e.g.
# a GitHub Release asset, or Cloudflare R2 / Backblaze B2) and set PARCELS_URL to its download link.
# On boot, if PARCELS_FILE isn't already present, it's downloaded once from PARCELS_URL to PARCELS_FILE.
# Use a writable path for PARCELS_FILE on Render, e.g. /tmp/kent_parcels.gpkg. NOTE: the container disk
# is ephemeral, so it re-downloads on each deploy / cold start -- fine, just adds a little startup time.
PARCELS_URL = os.environ.get("PARCELS_URL", "")


def _ensure_parcels_file():
    """Download PARCELS_FILE from PARCELS_URL once at startup if it isn't already on disk. Streams to a
    temporary '.part' file and renames on success, so an interrupted download never leaves a corrupt
    file behind. Any failure just leaves plots disabled (never crashes the service)."""
    if not PARCELS_FILE or os.path.exists(PARCELS_FILE) or not PARCELS_URL:
        return
    import shutil
    import urllib.request
    d = os.path.dirname(os.path.abspath(PARCELS_FILE))
    os.makedirs(d, exist_ok=True)
    tmp = PARCELS_FILE + ".part"
    print(f"[land-plots] PARCELS_FILE missing; downloading from PARCELS_URL -> {PARCELS_FILE} ...")
    try:
        req = urllib.request.Request(PARCELS_URL, headers={"User-Agent": "land-viability-api"})
        with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        os.replace(tmp, PARCELS_FILE)
        print(f"[land-plots] downloaded {os.path.getsize(PARCELS_FILE) / 1e6:.0f} MB.")
    except Exception as e:
        print(f"[land-plots] download FAILED ({e}); land-plot outlines will be disabled.")
        try:
            os.path.exists(tmp) and os.remove(tmp)
        except OSError:
            pass


_ensure_parcels_file()

# ---- Plot classification (land_type + owner_class on each land plot) --------------------------------
# land_type (OSM land-use) is ALWAYS on and free -- nothing to configure. The extra CORPORATE-ownership
# tier is optional: set CCOD_FILE (and, ideally, OS_API_KEY for reverse-geocoding) to a downloaded HM
# Land Registry CCOD file, and OCOD_FILE for the overseas-companies file. Without them, owner_class is
# Government (from OSM) or Other.
def _download_if_missing(url, path, label):
    """Download `url` to `path` once if the file isn't already present (same pattern as the parcels
    file). Streams to a .part then renames; any failure is non-fatal. No-op if url/path unset."""
    if not path or not url or os.path.exists(path):
        return
    import shutil
    import urllib.request
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    print(f"[{label}] downloading {url} -> {path} ...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "land-viability-api"})
        with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)
        os.replace(tmp, path)
        print(f"[{label}] downloaded {os.path.getsize(path) / 1e6:.0f} MB.")
    except Exception as e:
        print(f"[{label}] download FAILED ({e}).")
        try:
            os.path.exists(tmp) and os.remove(tmp)
        except OSError:
            pass


OS_API_KEY = os.environ.get("OS_API_KEY", "")
CCOD_FILE = os.environ.get("CCOD_FILE", "")
OCOD_FILE = os.environ.get("OCOD_FILE", "")
# Optional: host the (large) CCOD/OCOD CSVs and set these to download them on boot (like PARCELS_URL).
_download_if_missing(os.environ.get("CCOD_URL", ""), CCOD_FILE, "classify-ccod")
_download_if_missing(os.environ.get("OCOD_URL", ""), OCOD_FILE, "classify-ocod")
# Bump this whenever the classifier code changes, so /health tells you at a glance whether the code
# actually running on the server is the latest you pushed (rules out "did the deploy pick it up?").
API_VERSION = "classify-2026-08-08g"

# Kill switch: set CLASSIFY_PLOTS=0 to turn OFF all plot classification (types + ownership) if you ever
# need the area search to be as light as possible.
import land_viability as _lv
if os.environ.get("CLASSIFY_PLOTS", "1").lower() in ("0", "false", "no"):
    _lv.CLASSIFY_PLOTS = False
    print("[classify] DISABLED via CLASSIFY_PLOTS=0.")

# Can the server actually import the classifier module? (If False, classification silently no-ops.)
try:
    import plot_classify as _pc_probe  # noqa: F401
    _CLASSIFY_IMPORTABLE = True
except Exception as _pce:
    _CLASSIFY_IMPORTABLE = False
    print(f"[classify] plot_classify NOT importable ({_pce}) -> classification will not run.")
print(f"[startup] land-viability-api {API_VERSION} | classify_enabled={_lv.CLASSIFY_PLOTS} "
      f"| plot_classify_importable={_CLASSIFY_IMPORTABLE}")

_CCOD = None
_REVERSE_GEOCODE_BULK = None
if CCOD_FILE and os.path.exists(CCOD_FILE):
    try:
        import plot_classify as _pc
        _CCOD = _pc.CCOD().load(CCOD_FILE, OCOD_FILE)
        # Reverse-geocode plots -> postcodes in BULK (postcodes.io, free, ~100 per request) so a big
        # search does a couple of calls, not one per plot. This is what prevents the worker timeout.
        _REVERSE_GEOCODE_BULK = _pc.postcodesio_bulk_reverse_geocode()
        print("[classify] Corporate-ownership tier ON (CCOD loaded; bulk reverse-geocode via postcodes.io).")
    except Exception as e:
        print(f"[classify] CCOD load failed ({e}); owner_class = Government/Other only.")
else:
    print("[classify] land_type ON (free OSM); owner_class = Government/Other "
          "(set CCOD_FILE to add the Corporate tier).")

# Startup diagnostic: setting PARCELS_FILE is not enough -- the FILE itself must exist on the server
# (either committed/mounted, or downloaded via PARCELS_URL above). This prints to the Render logs the
# moment the service boots, so you can see at a glance whether the land-plot layer will work.
PARCELS_OK = bool(PARCELS_FILE) and os.path.exists(PARCELS_FILE)
if not PARCELS_FILE:
    print("[land-plots] PARCELS_FILE not set -> land-plot outlines DISABLED.")
elif not PARCELS_OK:
    print(f"[land-plots] PARCELS_FILE is set to {PARCELS_FILE!r} but NO FILE exists there -> land-plot "
          "outlines DISABLED. Deploy the .gpkg there, or set PARCELS_URL so it downloads on boot "
          "(an env var alone is not enough).")
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
    return jsonify({"ok": True, "version": API_VERSION,
                    "land_plots_enabled": PARCELS_OK,
                    "parcels_file_set": bool(PARCELS_FILE),
                    "parcels_file_exists": PARCELS_OK,
                    "classify_enabled": _lv.CLASSIFY_PLOTS,             # False if CLASSIFY_PLOTS=0
                    "plot_classify_importable": _CLASSIFY_IMPORTABLE,  # False if the module didn't deploy
                    "land_type_classification": _lv.CLASSIFY_PLOTS and _CLASSIFY_IMPORTABLE,
                    "corporate_ownership_tier": _CCOD is not None})


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
                               parcels_path=parcels_path, os_key=OS_API_KEY,
                               ccod=_CCOD, reverse_geocode_bulk=_REVERSE_GEOCODE_BULK)
        return jsonify(result)
    except ValueError as e:            # couldn't geocode the query
        return jsonify({"error": str(e)}), 422
    except Exception as e:             # upstream data source failed etc.
        return jsonify({"error": f"analysis failed: {e}"}), 502


@app.errorhandler(Exception)
def _json_errors(e):
    # Last-resort: return JSON (not a bare HTML 'Internal Server Error') for anything that slips through,
    # so the client always gets a readable message. (A worker that is KILLED for timeout/OOM can't be
    # caught here -- that shows in the Render logs -- but in-process errors will.)
    from werkzeug.exceptions import HTTPException
    code = e.code if isinstance(e, HTTPException) else 500
    return jsonify({"error": f"{type(e).__name__}: {e}"}), code


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 8000)))
