import geopandas as gpd
import pyogrio
from pathlib import Path

gdb_path = Path.home() / "Downloads" / "cmr_admin_boundaries.gdb"
out_dir = Path("csv_export")
out_dir.mkdir(exist_ok=True)

layers = pyogrio.list_layers(gdb_path)

for name, geom_type in layers:
    gdf = gpd.read_file(gdb_path, layer=name)

    # Save the geometry as WKT text so it fits in a CSV
    gdf["geometry"] = gdf.geometry.to_wkt()

    gdf.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"Saved {name}: {gdf.shape[0]} rows")