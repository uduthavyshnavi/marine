import numpy as np
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------------
# TEST FILE 1: Minor Debris off Visakhapatnam (Bay of Bengal)
# Location: 17.68° N, 83.30° E | 3-Band RGB | Small Debris Field
# Expected Result: SAFE / MODERATE (Blue/Yellow Hotspot)
# ---------------------------------------------------------
width, height = 256, 256
transform1 = from_origin(83.30, 17.68, 0.0001, 0.0001)

# Synthetic ocean image (16-bit Sentinel-2 L2A style, DN values ~1200)
rgb_data1 = np.full((3, height, width), 1200, dtype=np.uint16)
# Insert a small plastic anomaly cluster in center
rgb_data1[:, 120:135, 120:135] = 4500  

with rasterio.open(
    'test_minor_debris_vizag.tif', 'w',
    driver='GTiff', height=height, width=width, count=3,
    dtype=rgb_data1.dtype, crs='EPSG:4326', transform=transform1
) as dst:
    dst.write(rgb_data1)

print("Created: test_minor_debris_vizag.tif (17.68° N, 83.30° E)")

# ---------------------------------------------------------
# TEST FILE 2: Major Spill off Kochi (Arabian Sea / Kerala)
# Location: 9.93° N, 76.26° E | 4-Band RGB+NIR | Large Spill
# Expected Result: CRITICAL THREAT (Red Hexagon Hotspot)
# ---------------------------------------------------------
transform2 = from_origin(76.26, 9.93, 0.0002, 0.0002)

# Synthetic ocean image (16-bit Sentinel-2 L2A style, DN values ~1000)
nir_data2 = np.full((4, height, width), 1000, dtype=np.uint16)
# Insert a large plastic windrow cluster across the center
nir_data2[:, 80:180, 80:180] = 7800  

with rasterio.open(
    'test_major_spill_kerala.tif', 'w',
    driver='GTiff', height=height, width=width, count=4,
    dtype=nir_data2.dtype, crs='EPSG:4326', transform=transform2
) as dst:
    dst.write(nir_data2)

print("Created: test_major_spill_kerala.tif (9.93° N, 76.26° E)")