import numpy as np
import rasterio
from rasterio.transform import from_origin

# ---------------------------------------------------------
# FILE 1: Minor Debris off Visakhapatnam (Bay of Bengal)
# Location: 17.68° N, 83.30° E | 11 Sentinel-2 Bands
# ---------------------------------------------------------
width, height = 256, 256
transform1 = from_origin(83.30, 17.68, 0.0001, 0.0001)

# Create 11-band Sentinel-2 array (uint16 DN values ~1200)
data1 = np.full((11, height, width), 1200, dtype=np.uint16)
data1[:, 120:135, 120:135] = 4500  # Small plastic anomaly cluster

with rasterio.open(
    'test_minor_debris_vizag.tif', 'w',
    driver='GTiff', height=height, width=width, count=11,
    dtype=data1.dtype, crs='EPSG:4326', transform=transform1
) as dst:
    dst.write(data1)

# ---------------------------------------------------------
# FILE 2: Major Spill off Kerala (Arabian Sea)
# Location: 9.93° N, 76.26° E | 11 Sentinel-2 Bands
# ---------------------------------------------------------
transform2 = from_origin(76.26, 9.93, 0.0002, 0.0002)

data2 = np.full((11, height, width), 1000, dtype=np.uint16)
data2[:, 80:180, 80:180] = 7800  # Large plastic spill anomaly cluster

with rasterio.open(
    'test_major_spill_kerala.tif', 'w',
    driver='GTiff', height=height, width=width, count=11,
    dtype=data2.dtype, crs='EPSG:4326', transform=transform2
) as dst:
    dst.write(data2)

print("[OK] Generated 11-band test files: test_minor_debris_vizag.tif and test_major_spill_kerala.tif")