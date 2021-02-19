from coastsat.SDS_preprocess import *

# Main function to preprocess S1 satellite image
def preprocess_sar(file_name, satname):

    data = gdal.Open(file_name, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    sar_stack = np.stack(bands, 2)

    return sar_stack, georef
