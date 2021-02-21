from coastsat.SDS_preprocess import *

# Main function to preprocess S1 satellite image
def preprocess_sar(file_name, satname):

    data = gdal.Open(file_name, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    sar_stack = np.stack(bands, 2)

    return sar_stack, georef


def get_reference_shoreline(inputs):

    sitename = inputs['sitename']
    filepath_data = inputs['filepath']

    # check if reference shoreline already exists in the corresponding folder
    filepath = os.path.join(filepath_data, sitename)
    filename = sitename + '_reference_shoreline.pkl'
    # if it exist, load it and return it
    if filename in os.listdir(filepath):
        print('Reference shoreline already exists and was loaded')
        with open(os.path.join(filepath, sitename + '_reference_shoreline.pkl'), 'rb') as f:
            refsl = pickle.load(f)
        return refsl
    else:
        return np.zeros(1)

