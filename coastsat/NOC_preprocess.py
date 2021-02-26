from coastsat.SDS_preprocess import *

from utils.print_utils import printWarning, printProgress


def preprocess_sar(file_name):

    data = gdal.Open(file_name, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    sar_stack = np.stack(bands, 2)

    return sar_stack, georef


def get_reference_shoreline_median(inputs):

    site_name = inputs['site_name']
    median_dir_path = inputs['median_dir_path']

    # check if reference shoreline already exists in the corresponding folder
    file_name = site_name + '_reference_shoreline.pkl'
    # if it exist, load it and return it
    if file_name in os.listdir(median_dir_path):
        printProgress('reference shoreline loaded')
        with open(os.path.join(median_dir_path, site_name + '_reference_shoreline.pkl'), 'rb') as f:
            refsl = pickle.load(f)
        return refsl
    else:
        printWarning('no reference shoreline found')
        return np.zeros(1)


def preprocess_single(file_path, satname, cloud_mask_issue, pansharpen=False):

    # read 10m bands (R,G,B,NIR)
    file_path_10 = file_path[0]

    data = gdal.Open(file_path_10, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    image_10 = np.stack(bands, 2)
    image_10 = image_10 / 10000  # TOA scaled to 10000

    # if image contains only zeros (can happen with S2), skip the image
    if sum(sum(sum(image_10))) < 1:
        image_ms = []
        georef = []
        # skip the image by giving it a full cloud_mask
        cloud_mask = np.ones((image_10.shape[0], image_10.shape[1])).astype('bool')
        return image_ms, georef, cloud_mask, [], [], []

    # size of 10m bands
    nrows = image_10.shape[0]
    ncols = image_10.shape[1]

    # read 20m band (SWIR1)
    file_path_20 = file_path[1]
    data = gdal.Open(file_path_20, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    image_20 = np.stack(bands, 2)
    image_20 = image_20 / 10000  # TOA scaled to 10000

    if pansharpen:
        printProgress('pansharpening SWIR2')

        image_20m = transform.resize(image_20, (nrows, ncols),
                                      order=1, preserve_range=True,
                                      mode='constant')
        image_NIR = image_10[:,:,3]
        image_20m_ps = pansharpen_SWIR(image_20m, image_NIR)
        image_SWIR = image_20m_ps[:, :, 5]
    else:
        image_SWIR = image_20[:, :, 5]

    # resize the image using bilinear interpolation (order 1)
    image_SWIR = transform.resize(image_SWIR, (nrows, ncols), order=1, preserve_range=True,
                               mode='constant')
    image_SWIR = np.expand_dims(image_SWIR, axis=2)

    # append down-sampled SWIR band to the other 10m bands
    image_ms = np.append(image_10, image_SWIR, axis=2)

    # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
    file_path_60 = file_path[2]
    data = gdal.Open(file_path_60, gdal.GA_ReadOnly)
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    image_60 = np.stack(bands, 2)
    image_QA = image_60[:, :, 0]
    cloud_mask = create_cloud_mask(image_QA, satname, cloud_mask_issue)
    # resize the cloud mask using nearest neighbour interpolation (order 0)
    cloud_mask = transform.resize(cloud_mask, (nrows, ncols), order=0, preserve_range=True,
                                  mode='constant')
    # check if -inf or nan values on any band and create nodata image
    image_nodata = np.zeros(cloud_mask.shape).astype(bool)
    for k in range(image_ms.shape[2]):
        image_inf = np.isin(image_ms[:, :, k], -np.inf)
        image_nan = np.isnan(image_ms[:, :, k])
        image_nodata = np.logical_or(np.logical_or(image_nodata, image_inf), image_nan)
    # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
    # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
    image_zeros = np.ones(image_nodata.shape).astype(bool)
    image_zeros = np.logical_and(np.isin(image_ms[:, :, 1], 0), image_zeros)  # Green
    image_zeros = np.logical_and(np.isin(image_ms[:, :, 3], 0), image_zeros)  # NIR
    image_SWIR_zeros = transform.resize(np.isin(image_SWIR.flatten(), 0), (nrows, ncols), order=0,
                                   preserve_range=True, mode='constant').astype(bool)
    image_zeros = np.logical_and(image_SWIR_zeros, image_zeros)  # SWIR1
    # add to image_nodata
    image_nodata = np.logical_or(image_zeros, image_nodata)
    # dilate if image was merged as there could be issues at the edges
    if 'merged' in file_path_10:
        image_nodata = morphology.dilation(image_nodata, morphology.square(5))

    # update cloud mask with all the nodata pixels
    cloud_mask = np.logical_or(cloud_mask, image_nodata)

    # the extra image is the 20m SWIR band
    image_extra = image_20

    return image_ms, georef, cloud_mask, image_extra, image_QA, image_nodata


def pansharpen_SWIR(image_20m, image_NIR):
    
    # reshape image into vector
    image_vec = image_20m.reshape(image_20m.shape[0] * image_20m.shape[1], image_20m.shape[2])

    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pca = pca.fit_transform(image_vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_nir = image_NIR.reshape(image_NIR.shape[0] * image_NIR.shape[1])
    vec_pca[:, 0] = hist_match(vec_nir, vec_pca[:, 0])
    vec_20m_ps = pca.inverse_transform(vec_pca)

    # reshape vector into image
    image_20m_ps = vec_20m_ps.reshape(image_20m.shape[0], image_20m.shape[1], image_20m.shape[2])

    return image_20m_ps
