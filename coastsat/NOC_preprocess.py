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


def preprocess_S2(filname, satname, cloud_mask_issue):

    # read 10m bands (R,G,B,NIR)
    filname10 = filname[0]

    data = gdal.Open(filname10, gdal.GA_ReadOnly)
    georef = np.array(data.GetGeoTransform())
    bands = [data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)]
    image_10m = np.stack(bands, 2)
    image_10m = image_10m / 10000  # TOA scaled to 10000

    print(f'@@@   {image_10m.shape}')

    # if image contains only zeros (can happen with S2), skip the image
    if sum(sum(sum(image_10m))) < 1:
        im_ms = []
        georef = []
        # skip the image by giving it a full cloud_mask
        cloud_mask = np.ones((image_10m.shape[0], image_10m.shape[1])).astype('bool')
        return im_ms, georef, cloud_mask, [], [], []

    # size of 10m bands
    nrows = image_10m.shape[0]
    ncols = image_10m.shape[1]

    # read 20m band (SWIR1)
    filname20 = filname[1]
    data = gdal.Open(filname20, gdal.GA_ReadOnly)
    image_20m = data.GetRasterBand(1).ReadAsArray()
    image_20m = image_20m / 10000  # TOA scaled to 10000

    image_SWIR = np.copy(image_20m)

    # resize the SWIR using bi-linear interpolation (order 1)
    image_SWIR = transform.resize(image_SWIR, (nrows, ncols),
                                  order=1, preserve_range=True,
                                  mode='constant')
    
    # pansharpen SWIR
    image_SWIR_ps = pansharpen_SWIR(image_SWIR, image_10m)

    print(f'@@@   {image_SWIR.shape}')

    image_SWIR_ps = np.expand_dims(image_SWIR_ps, axis=2)

    print(f'@@@   {image_SWIR.shape}')

    # add pansharpened SWIR band to the other 10m bands
    image_10m = np.append(image_10m, image_SWIR_ps, axis=2)

    print(f'@@@   {image_10m.shape}')

    # update cloud mask with all the nodata pixels
    cloud_mask = np.zeros((nrows, ncols), dtype=bool)

    return image_10m, georef, cloud_mask, image_20m, cloud_mask, cloud_mask



def pansharpen_SWIR(image_SWIR, image_ms):

    print()
    print(f'@@@ {image_SWIR.shape}')
    print(f'@@@ {image_ms.shape}')
    
    # reshape image into vector
    image_vec = image_ms.reshape(image_ms.shape[0] * image_ms.shape[1], image_ms.shape[2])

    print(f'@@@ {image_vec.shape}')

    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pca = pca.fit_transform(image_vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_swir = image_SWIR.reshape(image_ms.shape[0] * image_ms.shape[1])
    vec_pca[:, 0] = hist_match(vec_swir, vec_pca[:, 0])
    vec_SWIR_ps = pca.inverse_transform(vec_pca)

    # reshape vector into image

    image_SWIR_ps = vec_SWIR_ps.reshape(image_SWIR.shape[0], image_SWIR.shape[1], image_ms.shape[2])

    return image_SWIR_ps
